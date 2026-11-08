"""Tests for the faster live pipeline: in-memory audio, per-source workers, merging of
waiting audio, committing finished text, event-driven suggestions and in-place UI rows."""

import datetime
import io
import os
import sys
import threading
import time
import types
import unittest
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sdk import audio_format
from sdk import transcriber_models as tm
from sdk.transcription_result import TranscriptSegment, TranscriptionHypothesis
from app.transcribe import audio_transcriber as at
from app.transcribe.live_transcription import LiveTranscriptManager

BPS = audio_format.BYTES_PER_SECOND
T0 = datetime.datetime(2026, 11, 1, 12, 0, 0)


def pcm(seconds: float, value: int = 1000) -> bytes:
    return np.full(int(seconds * 16000), value, dtype="<i2").tobytes()


# ------------------------------------------------------------------------------------------
# Audio conversion
# ------------------------------------------------------------------------------------------
class TestAudioFormat(unittest.TestCase):
    def test_48k_stereo_becomes_16k_mono(self):
        frames = 48000
        left = np.full(frames, 8000, dtype=np.int16)
        right = np.full(frames, 4000, dtype=np.int16)
        data = np.column_stack([left, right]).astype("<i2").tobytes()

        out = np.frombuffer(audio_format.to_pcm16_mono_16k(data, 48000, 2, 2), dtype="<i2")

        self.assertEqual(out.size, 16000)
        self.assertTrue(np.all(np.abs(out - 6000) <= 1))       # channels averaged

    def test_44100_keeps_length_and_pitch(self):
        t = np.arange(44100) / 44100
        tone = (np.sin(2 * np.pi * 440 * t) * 12000).astype("<i2").tobytes()

        out = np.frombuffer(audio_format.to_pcm16_mono_16k(tone, 44100, 2, 1), dtype="<i2").astype(float)

        self.assertEqual(out.size, 16000)
        spectrum = np.abs(np.fft.rfft(out))
        peak_hz = np.argmax(spectrum) * 16000 / out.size
        self.assertAlmostEqual(peak_hz, 440, delta=2)

    def test_sound_above_8khz_does_not_fold_back_into_speech(self):
        for rate in (48000, 44100):
            t = np.arange(rate) / rate
            hiss = (np.sin(2 * np.pi * 11000 * t) * 16000).astype("<i2").tobytes()
            out = np.frombuffer(audio_format.to_pcm16_mono_16k(hiss, rate, 2, 1), dtype="<i2").astype(float)
            self.assertLess(np.sqrt(np.mean(out[500:-500] ** 2)), 120)     # was ~5000 (-9 dB) before
            speech = (np.sin(2 * np.pi * 1000 * t) * 16000).astype("<i2").tobytes()
            kept = np.frombuffer(audio_format.to_pcm16_mono_16k(speech, rate, 2, 1), dtype="<i2").astype(float)
            self.assertGreater(np.sqrt(np.mean(kept[500:-500] ** 2)), 16000 / np.sqrt(2) * 0.95)

    def test_16k_mono_passes_through_unchanged(self):
        data = np.arange(-500, 500, dtype="<i2").tobytes()
        self.assertEqual(audio_format.to_pcm16_mono_16k(data, 16000, 2, 1), data)

    def test_partial_frames_and_empty_input(self):
        self.assertEqual(audio_format.to_pcm16_mono_16k(b"", 48000, 2, 2), b"")
        out = audio_format.to_pcm16_mono_16k(b"\x01\x00\x02\x00\x03", 16000, 2, 1)
        self.assertEqual(len(out), 4)                            # odd byte dropped

    def test_8_bit_unsigned_input(self):
        data = bytes([128 + 64] * 16000)
        out = np.frombuffer(audio_format.to_pcm16_mono_16k(data, 16000, 1, 1), dtype="<i2")
        self.assertTrue(np.all(np.abs(out - 16383) <= 1))

    def test_unsupported_width_is_reported(self):
        with self.assertRaises(ValueError):
            audio_format.to_pcm16_mono_16k(b"\x00" * 6, 16000, 3, 1)

    def test_wav_bytes_round_trip(self):
        data = pcm(0.5, 1234)
        with wave.open(io.BytesIO(audio_format.pcm16_to_wav_bytes(data)), "rb") as wav_file:
            self.assertEqual((wav_file.getframerate(), wav_file.getnchannels(), wav_file.getsampwidth()),
                             (16000, 1, 2))
            self.assertEqual(wav_file.readframes(wav_file.getnframes()), data)
        floats = audio_format.pcm16_to_float32(data)
        self.assertEqual(floats.dtype, np.float32)
        self.assertAlmostEqual(float(floats[0]), 1234 / 32768, places=6)


class TestRecorderConversion(unittest.TestCase):
    def test_phrases_are_queued_as_16k_mono(self):
        from sdk.audio_recorder import BaseRecorder
        import queue

        class Recorder(BaseRecorder):
            def get_name(self):
                return "test"

        recorder = Recorder.__new__(Recorder)
        recorder.source = SimpleNamespace(SAMPLE_RATE=48000, SAMPLE_WIDTH=2, channels=2, stream=None)
        recorder.source_name = "Speaker"
        recorder.enabled = True
        recorder.audio_file_name = None
        recorder.config = {"General": {"stt": "whisper", "transcript_audio_duration_seconds": 3}}
        captured = {}
        recorder.recorder = SimpleNamespace(
            listen_in_background=lambda **kwargs: captured.update(kwargs) or "stop")
        audio_queue = queue.Queue()

        self.assertEqual(recorder.record_audio(audio_queue), "stop")
        stereo = np.zeros((48000, 2), dtype="<i2").tobytes()
        captured["callback"](None, SimpleNamespace(get_raw_data=lambda: stereo, sample_rate=48000, sample_width=2))

        source, data, _ = audio_queue.get_nowait()
        self.assertEqual(source, "Speaker")
        self.assertEqual(len(data), BPS)                          # one second at 16 kHz mono
        self.assertEqual(recorder._saved_format, (16000, 2, 1))


# ------------------------------------------------------------------------------------------
# Transcriber
# ------------------------------------------------------------------------------------------
class FakeConversation:
    """Keeps rows per persona the way Conversation does (insert or replace the newest)."""

    def __init__(self):
        self.rows = {"You": [], "Speaker": []}
        self.lock = threading.Lock()
        self.cleared = 0

    def update_conversation(self, persona, time_spoken, text, update_previous=False):
        with self.lock:
            rows = self.rows[persona]
            if update_previous and rows:
                rows[-1] = text
            else:
                rows.append(text)

    def clear_conversation_data(self):
        with self.lock:
            self.rows = {"You": [], "Speaker": []}
            self.cleared += 1

    def get_conversation(self, sources=None, length=0):
        return ""


class ScriptedModel:
    """Speech to text double: one word per second of audio, with segment timestamps.

    The word for second N of a phrase is 'wN' (N counted from the phrase start), so the
    test can check that every word appears exactly once across rows.
    """

    def __init__(self, delay=0.0, words_per_segment=2, timestamps=True):
        self.delay = delay
        self.words_per_segment = words_per_segment
        self.timestamps = timestamps
        self.calls = []        # seconds of audio per request
        self.offsets = []
        self.lock = threading.Lock()
        self.block = None      # threading.Event to hold a request

    def transcribe_pcm(self, data, sample_rate):
        with self.lock:
            self.calls.append(len(data) / BPS)
        if self.block is not None:
            self.block.wait(5)
        if self.delay:
            time.sleep(self.delay)
        return {"seconds": len(data) / BPS}

    def process_response(self, response):
        return "x" if response["seconds"] >= 1 else ""

    def normalize_response(self, response, audio_start_seconds=0.0, audio_end_seconds=0.0):
        first = int(round(audio_start_seconds))
        last = int(audio_end_seconds)
        words = [(second, f"w{second}") for second in range(first, last)]
        segments = []
        for index in range(0, len(words), self.words_per_segment):
            chunk = words[index:index + self.words_per_segment]
            segments.append(TranscriptSegment(id=len(segments), start_seconds=float(chunk[0][0]),
                                              end_seconds=float(chunk[-1][0] + 1),
                                              text=" ".join(word for _, word in chunk)))
        text = " ".join(word for _, word in words)
        if not self.timestamps:
            segments = [TranscriptSegment(id=0, start_seconds=audio_start_seconds,
                                          end_seconds=audio_end_seconds, text=text)]
        return TranscriptionHypothesis(provider="fake", text=text, segments=segments,
                                       audio_start_seconds=audio_start_seconds,
                                       audio_end_seconds=audio_end_seconds)

    def set_lang(self, lang):
        pass


def make_transcriber(model, commit=10.0, window=30.0):
    convo = FakeConversation()
    transcriber = at.WhisperTranscriber(
        None, None, model, convo=convo,
        config={"General": {
            "clear_transcript_periodically": False,
            "clear_transcript_interval_seconds": 90,
            "live_transcription_window_seconds": window,
            "live_transcription_mutable_tail_seconds": 5,
            "live_transcription_stability_passes": 2,
            "live_transcription_commit_seconds": commit,
        }})
    return transcriber, convo


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


class TestTranscriberPipeline(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(at, "PHRASE_TIMEOUT", 0.4)
        patcher.start()
        self.addCleanup(patcher.stop)

    def feed(self, transcriber, source, chunks, start=T0, step=1.0, pause=0.0):
        """Queue chunks as captured phrases; time_spoken advances by `step` seconds each."""
        for index, chunk in enumerate(chunks):
            transcriber._source_queues[source].put((chunk, start + datetime.timedelta(seconds=step * index)))
            if pause:
                time.sleep(pause)

    def test_waiting_audio_is_merged_into_one_request(self):
        model = ScriptedModel()
        model.block = threading.Event()
        transcriber, convo = make_transcriber(model, commit=0)
        transcriber._start_workers()

        self.feed(transcriber, "Speaker", [pcm(1)])
        self.assertTrue(wait_for(lambda: len(model.calls) == 1))
        self.feed(transcriber, "Speaker", [pcm(1)] * 5, start=T0 + datetime.timedelta(seconds=0.2), step=0.05)
        model.block.set()

        self.assertTrue(wait_for(lambda: len(model.calls) == 2))
        time.sleep(0.6)
        self.assertEqual(model.calls, [1.0, 6.0])                # 6 phrases, 2 requests
        self.assertEqual(convo.rows["Speaker"], ["w0 w1 w2 w3 w4 w5"])

    def test_slow_speaker_does_not_hold_back_microphone(self):
        model = ScriptedModel()
        model.thread_safe = True          # a network engine: both sources may run at once
        speaker_started = threading.Event()
        release = threading.Event()
        original = model.transcribe_pcm

        def transcribe(data, sample_rate):
            if threading.current_thread().name.endswith("Speaker"):
                speaker_started.set()
                release.wait(5)
            return original(data, sample_rate)

        model.transcribe_pcm = transcribe
        transcriber, convo = make_transcriber(model)
        transcriber._start_workers()

        self.feed(transcriber, "Speaker", [pcm(2)])
        self.assertTrue(speaker_started.wait(2))
        self.feed(transcriber, "You", [pcm(2)])

        self.assertTrue(wait_for(lambda: convo.rows["You"] == ["w0 w1"]))
        self.assertEqual(convo.rows["Speaker"], [])
        release.set()
        self.assertTrue(wait_for(lambda: convo.rows["Speaker"] == ["w0 w1"]))

    def test_long_speech_is_committed_and_requests_stay_short(self):
        model = ScriptedModel()
        transcriber, convo = make_transcriber(model, commit=10)
        transcriber._start_workers()

        # 40 seconds of non-stop speech in 2-second phrases, fed in real order (queued less
        # than PHRASE_TIMEOUT apart, which is what continuous speech looks like).
        for index in range(20):
            self.feed(transcriber, "Speaker", [pcm(2)], start=T0 + datetime.timedelta(seconds=0.3 * index))
            wait_for(lambda: not transcriber._source_queues["Speaker"].qsize()
                     and not transcriber.audio_sources_properties["Speaker"]["dirty"])
        # After the speaker stops, the open tail is transcribed on its own.
        self.assertTrue(wait_for(lambda: not transcriber.audio_sources_properties["Speaker"]["dirty"], 3))
        time.sleep(0.6)

        words = " ".join(convo.rows["Speaker"]).split()
        self.assertEqual(words, [f"w{second}" for second in range(40)])   # nothing lost, nothing twice
        self.assertGreater(len(convo.rows["Speaker"]), 2)                  # finished text was committed
        self.assertLessEqual(max(model.calls), 12.0)                       # never the full 40 s / 30 s window

    def test_without_timestamps_the_window_cap_still_applies(self):
        model = ScriptedModel(timestamps=False)
        transcriber, convo = make_transcriber(model, commit=10, window=8)
        transcriber._start_workers()
        for index in range(10):
            self.feed(transcriber, "You", [pcm(2)], start=T0 + datetime.timedelta(seconds=0.3 * index))
            wait_for(lambda: not transcriber._source_queues["You"].qsize()
                     and not transcriber.audio_sources_properties["You"]["dirty"])
        self.assertLessEqual(max(model.calls), 8.0)
        self.assertTrue(convo.rows["You"])

    def test_new_phrase_with_empty_first_result_does_not_replace_previous_row(self):
        model = ScriptedModel()
        transcriber, convo = make_transcriber(model)
        transcriber._start_workers()

        self.feed(transcriber, "Speaker", [pcm(2)])
        self.assertTrue(wait_for(lambda: convo.rows["Speaker"] == ["w0 w1"]))
        # A new phrase (gap > PHRASE_TIMEOUT) whose first piece is too short to give text ...
        later = T0 + datetime.timedelta(seconds=10)
        self.feed(transcriber, "Speaker", [pcm(0.5)], start=later)
        self.assertTrue(wait_for(lambda: len(model.calls) == 2))
        time.sleep(0.1)
        # ... and then more of it, within the same phrase.
        self.feed(transcriber, "Speaker", [pcm(1.5)], start=later + datetime.timedelta(seconds=0.2))
        self.assertTrue(wait_for(lambda: len(convo.rows["Speaker"]) == 2))
        self.assertEqual(convo.rows["Speaker"][0], "w0 w1")

    def test_clear_while_a_request_runs_discards_its_result(self):
        model = ScriptedModel()
        model.block = threading.Event()
        transcriber, convo = make_transcriber(model)
        transcriber._start_workers()

        self.feed(transcriber, "You", [pcm(2)])
        self.assertTrue(wait_for(lambda: len(model.calls) == 1))
        transcriber.clear_transcriber_context(MagicMock(mutex=threading.Lock(), queue=[]))
        model.block.set()
        time.sleep(0.3)

        self.assertEqual(convo.rows["You"], [])
        self.assertEqual(convo.cleared, 1)

    def test_worker_survives_a_failing_request(self):
        model = ScriptedModel()
        original = model.transcribe_pcm
        failures = [RuntimeError("network down")]

        def transcribe(data, sample_rate):
            if failures:
                raise failures.pop()
            return original(data, sample_rate)

        model.transcribe_pcm = transcribe
        transcriber, convo = make_transcriber(model)
        transcriber._start_workers()

        self.feed(transcriber, "You", [pcm(2)])
        time.sleep(0.2)
        self.feed(transcriber, "You", [pcm(2)], start=T0 + datetime.timedelta(seconds=10))
        self.assertTrue(wait_for(lambda: convo.rows["You"] == ["w0 w1"]))

    def test_dispatcher_routes_by_source(self):
        model = ScriptedModel()
        transcriber, convo = make_transcriber(model)
        import queue
        audio_queue = queue.Queue()
        threading.Thread(target=transcriber.transcribe_audio_queue, args=(audio_queue,), daemon=True).start()
        audio_queue.put(("Speaker", pcm(2), T0))
        audio_queue.put(("Nobody", pcm(2), T0))
        audio_queue.put(("You", pcm(3), T0))
        self.assertTrue(wait_for(lambda: convo.rows == {"You": ["w0 w1 w2"], "Speaker": ["w0 w1"]}))

    def test_paused_transcription_keeps_nothing(self):
        model = ScriptedModel()
        transcriber, convo = make_transcriber(model)
        transcriber.set_transcription_enabled(False)
        transcriber._start_workers()
        self.feed(transcriber, "You", [pcm(2)])
        time.sleep(0.3)
        self.assertEqual(model.calls, [])
        self.assertEqual(convo.rows["You"], [])


class TestTranscriberSafety(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(at, "PHRASE_TIMEOUT", 0.4)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_language_names_reach_the_engine_as_codes(self):
        model = ScriptedModel()
        model.set_lang = MagicMock()
        transcriber, _ = make_transcriber(model)
        for given, expected in (("English", "en"), ("german", "de"), ("fa", "fa"), ("Klingon", "Klingon")):
            transcriber.set_language(given)
            self.assertEqual(model.set_lang.call_args.args[0], expected)

    def test_a_local_engine_never_runs_two_transcriptions_at_once(self):
        model = ScriptedModel(delay=0.15)
        running, peak = [0], [0]
        original = model.transcribe_pcm

        def transcribe(data, sample_rate):
            running[0] += 1
            peak[0] = max(peak[0], running[0])
            try:
                return original(data, sample_rate)
            finally:
                running[0] -= 1

        model.transcribe_pcm = transcribe
        transcriber, convo = make_transcriber(model)          # ScriptedModel is not thread_safe
        transcriber._start_workers()
        transcriber._source_queues["You"].put((pcm(2), T0))
        transcriber._source_queues["Speaker"].put((pcm(2), T0))
        self.assertTrue(wait_for(lambda: convo.rows["You"] and convo.rows["Speaker"]))
        self.assertEqual(peak[0], 1)

    def test_network_engines_run_both_sources_at_once(self):
        model = ScriptedModel(delay=0.3)
        model.thread_safe = True
        transcriber, convo = make_transcriber(model)
        transcriber._start_workers()
        started = time.monotonic()
        transcriber._source_queues["You"].put((pcm(2), T0))
        transcriber._source_queues["Speaker"].put((pcm(2), T0))
        self.assertTrue(wait_for(lambda: convo.rows["You"] and convo.rows["Speaker"]))
        self.assertLess(time.monotonic() - started, 0.55)

    def test_audio_taken_before_a_clear_is_not_written_afterwards(self):
        model = ScriptedModel()
        gate = threading.Event()
        calls = []
        original = model.transcribe_pcm

        def transcribe(data, sample_rate):
            calls.append(len(data))
            if len(calls) == 2:
                gate.wait(5)
            return original(data, sample_rate)

        model.transcribe_pcm = transcribe
        transcriber, convo = make_transcriber(model)
        transcriber._start_workers()
        source_queue = transcriber._source_queues["Speaker"]
        source_queue.put((pcm(2), T0))
        self.assertTrue(wait_for(lambda: convo.rows["Speaker"] == ["w0 w1"]))
        transcriber.clear_transcript_data()
        self.assertEqual(convo.rows["Speaker"], [])
        # One batch: more of the same phrase, then a new phrase. The worker transcribes the first
        # (held at the gate) before it would start the second; a clear comes meanwhile.
        with source_queue.mutex:
            source_queue.queue.extend([(pcm(1), T0 + datetime.timedelta(seconds=0.1)),
                                       (pcm(3), T0 + datetime.timedelta(seconds=10))])
            source_queue.unfinished_tasks += 2
            source_queue.not_empty.notify()
        self.assertTrue(wait_for(lambda: len(calls) == 2))
        transcriber.clear_transcript_data()
        gate.set()
        time.sleep(0.8)
        self.assertEqual(len(calls), 2)                    # the new phrase was dropped, not transcribed
        self.assertEqual(convo.rows["Speaker"], [])

    def test_a_row_that_is_all_open_tail_is_not_committed(self):
        manager = LiveTranscriptManager({"General": {}})
        manager.process_hypothesis("You", TranscriptionHypothesis("t", "still talking"), new_phrase=True)
        self.assertIsNone(manager.commit("You", "still talking"))
        self.assertEqual(manager.commit("You", "talking"), "still")


class TestLiveTranscriptCommit(unittest.TestCase):
    def manager(self):
        return LiveTranscriptManager({"General": {}})

    def test_commit_removes_the_open_tail(self):
        manager = self.manager()
        manager.process_hypothesis("You", TranscriptionHypothesis("t", "one two three four"), new_phrase=True)
        self.assertEqual(manager.commit("You", "three four"), "one two")
        self.assertIsNone(manager.commit("You", ""))              # state was closed

    def test_commit_refuses_when_the_row_does_not_end_with_the_tail(self):
        manager = self.manager()
        manager.process_hypothesis("You", TranscriptionHypothesis("t", "one two three"), new_phrase=True)
        self.assertIsNone(manager.commit("You", "four"))
        self.assertEqual(manager.commit("You", "Three."), "one two")   # punctuation and case ignored

    def test_languages_without_spaces_can_be_committed(self):
        manager = self.manager()
        manager.process_hypothesis("You", TranscriptionHypothesis("t", "你好世界。今天天气很好"), new_phrase=True)
        self.assertIsNone(manager.commit("You", "明天"))
        self.assertEqual(manager.commit("You", "今天天气很好"), "你好世界。")

    def test_vowel_signs_and_joiners_stay_inside_words(self):
        self.assertEqual(LiveTranscriptManager._normalize_for_match("किताब"), "किताब")
        self.assertEqual(LiveTranscriptManager._normalize_for_match("می\u200cخواهم!"), "می\u200cخواهم")

    def test_words_of_other_scripts_are_compared_by_their_letters(self):
        manager = self.manager()
        manager.process_hypothesis("You", TranscriptionHypothesis("t", "سلام دنیا خوبی"), new_phrase=True)
        self.assertIsNone(manager.commit("You", "چطوری"))         # was treated as equal before
        self.assertEqual(manager.commit("You", "خوبی"), "سلام دنیا")
        self.assertEqual(LiveTranscriptManager._normalize_for_match("Größe!"), "größe")


# ------------------------------------------------------------------------------------------
# Speech to text models
# ------------------------------------------------------------------------------------------
class TestModelsTakeAudioFromMemory(unittest.TestCase):
    def test_api_whisper_uploads_wav_from_memory_with_language_and_timestamps(self):
        model = tm.APIWhisperSTTModel({"api_key": "k", "timeout": 7, "audio_lang": "de"})
        segment = SimpleNamespace(id=0, start=0.0, end=1.5, text=" Hallo")
        response = SimpleNamespace(text=" Hallo ", segments=[segment])
        model.stt_client = MagicMock()
        model.stt_client.audio.transcriptions.create.return_value = response

        self.assertIs(model.transcribe_pcm(pcm(1)), response)

        kwargs = model.stt_client.audio.transcriptions.create.call_args.kwargs
        name, wav_bytes, content_type = kwargs["file"]
        self.assertEqual((name, content_type), ("audio.wav", "audio/wav"))
        self.assertTrue(wav_bytes.startswith(b"RIFF"))
        self.assertEqual((kwargs["model"], kwargs["language"], kwargs["response_format"], kwargs["timeout"]),
                         ("whisper-1", "de", "verbose_json", 7))

        hypothesis = model.normalize_response(response, audio_start_seconds=10, audio_end_seconds=11)
        self.assertEqual(hypothesis.text, "Hallo")
        self.assertEqual((hypothesis.segments[0].start_seconds, hypothesis.segments[0].end_seconds), (10, 11.5))

    def test_api_models_without_timestamps_get_plain_json(self):
        model = tm.APIWhisperSTTModel({"api_key": "k", "timeout": 7, "audio_lang": "en",
                                       "model": "gpt-4o-mini-transcribe"})
        model.stt_client = MagicMock()
        model.stt_client.audio.transcriptions.create.return_value = SimpleNamespace(text="hi")
        model.transcribe_pcm(pcm(1))
        kwargs = model.stt_client.audio.transcriptions.create.call_args.kwargs
        self.assertNotIn("response_format", kwargs)
        hypothesis = model.normalize_response(SimpleNamespace(text="hi"), 0, 1)
        self.assertEqual(len(hypothesis.segments), 1)

    def test_api_never_sends_a_language_name(self):
        model = tm.APIWhisperSTTModel({"api_key": "k", "timeout": 7, "audio_lang": "English"})
        model.stt_client = MagicMock()
        model.stt_client.audio.transcriptions.create.return_value = SimpleNamespace(text="x", segments=[])
        model.transcribe_pcm(pcm(1))
        self.assertNotIn("language", model.stt_client.audio.transcriptions.create.call_args.kwargs)

    def test_api_failure_returns_empty(self):
        model = tm.APIWhisperSTTModel({"api_key": "k", "timeout": 7, "audio_lang": "en"})
        model.stt_client = MagicMock()
        model.stt_client.audio.transcriptions.create.side_effect = RuntimeError("offline")
        self.assertEqual(model.transcribe_pcm(pcm(1)), "")

    def test_faster_whisper_is_used_when_installed(self):
        calls = {}

        class FakeWhisperModel:
            def __init__(self, name, **kwargs):
                calls["init"] = (name, kwargs)

            def transcribe(self, audio, **kwargs):
                calls["audio"] = audio
                calls["kwargs"] = kwargs
                segments = iter([SimpleNamespace(start=0.0, end=1.0, text=" hello"),
                                 SimpleNamespace(start=1.0, end=2.0, text=" world")])
                return segments, SimpleNamespace(language="en")

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeWhisperModel
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": None}):
            model = tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en",
                                        "local_engine": "auto"})
            result = model.transcribe_pcm(pcm(2, 3276))

        self.assertEqual(model.engine, tm.FASTER_WHISPER)
        self.assertEqual(calls["init"][0], "base")
        self.assertEqual((calls["init"][1]["device"], calls["init"][1]["compute_type"]), ("cpu", "int8"))
        self.assertEqual(calls["audio"].dtype, np.float32)
        self.assertEqual(calls["audio"].size, 32000)
        self.assertFalse(calls["kwargs"]["condition_on_previous_text"])
        self.assertEqual(result["text"], "hello world")
        hypothesis = model.normalize_response(result, audio_start_seconds=5, audio_end_seconds=7)
        self.assertEqual([s.end_seconds for s in hypothesis.segments], [6.0, 7.0])

    def _fake_faster(self, fail_gpu_load=False, fail_gpu_run=False, fail_any_load=False):
        made = []

        class FakeWhisperModel:
            def __init__(self, name, **kwargs):
                if fail_any_load:
                    raise OSError("no internet")
                if fail_gpu_load and kwargs["device"] == "cuda":
                    raise ValueError("float16 not supported")
                self.device = kwargs["device"]
                made.append(kwargs)

            def transcribe(self, audio, **kwargs):
                if fail_gpu_run and self.device == "cuda":
                    raise RuntimeError("cublas64_12.dll not found")
                return iter([SimpleNamespace(start=0.0, end=1.0, text=" ok")]), SimpleNamespace(language="en")

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeWhisperModel
        gpu = types.ModuleType("ctranslate2")
        gpu.get_cuda_device_count = lambda: 1
        return fake, gpu, made

    def test_faster_whisper_falls_back_to_cpu_when_the_gpu_cannot_load(self):
        fake, gpu, made = self._fake_faster(fail_gpu_load=True)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": gpu}):
            model = tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en"})
        self.assertEqual([m["device"] for m in made], ["cpu"])
        self.assertTrue(model.thread_safe)
        self.assertEqual(made[0]["num_workers"], 2)

    def test_faster_whisper_moves_to_cpu_when_cuda_libraries_are_missing(self):
        fake, gpu, made = self._fake_faster(fail_gpu_run=True)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": gpu}):
            model = tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en"})
            self.assertTrue(model.use_gpu)
            self.assertEqual(model.transcribe_pcm(pcm(1))["text"], "ok")
        self.assertFalse(model.use_gpu)
        self.assertEqual([m["device"] for m in made], ["cuda", "cpu"])

    def test_both_workers_failing_on_the_gpu_load_the_cpu_model_once(self):
        fake, gpu, made = self._fake_faster(fail_gpu_run=True)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": gpu}):
            model = tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en"})
            results = []
            workers = [threading.Thread(target=lambda: results.append(model.transcribe_pcm(pcm(1))))
                       for _ in range(2)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(5)
        self.assertEqual([r["text"] for r in results], ["ok", "ok"])
        self.assertEqual([m["device"] for m in made], ["cuda", "cpu"])

    def test_faster_whisper_load_failure_uses_openai_whisper(self):
        fake, gpu, _ = self._fake_faster(fail_any_load=True)
        fake_whisper = types.ModuleType("whisper")
        fake_whisper.load_model = MagicMock(return_value=MagicMock())
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": gpu, "whisper": fake_whisper,
                                      "torch": fake_torch}), \
                patch.object(tm.WhisperSTTModel, "download_model"), \
                patch.object(tm, "_installed", return_value=True):     # the optional engine is installed
            model = tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en"})
        self.assertEqual(model.engine, tm.OPENAI_WHISPER)
        self.assertFalse(model.thread_safe)

    def test_openai_whisper_gets_an_array_and_no_file(self):
        fake_whisper = types.ModuleType("whisper")
        loaded = MagicMock()
        loaded.transcribe.return_value = {"text": " hi", "segments": []}
        fake_whisper.load_model = MagicMock(return_value=loaded)
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"whisper": fake_whisper, "torch": fake_torch}), \
                patch.object(tm.WhisperSTTModel, "download_model"):
            model = tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en",
                                        "local_engine": "openai-whisper"})
            model.transcribe_pcm(pcm(1))
            model.set_lang("de")

        audio = loaded.transcribe.call_args.args[0]
        self.assertIsInstance(audio, np.ndarray)
        self.assertFalse(loaded.transcribe.call_args.kwargs["fp16"])
        self.assertEqual(fake_whisper.load_model.call_count, 1)       # set_lang no longer reloads
        self.assertEqual(model.lang, "de")

    def test_engine_choice(self):
        def installed(*names):
            return patch.object(tm, "_installed", side_effect=lambda module: module in names)
        with installed("faster_whisper", "whisper"):
            self.assertEqual(tm.pick_local_whisper_engine("auto"), tm.FASTER_WHISPER)
        with installed("whisper"):                     # only the old engine (an old install)
            self.assertEqual(tm.pick_local_whisper_engine("auto"), tm.OPENAI_WHISPER)
        with installed():                              # nothing: faster-whisper, which reports clearly
            self.assertEqual(tm.pick_local_whisper_engine("auto"), tm.FASTER_WHISPER)
        self.assertEqual(tm.pick_local_whisper_engine("openai-whisper"), tm.OPENAI_WHISPER)
        self.assertEqual(tm.pick_local_whisper_engine("Faster-Whisper"), tm.FASTER_WHISPER)

    def test_requirements_no_longer_pull_in_pytorch(self):
        here = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as handle:
            lines = [line.split("#")[0].strip().lower() for line in handle]
        packages = [line for line in lines if line]
        self.assertTrue(any(p.startswith("faster-whisper") for p in packages))
        self.assertFalse(any(p.startswith(("openai-whisper", "torch")) for p in packages))
        setup = os.path.join(os.path.dirname(os.path.dirname(here)), "setup.bat")
        with open(setup, encoding="utf-8") as handle:
            self.assertNotIn("torch", handle.read())

    def test_missing_optional_openai_whisper_is_reported_clearly(self):
        with patch.dict(sys.modules, {"whisper": None, "torch": None}):
            with self.assertRaises(RuntimeError) as raised:
                tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en",
                                    "local_engine": "openai-whisper"})
        self.assertIn("setup-openai-whisper.bat", str(raised.exception))

    def test_first_download_failing_without_openai_whisper_is_reported_clearly(self):
        fake, gpu, _ = self._fake_faster(fail_any_load=True)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": gpu}), \
                patch.object(tm, "_installed", return_value=False):
            with self.assertRaises(RuntimeError) as raised:
                tm.WhisperSTTModel({"local_transcripton_model_file": "base", "audio_lang": "en"})
        self.assertIn("internet", str(raised.exception))

    def test_gpu_libraries_installed_with_pip_are_found_on_windows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as root:
            for package in ("cublas", "cudnn"):
                os.makedirs(os.path.join(root, "nvidia", package, "bin"))
            spec = SimpleNamespace(submodule_search_locations=[os.path.join(root, "nvidia")])
            added_dirs = []
            with patch.object(tm.sys, "platform", "win32"), \
                    patch("importlib.util.find_spec", return_value=spec), \
                    patch.object(tm.os, "add_dll_directory", side_effect=added_dirs.append, create=True), \
                    patch.dict(tm.os.environ, {"PATH": "x"}):
                added = tm.add_nvidia_dll_directories()
                path = tm.os.environ["PATH"]
        self.assertEqual(len(added), 2)
        self.assertEqual(added_dirs, added)
        self.assertTrue(all(folder in path for folder in added))
        with patch.object(tm.sys, "platform", "linux"):
            self.assertEqual(tm.add_nvidia_dll_directories(), [])

    def test_deepgram_sends_wav_bytes_from_memory(self):
        fake = types.ModuleType("deepgram")
        fake.DeepgramClient = MagicMock()
        fake.PrerecordedOptions = lambda **kwargs: kwargs
        with patch.dict(sys.modules, {"deepgram": fake}):
            model = tm.DeepgramSTTModel({"api_key": "k", "audio_lang": "en"})
            model.transcribe_pcm(pcm(1))
        transcribe_file = fake.DeepgramClient.return_value.listen.prerecorded.v.return_value.transcribe_file
        payload, options = transcribe_file.call_args.args
        self.assertTrue(payload["buffer"].startswith(b"RIFF"))
        self.assertEqual(options["model"], "nova-3")

    def test_whispercpp_cleans_up_its_files(self):
        import tempfile
        model = tm.WhisperCPPSTTModel.__new__(tm.WhisperCPPSTTModel)
        model.lang, model.model_filename = "en", "ggml-base.bin"
        seen = {}

        def fake_call(args, stdout, stderr, timeout=None):
            seen["timeout"] = timeout
            seen["args"] = args
            with open(args[1] + ".json", "w", encoding="utf-8") as handle:
                handle.write('{"transcription": [{"text": " hi", "offsets": {"from": 0, "to": 900}}]}')
            return 0

        with tempfile.TemporaryDirectory() as data_dir:
            os.makedirs(os.path.join(data_dir, "logs"))
            with patch.object(tm.WhisperCPPSTTModel, "_find_executable", return_value="whisper-cli.exe"), \
                    patch.object(tm.subprocess, "call", side_effect=fake_call), \
                    patch.object(tm.utilities, "get_data_path", return_value=data_dir):
                response = model.transcribe_pcm(pcm(1))
        wav_path = seen["args"][1]
        self.assertEqual(seen["args"][0], "whisper-cli.exe")
        self.assertEqual(seen["timeout"], 120)
        self.assertEqual(model.process_response(response), " hi")
        self.assertFalse(os.path.exists(wav_path))
        self.assertFalse(os.path.exists(wav_path + ".json"))

    def test_whispercpp_missing_executable_is_reported(self):
        model = tm.WhisperCPPSTTModel.__new__(tm.WhisperCPPSTTModel)
        model.lang, model.model_filename = "en", "ggml-base.bin"
        with patch.object(tm.WhisperCPPSTTModel, "_find_executable", return_value=None):
            self.assertIsNone(model.transcribe_pcm(pcm(1)))


# ------------------------------------------------------------------------------------------
# Conversation storage
# ------------------------------------------------------------------------------------------
class TestConversationStorage(unittest.TestCase):
    def setUp(self):
        from app.transcribe.core import conversation as core_conversation
        self.module = core_conversation
        self.table = MagicMock()
        self.table.get_max_convo_id.return_value = 41
        self.table.insert_conversation.side_effect = iter(range(100, 200))
        fake_db = MagicMock()
        fake_db.get_invocation_id.return_value = 1
        fake_db.get_object.return_value = self.table
        patcher = patch.object(core_conversation, "appdb", return_value=fake_db)
        patcher.start()
        self.addCleanup(patcher.stop)
        config = {"General": {"system_prompt": "p", "initial_convo": {
            "first": {"role": "You", "content": "hi"}, "second": {"role": "assistant", "content": "hey"}},
            "summary_prompt": "s"}, "OpenAI": {"response_lang": "English"}}
        with patch.object(core_conversation.configuration, "Config", return_value=SimpleNamespace(data=config)):
            self.convo = core_conversation.Conversation(context=SimpleNamespace())

    def test_streamed_assistant_tokens_do_not_touch_the_database(self):
        self.convo.update_conversation("assistant", "Hel", T0, update_previous=False)
        for text in ("Hello", "Hello wor", "Hello world"):
            self.convo.update_conversation("assistant", text, T0, update_previous=True)
        self.table.get_max_convo_id.assert_not_called()
        self.table.update_conversation.assert_not_called()
        self.table.insert_conversation.assert_not_called()

    def test_row_updates_use_the_remembered_id(self):
        inserts, updates = [], []
        self.convo.set_handlers(update=lambda *a: updates.append(a), insert=lambda *a: inserts.append(a))
        self.convo.update_conversation("Speaker", "one", T0)
        self.convo.update_conversation("Speaker", "one two", T0, update_previous=True)
        self.convo.update_conversation("Speaker", "one two three", T0, update_previous=True)
        self.table.get_max_convo_id.assert_not_called()
        self.assertEqual([c.args for c in self.table.update_conversation.call_args_list],
                         [(100, "one two"), (100, "one two three")])
        self.assertEqual(inserts, [("Speaker: [one]\n", "Speaker")])
        self.assertEqual(updates[-1], ("Speaker", "Speaker: [one two three]\n"))


# ------------------------------------------------------------------------------------------
# Suggestions
# ------------------------------------------------------------------------------------------
class TestEventDrivenResponses(unittest.TestCase):
    def make(self, interval):
        from app.transcribe.gpt_responder import GPTResponder
        responder = GPTResponder.__new__(GPTResponder)
        responder.enabled = True
        responder.llm_response_interval = interval
        responder.calls = []
        responder.generate_response_from_transcript = lambda: responder.calls.append(time.monotonic())
        transcriber = SimpleNamespace(transcript_changed_event=threading.Event())
        threading.Thread(target=responder.respond_to_transcriber, args=(transcriber,), daemon=True).start()
        return responder, transcriber.transcript_changed_event

    def test_a_change_is_answered_without_waiting_for_the_interval(self):
        responder, changed = self.make(interval=30)
        started = time.monotonic()
        changed.set()
        self.assertTrue(wait_for(lambda: responder.calls, 3))
        self.assertLess(responder.calls[0] - started, 1.5)

    def test_a_burst_of_changes_gets_one_answer_and_the_interval_is_kept(self):
        responder, changed = self.make(interval=1.5)
        for _ in range(5):
            changed.set()
            time.sleep(0.1)
        self.assertTrue(wait_for(lambda: len(responder.calls) == 1, 3))
        changed.set()
        self.assertTrue(wait_for(lambda: len(responder.calls) == 2, 4))
        self.assertGreaterEqual(responder.calls[1] - responder.calls[0], 1.45)

    def test_disabled_responder_sends_nothing(self):
        responder, changed = self.make(interval=0.1)
        responder.enabled = False
        changed.set()
        time.sleep(1.0)
        self.assertEqual(responder.calls, [])


# ------------------------------------------------------------------------------------------
# Display
# ------------------------------------------------------------------------------------------
class FakeTranscriptWidget:
    def __init__(self):
        self.rows = []          # [key, text]
        self.calls = []

    def upsert_keyed_row(self, key, text):
        self.calls.append(("upsert", key))
        for row in self.rows:
            if row[0] == key:
                row[1] = text
                return
        self.rows.append([key, text])

    def release_keyed_row(self, key):
        self.calls.append(("release", key))

    def add_text_to_bottom(self, text):
        self.calls.append(("add", text))
        self.rows.append([None, text])

    def delete_row_starting_with(self, start_text):
        self.calls.append(("delete", start_text))

    def replace_multiple_newlines(self):
        self.calls.append(("reflow",))

    def scroll_to_bottom(self):
        pass


class TestDisplayRows(unittest.TestCase):
    def setUp(self):
        from app.transcribe.desktop.display import DesktopDisplayManager
        self.manager = DesktopDisplayManager()
        self.widget = FakeTranscriptWidget()
        self.manager.bind_ui(SimpleNamespace(transcript_text=self.widget))

    def test_rows_are_updated_in_place_without_reflowing_the_transcript(self):
        self.manager.add_transcript_line("You: [hi]\n", "You")
        self.manager.add_transcript_line("Speaker: [how]\n", "Speaker")
        self.manager.update_last_row("You", "You: [hi there]\n")
        self.manager.update_last_row("Speaker", "Speaker: [how are you]\n")

        self.assertEqual([row[1] for row in self.widget.rows],
                         ["You: [hi there]\n", "Speaker: [how are you]\n"])   # order kept
        self.assertNotIn(("reflow",), self.widget.calls)

    def test_a_new_row_releases_the_previous_one(self):
        self.manager.add_transcript_line("You: [one]\n", "You")
        first_key = self.widget.rows[0][0]
        self.manager.add_transcript_line("You: [two]\n", "You")
        self.manager.update_last_row("You", "You: [two!]\n")
        self.assertIn(("release", first_key), self.widget.calls)
        self.assertEqual([row[1] for row in self.widget.rows], ["You: [one]\n", "You: [two!]\n"])

    def test_untracked_rows_fall_back_and_reset_forgets_rows(self):
        self.manager.add_transcript_line("You: [one]\n", "You")
        self.manager.reset_rows()
        self.manager.update_last_row("You", "You: [x]\n")
        self.assertIn(("delete", "You"), self.widget.calls)
        self.manager.add_transcript_line("plain line\n")
        self.assertEqual(self.widget.calls[-1], ("add", "plain line\n"))

    def test_row_updates_queued_before_a_clear_are_skipped(self):
        queued = []
        self.manager.ui.enqueue_ui_action = lambda fn, *args: queued.append((fn, args))
        self.manager.queue_add_transcript_line("You: [old]\n", "You")
        self.manager.reset_rows()
        self.manager.queue_add_transcript_line("You: [new]\n", "You")
        for fn, args in queued:
            fn(*args)
        self.assertEqual([row[1] for row in self.widget.rows], ["You: [new]\n"])

    def test_response_box_is_rewritten_only_when_the_text_changes(self):
        textbox = MagicMock()
        textbox.tag_ranges.return_value = ()
        slider = SimpleNamespace(get=lambda: 5)
        responder = SimpleNamespace(response="answer", update_response_interval=lambda value: None)
        runtime = SimpleNamespace(responder=SimpleNamespace(enabled=True), update_response_now=False,
                                  previous_response=None)
        for _ in range(3):
            self.manager.update_response_ui(responder, textbox, MagicMock(), slider, runtime)
        self.assertEqual(textbox.insert.call_count, 1)
        responder.response = "answer, longer"
        self.manager.update_response_ui(responder, textbox, MagicMock(), slider, runtime)
        self.assertEqual(textbox.insert.call_args.args, ("0.0", "answer, longer"))


if __name__ == "__main__":
    unittest.main()


class TestFileTranscriptionNeedsFfmpegOnlyWhereUsed(unittest.TestCase):
    def run_batch(self, stt, engine):
        import tempfile
        from app.transcribe.cli import batch
        model = MagicMock(engine=engine)
        model.get_sentences.return_value = ["0:00 - 0:01: hi"]
        with tempfile.TemporaryDirectory() as folder:
            audio = os.path.join(folder, "talk.mp3")
            with open(audio, "wb") as handle:
                handle.write(b"x")
            args = SimpleNamespace(output_file=os.path.join(folder, "out.txt"), transcribe=audio)
            config = {"General": {"stt": stt, "use_api": False}}
            cwd = os.getcwd()
            os.chdir(folder)
            try:
                with patch.object(batch, "create_stt_model", return_value=model) as create, \
                        patch.object(batch, "pick_local_whisper_engine", return_value=engine), \
                        patch.object(batch, "ensure_ffmpeg_available",
                                     side_effect=lambda: self.assertFalse(create.called)) as ffmpeg, \
                        patch.object(batch, "convert_audio_to_16khz", return_value=audio + ".wav"):
                    batch.transcribe_audio_file(args, config)
            finally:
                os.chdir(cwd)
        return ffmpeg.called

    def test_faster_whisper_reads_files_itself(self):
        self.assertFalse(self.run_batch("whisper", "faster-whisper"))

    def test_openai_whisper_and_whispercpp_still_check_for_ffmpeg(self):
        self.assertTrue(self.run_batch("whisper", "openai-whisper"))
        self.assertTrue(self.run_batch("whisper.cpp", None))


class TestOptionalEngines(unittest.TestCase):
    def test_gpu_check_codes(self):
        no_gpu = types.ModuleType("ctranslate2")
        no_gpu.get_cuda_device_count = lambda: 0
        one_gpu = types.ModuleType("ctranslate2")
        one_gpu.get_cuda_device_count = lambda: 1
        one_gpu.__version__ = "4.6.0"
        with patch.dict(sys.modules, {"ctranslate2": no_gpu}):
            self.assertEqual(tm.gpu_check(), 2)
        with patch.dict(sys.modules, {"ctranslate2": one_gpu}), patch.object(tm.sys, "platform", "linux"):
            self.assertEqual(tm.gpu_check(), 0)
        with patch.dict(sys.modules, {"ctranslate2": None}):
            self.assertEqual(tm.gpu_check(), 1)

    def test_gpu_check_fails_when_cuda_libraries_do_not_load_on_windows(self):
        import ctypes
        one_gpu = types.ModuleType("ctranslate2")
        one_gpu.get_cuda_device_count = lambda: 1
        with patch.dict(sys.modules, {"ctranslate2": one_gpu}), patch.object(tm.sys, "platform", "win32"), \
                patch.object(tm, "add_nvidia_dll_directories"), \
                patch.object(ctypes, "WinDLL", side_effect=OSError("not found"), create=True):
            self.assertEqual(tm.gpu_check(), 1)

    def test_turbo_with_whispercpp_uses_its_file_name(self):
        with patch.object(tm.os.path, "isfile", return_value=True):
            model = tm.WhisperCPPSTTModel({"audio_lang": "en", "local_transcripton_model_file": "ggml-turbo"})
        self.assertTrue(model.model_filename.endswith("ggml-large-v3-turbo.bin"))

    def test_turbo_with_openai_whisper_is_loaded_by_name(self):
        fake_whisper = types.ModuleType("whisper")
        fake_whisper.load_model = MagicMock(return_value=MagicMock())
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
        with patch.dict(sys.modules, {"whisper": fake_whisper, "torch": fake_torch}), \
                patch.object(tm.utilities, "ensure_directory_exists"):
            tm.WhisperSTTModel({"local_transcripton_model_file": "turbo", "audio_lang": "en",
                                "local_engine": "openai-whisper"})
        self.assertEqual(fake_whisper.load_model.call_args.args[0], "turbo")
        self.assertEqual(fake_whisper.load_model.call_args.kwargs["download_root"], tm.MODELS_DIR)


class TestMissingAudioDevices(unittest.TestCase):
    def test_a_device_that_cannot_be_opened_gives_a_stand_in(self):
        from sdk import audio_recorder as ar

        class Broken:
            def __init__(self, **kwargs):
                raise OSError("Default Mic could not be opened")

        recorder = ar.open_recorder(Broken, "You", audio_file_name="x.wav")
        self.assertIsInstance(recorder, ar.UnavailableRecorder)
        self.assertTrue(recorder.enabled)        # (no audio comes; a real device later takes over "on")
        self.assertIn("could not be opened", recorder.reason)
        stop = recorder.record_audio(None)
        stop(wait_for_stop=True)
        started = time.monotonic()
        self.assertFalse(recorder.wait_until_capturing(timeout=2))
        self.assertLess(time.monotonic() - started, 0.1)            # no waiting for a device that is not there
        self.assertEqual((recorder.source.SAMPLE_RATE, recorder.source.channels), (16000, 1))
        with self.assertRaises(RuntimeError):
            recorder.set_device(1)

    def test_the_program_starts_with_only_one_source(self):
        from sdk import audio_recorder as ar
        from app.transcribe.core import state

        class GoodSpeaker:
            available = True

            def __init__(self, **kwargs):
                self.source = SimpleNamespace(SAMPLE_RATE=48000, channels=2)

        def broken_mic(**kwargs):
            raise OSError("no microphone")

        runtime = state.create_app_runtime()
        config = {"General": {"disable_mic": False, "mic_device_index": 3,
                              "disable_speaker": False, "speaker_device_index": -1}}
        with patch.object(ar, "MicRecorder", broken_mic), patch.object(ar, "OutputsRecorder", GoodSpeaker), \
                patch.object(state.utilities, "get_data_path", return_value="."):
            runtime.initiate_audio_devices(config)
        self.assertIsInstance(runtime.user_audio_recorder, ar.UnavailableRecorder)
        self.assertIsInstance(runtime.speaker_audio_recorder, GoodSpeaker)

    def test_a_configured_device_that_fails_keeps_the_default(self):
        from app.transcribe.core import state
        recorder = MagicMock()
        recorder.set_device.side_effect = OSError("gone")
        state.AppRuntime._use_device(recorder, 7)                  # does not raise
        recorder.set_device.assert_called_once_with(index=7)


class TestFastStart(unittest.TestCase):
    def fake_faster(self, cached=True, delay=0.0, fail=False):
        calls = []

        class FakeWhisperModel:
            def __init__(self, name, **kwargs):
                calls.append(kwargs.get("local_files_only", False))
                if delay:
                    time.sleep(delay)
                if fail:
                    raise OSError("connection timed out")
                if kwargs.get("local_files_only") and not cached:
                    raise OSError("not in the cache")

            def transcribe(self, audio, **kwargs):
                return iter([SimpleNamespace(start=0.0, end=1.0, text=" ok")]), SimpleNamespace(language="en")

        fake = types.ModuleType("faster_whisper")
        fake.WhisperModel = FakeWhisperModel
        return fake, calls

    def config(self):
        return {"local_transcripton_model_file": "base", "audio_lang": "en", "local_engine": "faster-whisper"}

    def test_a_downloaded_model_is_opened_without_going_online(self):
        fake, calls = self.fake_faster(cached=True)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": None}):
            tm.WhisperSTTModel(self.config())
        self.assertEqual(calls, [True])

    def test_a_missing_model_is_downloaded(self):
        fake, calls = self.fake_faster(cached=False)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": None}):
            model = tm.WhisperSTTModel(self.config())
        self.assertEqual(calls, [True, False])
        self.assertEqual(model.load_state, "ready")

    def test_in_the_background_the_constructor_returns_at_once_and_transcription_waits(self):
        fake, _ = self.fake_faster(delay=0.5)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": None}):
            started = time.monotonic()
            model = tm.WhisperSTTModel(self.config(), background=True)
            self.assertLess(time.monotonic() - started, 0.2)
            self.assertEqual(model.load_state, "loading")
            result = model.transcribe_pcm(pcm(1))
        self.assertEqual(result["text"], "ok")
        self.assertEqual(model.load_state, "ready")

    def test_a_failed_background_load_is_reported(self):
        fake, _ = self.fake_faster(cached=False, fail=True)
        with patch.dict(sys.modules, {"faster_whisper": fake, "ctranslate2": None}), \
                patch.object(tm, "_installed", return_value=False):
            model = tm.WhisperSTTModel(self.config(), background=True)
            self.assertTrue(model._loaded.wait(3))
            self.assertIn("internet", model.load_state)
            self.assertEqual(model.transcribe_pcm(pcm(1)), "")

    def test_the_live_transcriber_asks_for_a_background_load(self):
        from app.transcribe.providers import stt
        with patch.object(stt.tm.STTModelFactory, "get_stt_model_instance") as factory:
            stt.create_stt_model("whisper", {"OpenAI": {"api_key": "", "local_transcripton_model_file": "base",
                                                         "audio_lang": "English"}, "General": {}},
                                 api=False, background=True)
        self.assertTrue(factory.call_args.kwargs["stt_model_config"]["background"])

    def test_window_title_follows_the_model(self):
        from app.transcribe.appui import AppUI
        ui = AppUI.__new__(AppUI)
        model = SimpleNamespace(load_state="loading")
        ui.global_vars = SimpleNamespace(transcriber=SimpleNamespace(stt_model=model))
        titles, popups, timers = [], [], []
        ui.title = titles.append
        ui.after = lambda ms, fn: timers.append(fn)
        ui.display_manager = SimpleNamespace(show_message_popup=lambda *a: popups.append(a))
        ui.show_model_status()
        model.load_state = "ready"
        timers.pop()()
        self.assertEqual(titles, ["Transcribe - loading the speech to text model...", "Transcribe"])
        model.load_state = "no internet"
        ui.show_model_status()
        self.assertEqual(popups, [("Speech to text", "no internet")])



class FakeStream:
    def __init__(self, active=True):
        self.active = active

    def is_active(self):
        return self.active


class TestDeviceStreams(unittest.TestCase):
    def stream(self, speaker, active=True):
        import custom_speech_recognition as csr
        stream = csr.Microphone.MicrophoneStream(frame_bytes=4, sample_rate=1000, speaker=speaker)
        stream.pyaudio_stream = FakeStream(active)
        return stream

    def test_blocks_of_frames_are_put_together_from_the_callback(self):
        stream = self.stream(speaker=False)
        stream.callback(b"\x01" * 12, 3, None, 0)
        stream.callback(b"\x02" * 20, 5, None, 0)
        self.assertEqual(stream.read(4), b"\x01" * 12 + b"\x02" * 4)
        self.assertEqual(stream.read(4), b"\x02" * 16)

    def test_a_silent_output_gives_silence_instead_of_blocking(self):
        stream = self.stream(speaker=True)
        started = time.monotonic()
        self.assertEqual(stream.read(100), bytes(400))
        self.assertLess(time.monotonic() - started, 1.0)
        started = time.monotonic()
        for _ in range(5):                                  # then at about the pace of real audio
            stream.read(100)
        self.assertLess(time.monotonic() - started, 0.9)

    def test_audio_that_is_only_late_is_not_replaced_by_silence(self):
        stream = self.stream(speaker=True)
        stream.callback(b"\x01" * 400, 100, None, 0)
        stream.read(100)
        threading.Timer(0.15, lambda: stream.callback(b"\x02" * 400, 100, None, 0)).start()
        self.assertEqual(stream.read(100), b"\x02" * 400)

    def test_a_microphone_that_stops_sending_ends_the_read(self):
        stream = self.stream(speaker=False)
        stream.MIC_SILENT_LIMIT = 0.3
        with self.assertRaises(OSError):
            stream.read(100)

    def test_a_stream_that_stopped_ends_the_read(self):
        stream = self.stream(speaker=True, active=False)
        with self.assertRaises(OSError):
            stream.read(100)

    def test_the_listener_can_be_stopped_while_nothing_plays(self):
        import custom_speech_recognition as csr
        stream = self.stream(speaker=True)
        stream.sample_rate = 16000
        source = csr.Microphone.__new__(csr.Microphone)
        source.stream, source.CHUNK, source.SAMPLE_RATE, source.SAMPLE_WIDTH = stream, 1024, 16000, 2
        source.__enter__ = None
        recognizer = csr.Recognizer()
        recognizer.energy_threshold = 1000
        started = time.monotonic()
        with self.assertRaises(csr.WaitTimeoutError):
            recognizer.listen(source, timeout=0.5)
        self.assertLess(time.monotonic() - started, 3.0)


def fake_pyaudio(outputs, default_index=0):
    module = types.SimpleNamespace(paWASAPI=13)

    class PyAudio:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get_host_api_info_by_type(self, kind):
            return {"defaultOutputDevice": default_index}

        def get_device_info_by_index(self, index):
            return {"name": outputs[index]}

        def get_loopback_device_info_generator(self):
            for index, name in enumerate(outputs):
                yield {"index": 100 + index, "name": f"{name} [Loopback]", "defaultSampleRate": 48000,
                       "maxInputChannels": 2}

    module.PyAudio = PyAudio
    return module


class TestOutputs(unittest.TestCase):
    def make(self, outputs, default_index=0, broken=(), only_default=False, call=None):
        from sdk import audio_recorder as ar
        made = []

        class Member:
            def __init__(self, source_name, device, audio_file_name, keep_audio_file=False):
                if device["name"].split(" [")[0] in broken:
                    raise OSError("in use")
                self.device_info, self.audio_file_name = device, audio_file_name
                self.keep = keep_audio_file
                self.enabled, self.stop_record_func, self.started, self.alive = True, None, 0, True
                self.source = SimpleNamespace(name=device["name"])
                made.append(self)

            def record_audio(self, audio_queue):
                self.started += 1
                stopper = MagicMock()
                stopper.thread = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None)
                return stopper

            def capture_alive(self):
                return self.alive

        with patch.object(ar, "pyaudio", fake_pyaudio(outputs, default_index)), \
                patch.object(ar, "SpeakerRecorder", Member):
            return ar.OutputsRecorder(audio_file_name="s.wav", only_default=only_default,
                                      call_output_name=lambda: call), made

    def test_the_default_and_the_call_output_are_captured(self):
        # Windows gives the call device name cut to 31 characters
        group, members = self.make(["Speakers (Realtek(R) Audio)", "Headset (WH-1000XM4 Hands-Free AG Audio)", "HDMI"],
                                   default_index=0, call="Headset (WH-1000XM4 Hands-Free ")
        self.assertEqual([m.device_info["name"] for m in group.members],
                         ["Speakers (Realtek(R) Audio) [Loopback]",
                          "Headset (WH-1000XM4 Hands-Free AG Audio) [Loopback]"])
        self.assertEqual([m.audio_file_name for m in members], ["s.wav", None])

    def test_one_output_when_the_call_output_is_the_default_or_unknown(self):
        group, _ = self.make(["Speakers", "HDMI"], call="Speakers")
        self.assertEqual(len(group.members), 1)
        group, _ = self.make(["Speakers", "HDMI"], default_index=1, call=None)
        self.assertEqual([m.device_info["name"] for m in group.members], ["HDMI [Loopback]"])

    def test_on_off_and_start_stop_apply_to_all(self):
        group, members = self.make(["Speakers", "Headphones"], call="Headphones")
        group.enabled = False
        self.assertEqual([m.enabled for m in members], [False, False])
        stop = group.record_audio(None)
        self.assertEqual([m.started for m in members], [1, 1])
        stop(wait_for_stop=True)
        for member in members:
            member.stop_record_func.assert_called_once_with(wait_for_stop=False)

    def test_an_output_that_stops_is_dropped_and_the_others_go_on(self):
        group, members = self.make(["Speakers", "Headphones"], call="Headphones")
        members[1].alive = False
        self.assertTrue(group.capture_alive())
        self.assertEqual(len(group.members), 1)
        members[0].alive = False
        self.assertFalse(group.capture_alive())

    def test_an_output_that_cannot_be_opened_is_skipped(self):
        group, _ = self.make(["Speakers", "Headphones"], broken=("Speakers",), call="Headphones")
        self.assertEqual([m.device_info["name"] for m in group.members], ["Headphones [Loopback]"])
        with self.assertRaises(OSError):
            self.make(["Speakers"], broken=("Speakers",))

    def test_only_the_default_when_asked(self):
        group, _ = self.make(["Speakers", "Headphones"], default_index=1, only_default=True, call="Speakers")
        self.assertEqual([m.device_info["name"] for m in group.members], ["Headphones [Loopback]"])


class Rec:
    def __init__(self, name, alive=True, available=True):
        self.name, self.alive, self.available = name, alive, available
        self.enabled = True
        self.stop_record_func = MagicMock()
        self.source = SimpleNamespace(name=name)
        self.starts = 0

    def capture_alive(self):
        return self.alive

    def record_audio(self, audio_queue):
        self.starts += 1
        return MagicMock()

    def wait_until_capturing(self, timeout):
        return True

    def get_name(self):
        return self.name


class TestDeviceWatcher(unittest.TestCase):
    def make(self, signatures, mic=None, speaker=None):
        from sdk.audio_devices import AudioDeviceWatcher
        runtime = SimpleNamespace(user_audio_recorder=mic or Rec("old mic"),
                                  speaker_audio_recorder=speaker or Rec("old speakers"))
        opened, changes = [], []

        def open_devices():
            pair = (Rec(f"mic {len(opened)}"), Rec(f"speakers {len(opened)}"))
            opened.append(pair)
            return pair

        values = iter(signatures)
        last = [None]

        def signature():
            last[0] = next(values, last[0])
            return last[0]

        watcher = AudioDeviceWatcher(runtime, "queue", open_devices, on_change=lambda m, s: changes.append((m, s)),
                                     signature=signature, interval=0.05, min_gap=0.2)
        return watcher, runtime, opened, changes

    def test_a_device_change_reopens_capture_once_it_has_settled(self):
        watcher, runtime, opened, changes = self.make([(1, 1), (1, 1), (2, 1), (2, 2), (2, 2), (2, 2)])
        old_mic, old_speaker = runtime.user_audio_recorder, runtime.speaker_audio_recorder
        old_mic.enabled = False                               # the user had turned the microphone off
        mic_stop, speaker_stop = old_mic.stop_record_func, old_speaker.stop_record_func
        watcher.start()
        self.assertTrue(wait_for(lambda: len(opened) == 1, 3))
        time.sleep(0.4)
        watcher.stop()
        self.assertEqual(len(opened), 1)                      # one restart for the whole change
        mic_stop.assert_called_once_with(wait_for_stop=True)
        speaker_stop.assert_called_once_with(wait_for_stop=True)
        mic, speaker = opened[0]
        self.assertIs(runtime.user_audio_recorder, mic)
        self.assertIs(runtime.speaker_audio_recorder, speaker)
        self.assertFalse(mic.enabled)                         # choices kept
        self.assertTrue(speaker.enabled)
        self.assertEqual((mic.starts, speaker.starts), (1, 1))
        self.assertEqual(changes, [(mic, speaker)])

    def test_a_capture_that_ended_is_reopened(self):
        watcher, runtime, opened, _ = self.make([(1, 1)], speaker=Rec("old speakers", alive=False))
        watcher.start()
        self.assertTrue(wait_for(lambda: len(opened) == 1, 3))
        watcher.stop()

    def test_a_device_that_keeps_failing_is_retried_less_and_less_often(self):
        watcher, runtime, opened, _ = self.make([(1, 1)], speaker=Rec("old speakers", alive=False))
        original = watcher.open_devices
        watcher.open_devices = lambda: (lambda pair: (pair[0], Rec("dead", alive=False)))(original())
        watcher.start()
        time.sleep(1.6)
        watcher.stop()
        # min_gap 0.2 s doubling: about 0.2, 0.6, 1.4 s -> 3 tries, not 8
        self.assertLessEqual(len(opened), 4)

    def test_stop_waits_for_a_restart_under_way_and_nothing_opens_after(self):
        watcher, runtime, opened, _ = self.make([(1, 1)])
        gate = threading.Event()
        original = watcher.open_devices
        watcher.open_devices = lambda: (gate.wait(2), original())[1]
        threading.Thread(target=watcher.restart, args=("test",), daemon=True).start()
        time.sleep(0.1)
        threading.Timer(0.3, gate.set).start()
        watcher.stop()                                     # returns only after that restart
        self.assertEqual(len(opened), 1)
        watcher.restart("after stop")
        self.assertEqual(len(opened), 1)

    def test_nothing_happens_while_nothing_changes(self):
        watcher, _, opened, _ = self.make([(1, 1)], mic=Rec("none", alive=False, available=False))
        watcher.start()
        time.sleep(0.4)
        watcher.stop()
        self.assertEqual(opened, [])

    def test_signature_is_not_read_outside_windows(self):
        from sdk.audio_devices import windows_audio_signature
        with patch.object(sys, "platform", "linux"):
            self.assertIsNone(windows_audio_signature())

    def test_listeners_expose_their_thread(self):
        import custom_speech_recognition as csr
        source = csr.AudioSource.__new__(csr.AudioSource)
        recognizer = csr.Recognizer()
        with patch.object(csr, "AudioSource", object), \
                patch("threading.Thread.start"):
            stopper = recognizer.listen_in_background(source, "x", lambda *a: None)
        self.assertTrue(hasattr(stopper, "thread"))


class TestChosenDevices(unittest.TestCase):
    def test_a_chosen_device_is_found_again_by_name_after_numbers_change(self):
        from sdk import audio_recorder as ar
        from app.transcribe.core import state
        runtime = state.AppRuntime.__new__(state.AppRuntime)
        runtime._chosen_devices = {}
        recorder = MagicMock()
        with patch.object(ar, "device_name", return_value="Headset Mic"):
            runtime._use_chosen_device(recorder, "mic", 3, want_input=True)
        recorder.set_device.assert_called_with(index=3)
        with patch.object(ar, "device_index_by_name", return_value=7) as lookup:
            runtime._use_chosen_device(recorder, "mic", 3, want_input=True)    # numbers moved
        lookup.assert_called_with("Headset Mic", want_input=True)
        recorder.set_device.assert_called_with(index=7)
        recorder.reset_mock()
        with patch.object(ar, "device_index_by_name", return_value=None):
            runtime._use_chosen_device(recorder, "mic", 3, want_input=True)    # unplugged
        recorder.set_device.assert_not_called()

    def test_a_stand_in_device_counts_as_on(self):
        from sdk import audio_recorder as ar
        self.assertTrue(ar.UnavailableRecorder("You", "none").enabled)
