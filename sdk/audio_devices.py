"""Keeps the microphone and speaker capture on the devices Windows is using right now.

Headphones plugged in, a Bluetooth headset that connects (or switches to its hands-free mode
when a call opens the microphone), a different default device chosen in Windows, a device that
is unplugged: in all these cases capture is opened again on the current devices, by itself.

The audio library (PortAudio) only sees devices that existed when it started, and only looks
again once every stream is closed. So on a change, all capture stops for a moment, the devices
are read again, and capture restarts.
"""
from __future__ import annotations

import sys
import threading
import time

from tsutils import app_logging as al

logger = al.get_module_logger(al.AUDIO_RECORDER_LOGGER)

WAVE_MAPPER = 0xFFFFFFFF
DRVM_MAPPER_PREFERRED_GET = 0x2015
DRVM_MAPPER_CONSOLEVOICECOM_GET = 0x2017


def windows_audio_signature():
    """What would change when devices change: the number of input and output devices and which
    ones are the default and the default for calls. None when it cannot be read (not Windows)."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes  # pylint: disable=import-outside-toplevel
        from ctypes import wintypes  # pylint: disable=import-outside-toplevel
        winmm = ctypes.windll.winmm

        def preferred(message_function, message):
            device, flags = wintypes.DWORD(0xFFFFFFFF), wintypes.DWORD(0)
            result = message_function(ctypes.c_void_p(WAVE_MAPPER), message,
                                      ctypes.byref(device), ctypes.byref(flags))
            return device.value if result == 0 else None

        return (winmm.waveInGetNumDevs(), winmm.waveOutGetNumDevs(),
                preferred(winmm.waveInMessage, DRVM_MAPPER_PREFERRED_GET),
                preferred(winmm.waveOutMessage, DRVM_MAPPER_PREFERRED_GET),
                preferred(winmm.waveInMessage, DRVM_MAPPER_CONSOLEVOICECOM_GET),
                preferred(winmm.waveOutMessage, DRVM_MAPPER_CONSOLEVOICECOM_GET))
    except Exception:  # pylint: disable=broad-except
        return None


class AudioDeviceWatcher:
    """Restarts capture when the devices change or a capture stops by itself.

    open_devices() must create both recorders on the current devices and return them as
    (microphone, speaker). on_change(microphone, speaker) receives the new ones.
    """

    def __init__(self, runtime, audio_queue, open_devices, on_change=None,
                 signature=windows_audio_signature, interval: float = 2.0, min_gap: float = 5.0):
        self.runtime = runtime
        self.audio_queue = audio_queue
        self.open_devices = open_devices
        self.on_change = on_change
        self.signature = signature
        self.interval = interval
        self.min_gap = min_gap
        self.restarts = 0
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._last_restart = 0.0
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, name="AudioDeviceWatcher", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _recorders(self):
        return [self.runtime.user_audio_recorder, self.runtime.speaker_audio_recorder]

    def _capture_ended(self) -> bool:
        return any(getattr(recorder, "available", True) and not getattr(recorder, "capture_alive", lambda: True)()
                   for recorder in self._recorders())

    def _run(self):
        known = self.signature()
        while not self._stop.wait(self.interval):
            try:
                current = self.signature()
                if current != known:
                    # Windows changes several things in a row when a device comes or goes:
                    # wait until it has settled.
                    settled = current
                    while not self._stop.wait(self.interval):
                        current = self.signature()
                        if current == settled:
                            break
                        settled = current
                    known = current
                    self.restart("the audio devices changed")
                elif self._capture_ended() and time.monotonic() - self._last_restart >= self.min_gap:
                    self.restart("an audio device stopped")
                    known = self.signature()
            except Exception as exception:  # pylint: disable=broad-except
                logger.exception(f"Audio device watcher: {exception}")

    def restart(self, reason: str):
        """Stop all capture, open the current devices and start again (keeps on/off choices)."""
        with self._lock:
            self._last_restart = time.monotonic()
            self.restarts += 1
            print(f"[INFO] {reason}: opening the current microphone and speakers again.")
            old_mic, old_speaker = self._recorders()
            for recorder in (old_mic, old_speaker):
                stop = getattr(recorder, "stop_record_func", None)
                if stop is not None:
                    stop(wait_for_stop=True)
                    recorder.stop_record_func = None
            mic, speaker = self.open_devices()
            mic.enabled = old_mic.enabled
            speaker.enabled = old_speaker.enabled
            self.runtime.user_audio_recorder = mic
            self.runtime.speaker_audio_recorder = speaker
            mic.stop_record_func = mic.record_audio(self.audio_queue)
            mic.wait_until_capturing(timeout=2.0)
            speaker.stop_record_func = speaker.record_audio(self.audio_queue)
            if self.on_change is not None:
                self.on_change(mic, speaker)
            print(f"[INFO] Microphone: {mic.get_name()}")
            print(f"[INFO] Speakers: {speaker.get_name()}")
