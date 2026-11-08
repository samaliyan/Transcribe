import sys
import os
import queue
import threading
import time
import wave
from datetime import datetime
from abc import abstractmethod

import pyaudiowpatch as pyaudio
import custom_speech_recognition as sr
from sdk import audio_format
from tsutils import app_logging as al
from tsutils import configuration  # noqa: E402 pylint: disable=C0413

sys.path.append('../..')

ENERGY_THRESHOLD = 1000
DYNAMIC_ENERGY_THRESHOLD = False

logger = al.get_module_logger(al.AUDIO_RECORDER_LOGGER)

# Mapping of driver types
# https://people.csail.mit.edu/hubert/pyaudio/docs/#id6
DRIVER_TYPE = {
    -1: 'Not actually an audio device',
    0: 'Still in development',
    1: 'DirectSound (Windows only)',
    2: 'Multimedia Extension (Windows only)',
    3: 'Steinberg Audio Stream Input/Output',
    4: 'SoundManager (OSX only)',
    5: 'CoreAudio (OSX only)',
    7: 'Open Sound System (Linux only)',
    8: 'Advanced Linux Sound Architecture (Linux only)',
    9: 'Open Audio Library',
    10: 'BeOS Sound System',
    11: 'Windows Driver Model (Windows only)',
    12: 'JACK Audio Connection Kit',
    13: 'Windows Vista Audio stack architecture'
}


def print_detailed_audio_info(print_func=print):
    """
    Print information about Host APIs and devices using the provided print function.

    :param print_func: Print function or wrapper
    :type print_func: function
    :rtype: None
    """
    print_func("\n|", "~ Audio Drivers on this machine ~".center(20), "|\n")
    header = f" ^ #{'INDEX'.center(7)}#{'DRIVER TYPE'.center(13)}#{'DEVICE COUNT'.center(15)}#{'NAME'.center(5)}"
    print_func(header)
    print_func("-"*len(header))
    py_audio = pyaudio.PyAudio()

    for host_api in py_audio.get_host_api_info_generator():
        print_func(
            (
                f" » "
                f"{('['+str(host_api['index'])+']').center(8)}|"
                f"{str(host_api['type']).center(13)}|"
                f"{str(host_api['deviceCount']).center(15)}|"
                f"  {host_api['name']}"
            )
        )

    print_func("\n\n\n|", "~ Audio Devices on this machine ~".center(20), "|\n")
    header = f" ^ #{'INDEX'.center(7)}# HOST API INDEX #{'LOOPBACK'.center(10)}#{'NAME'.center(5)}"
    print_func(header)
    print_func("-"*len(header))

    for device in py_audio.get_device_info_generator():
        print_func(
            (
                f" » "
                f"{('['+str(device['index'])+']').center(8)}"
                f"{str(device['hostApi']).center(16)}"
                f"  {str(device['isLoopbackDevice']).center(10)}"
                f"  {device['name']}"
            )
        )

    # Below statements are useful to view all available fields in the
    # driver and device list
    # Do not remove these statements from here
    # print('Windows Audio Drivers')
    # for host_api_info_gen in py_audio.get_host_api_info_generator():
    #    print(host_api_info_gen)

    # print('Windows Audio Devices')
    # for device_info_gen in py_audio.get_device_info_generator():
    #    print(device_info_gen)


class BaseRecorder:
    """Base class for Speaker, Microphone classes
    """

    def __init__(self, source, source_name, audio_file_name: str = None, keep_audio_file: bool = False):
        logger.info(BaseRecorder.__name__)
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = ENERGY_THRESHOLD
        self.recorder.dynamic_energy_threshold = DYNAMIC_ENERGY_THRESHOLD
        # Determines if this device is being used for transcription
        self.enabled: bool = True

        if source is None:
            raise ValueError("audio source can't be None")

        self.source = source
        self.source_name: str = source_name
        self.config = configuration.Config().data
        self.stop_record_func = None
        self.audio_file_name = audio_file_name
        # (sample_rate, sample_width, channels) of the data written to audio_file_name.
        self._saved_format = None
        if not keep_audio_file:         # (kept when capture is reopened after a device change)
            self._remove_existing_audio_files()

    def _remove_existing_audio_files(self):
        """Remove existing audio files if they exist"""
        if self.audio_file_name and os.path.exists(self.audio_file_name):
            os.remove(self.audio_file_name)
        if self.audio_file_name and os.path.exists(self.audio_file_name + '.bak'):
            os.remove(self.audio_file_name + '.bak')

    @abstractmethod
    def get_name(self):
        """Get the name of this device
        """

    def enable(self):
        """Enable transcription from this device
        """
        self.enabled = True

    def disable(self):
        """Disable transcription from this device
        """
        self.enabled = False

    def adjust_for_noise(self, device_name, msg):
        """Adjust based on noise from surroundings.
        """
        logger.info(BaseRecorder.adjust_for_noise.__name__)
        print(f"[INFO] Adjusting for ambient noise from {device_name}. {msg}")
        with self.source:
            if getattr(self.source, "stream", None) is None:
                # (the audio library reports the cause on the console; say it plainly here)
                raise OSError(f"{device_name} could not be opened")
            self.recorder.adjust_for_ambient_noise(self.source)
        print(f"[INFO] Completed ambient noise adjustment for {device_name}.")

    def record_audio(self, audio_queue: queue.Queue):
        """Start recording audion from the stream and add data to queue
        """
        if self.config["General"].get("stt", "").lower() == "openai-realtime":
            return self._record_audio_continuously(audio_queue)

        # Phrases are converted to 16 kHz mono once, here, so the transcriber keeps small
        # buffers and no provider needs ffmpeg or temporary files to read them.
        self._saved_format = (audio_format.TARGET_SAMPLE_RATE, audio_format.TARGET_SAMPLE_WIDTH,
                              audio_format.TARGET_CHANNELS)
        channels = int(getattr(self.source, "channels", 1) or 1)

        def record_callback(_, audio: sr.AudioData) -> None:
            if self.enabled:
                try:
                    data = audio_format.to_pcm16_mono_16k(audio.get_raw_data(), audio.sample_rate,
                                                          audio.sample_width, channels)
                except ValueError as exception:
                    logger.error("Could not convert audio from %s: %s", self.source_name, exception)
                    return
                self._enqueue_audio(audio_queue, data)

        stop_func = self.recorder.listen_in_background(source=self.source,
                                                       source_name=self.source_name,
                                                       callback=record_callback,
                                                       phrase_time_limit=self.config['General']['transcript_audio_duration_seconds'])
        return stop_func

    def capture_alive(self) -> bool:
        """False when capture was started and its thread has ended (device unplugged or failed)."""
        thread = getattr(self.stop_record_func, "thread", None)
        return thread is None or thread.is_alive()

    def wait_until_capturing(self, timeout: float = 3.0) -> bool:
        """Block until the capture thread has opened its audio stream (or the timeout passes)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if getattr(self.source, "stream", None) is not None:
                return True
            time.sleep(0.02)
        return False

    def _record_audio_continuously(self, audio_queue: queue.Queue):
        """Capture raw PCM blocks continuously for persistent streaming providers."""
        stop_event = threading.Event()
        chunk_duration_ms = int(
            self.config.get("OpenAIRealtime", {}).get("capture_chunk_duration_ms", 100)
        )
        chunk_frames = max(1, int(self.source.SAMPLE_RATE * chunk_duration_ms / 1000))
        self._saved_format = (self.source.SAMPLE_RATE, getattr(self.source, "SAMPLE_WIDTH", 2),
                              getattr(self.source, "channels", 1))

        def capture_loop():
            try:
                with self.source as active_source:
                    while not stop_event.is_set():
                        data = active_source.stream.read(chunk_frames)
                        if not data:
                            break
                        if self.enabled:
                            self._enqueue_audio(audio_queue, data)
            except Exception as exception:
                if not stop_event.is_set():
                    logger.error("Continuous capture failed for %s: %s", self.source_name, exception)

        capture_thread = threading.Thread(
            target=capture_loop,
            name=f"{self.source_name} Streaming Thread",
            daemon=True,
        )
        capture_thread.start()

        def stop_capture(wait_for_stop=True):
            stop_event.set()
            if wait_for_stop and capture_thread.is_alive():
                capture_thread.join(timeout=2)

        stop_capture.thread = capture_thread

        return stop_capture

    def _enqueue_audio(self, audio_queue: queue.Queue, data: bytes):
        audio_item = (self.source_name, data, datetime.utcnow())
        try:
            audio_queue.put_nowait(audio_item)
        except queue.Full:
            try:
                audio_queue.get_nowait()
                audio_queue.put_nowait(audio_item)
            except (queue.Empty, queue.Full):
                logger.warning("Audio ingress buffer remained full for %s; newest block was dropped", self.source_name)
        if self.audio_file_name:
            with open(file=self.audio_file_name+'.bak', mode='ab') as file_handle:
                file_handle.write(data)

    def write_wav_data_to_file(self) -> str:
        """Write the raw input data into a wave file
        """
        if self.audio_file_name is None:
            return

        if not os.path.exists(self.audio_file_name+'.bak'):
            return

        frame_rate, sample_width, channels = self._saved_format or (
            self.source.SAMPLE_RATE, self.source.SAMPLE_WIDTH, self.source.channels)

        with open(file=self.audio_file_name+'.bak', mode='rb') as input_file_handle:
            data = input_file_handle.read()

        with wave.open(self.audio_file_name, 'wb') as wf:
            # print(f'{datetime.datetime.now()} - Writing speaker data into file: {file_path}')
            wf.setnchannels(channels)    # pylint: disable=E1101
            wf.setsampwidth(sample_width)    # pylint: disable=E1101
            wf.setframerate(frame_rate)    # pylint: disable=E1101
            wf.writeframes(data)    # pylint: disable=E1101
            # print(f'datasize: {len(data)}')
        # print(f'filesize: {os.path.getsize(self.audio_file_name)}')


class UnavailableRecorder:
    """Stands in for a microphone or speaker that could not be opened (none connected, no
    permission, or the device failed). The program keeps working with the other source."""

    def __init__(self, source_name: str, reason: str):
        self.source_name = source_name
        self.reason = reason
        # "On" like a real device: no audio comes anyway, and a device that appears later and
        # replaces this stand-in takes over the on/off choice.
        self.enabled = True
        self.available = False
        self.stop_record_func = None
        self.device_index = None
        self.device_info = {"name": "not available"}
        # The audio format the transcribers expect from a source; no audio ever arrives.
        self.source = type("NoAudioSource", (), {"SAMPLE_RATE": 16000, "SAMPLE_WIDTH": 2,
                                                 "channels": 1, "stream": None})()
        print(f"[WARN] {source_name}: no audio device could be opened ({reason}). "
              "Continuing without it.")
        logger.warning("%s: no audio device could be opened (%s)", source_name, reason)

    def get_name(self):
        return f"{self.source_name} - not available"

    def record_audio(self, audio_queue):  # pylint: disable=unused-argument
        return lambda wait_for_stop=True: None

    def wait_until_capturing(self, timeout: float = 3.0) -> bool:  # pylint: disable=unused-argument
        return False

    def write_wav_data_to_file(self):
        return None

    def set_device(self, index: int):
        raise RuntimeError(f"{self.source_name} is not available: {self.reason}")


def open_recorder(recorder_class, source_name: str, **kwargs):
    """Open a recorder; a missing or failing device gives an UnavailableRecorder instead of a crash."""
    try:
        return recorder_class(source_name=source_name, **kwargs)
    except Exception as exception:  # pylint: disable=broad-except
        reason = str(exception).strip() or exception.__class__.__name__
        return UnavailableRecorder(source_name, reason)


class MicRecorder(BaseRecorder):
    """Encapsultes the Microphone device audio input
    """
    def __init__(self, source_name='You', audio_file_name: str = None, keep_audio_file: bool = False):
        self.source = sr.Microphone(sample_rate=16000)
        self.device_index = self.source.device_index
        self.device_info = None
        super().__init__(source=self.source, source_name=source_name, audio_file_name=audio_file_name,
                         keep_audio_file=keep_audio_file)
        self.adjust_for_noise("Default Mic", "Please make some noise from the Default Mic...")

#    def __init__(self):
#        logger.info(MicRecorder.__name__)
#        with pyaudio.PyAudio() as py_audio:
#             WASAPI is windows specific
#            wasapi_info = py_audio.get_host_api_info_by_type(pyaudio.paWASAPI)
#            self.device_index = wasapi_info["defaultInputDevice"]
#            default_mic = py_audio.get_device_info_by_index(self.device_index)

#        self.device_info = default_mic
#        print(f'default_mic: {default_mic}')

#        source = sr.Microphone(device_index=default_mic["index"],
#                               sample_rate=int(default_mic["defaultSampleRate"]),
#                               channels=1
#                               )
#        self.source = source
#        super().__init__(source=source, source_name="You")
#        print(f'[INFO] Listening to sound from Microphone: {self.get_name()} ')
        # This line is commented because in case of non default microphone it can occasionally take
        # several minutes to execute, thus delaying the start of the application.
#        self.adjust_for_noise("Default Mic", "Please make some noise from the Default Mic...")

    def get_name(self):
        return f'#{self.device_index} - {self.device_info["name"]}'

    def set_device(self, index: int):
        """Set active device based on index.
        """
        logger.info(MicRecorder.set_device.__name__)
        with pyaudio.PyAudio() as py_audio:
            self.device_index = index
            mic = py_audio.get_device_info_by_index(self.device_index)

        # Stop the current stream
        if self.stop_record_func is not None:
            self.stop_record_func(wait_for_stop=True)
            self.stop_record_func = None
        self.device_info = mic
        self.source = sr.Microphone(device_index=mic["index"],
                                    sample_rate=int(mic["defaultSampleRate"]),
                                    channels=1
                                    )

        print(f'[INFO] Listening to sound from Microphone: {self.get_name()} ')
        self.adjust_for_noise("Mic", "Please make some noise from the chosen Mic...")


def loopback_source(device: dict):
    """A capture source for what an output device plays (WASAPI loopback)."""
    # 1024 frames per read. It was pyaudio.get_sample_size(paInt16) = 2 frames, which made the
    # listener run its Python loop about 24,000 times a second for 48 kHz audio.
    return sr.Microphone(speaker=True,
                         device_index=device["index"],
                         sample_rate=int(device["defaultSampleRate"]),
                         chunk_size=1024,
                         channels=device["maxInputChannels"])


class SpeakerRecorder(BaseRecorder):
    """Encapsultes the Speaer device audio input
    """
    def __init__(self, source_name='Speaker', audio_file_name: str = None, device: dict = None,
                 keep_audio_file: bool = False):
        logger.info(SpeakerRecorder.__name__)
        if device is not None:          # one given output device (see OutputsRecorder)
            self.device_index = device["index"]
            self.device_info = device
            super().__init__(source=loopback_source(device), source_name=source_name,
                             audio_file_name=audio_file_name, keep_audio_file=keep_audio_file)
            print(f'[INFO] Listening to sound from Speaker: {self.get_name()} ')
            return
        with pyaudio.PyAudio() as p:
            wasapi_inf = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            self.device_index = wasapi_inf["defaultOutputDevice"]
            default_speakers = p.get_device_info_by_index(self.device_index)

            if not default_speakers["isLoopbackDevice"]:
                for loopback in p.get_loopback_device_info_generator():
                    if default_speakers["name"] in loopback["name"]:
                        default_speakers = loopback
                        break
                else:
                    print("[ERROR] No loopback device found.")

        self.device_info = default_speakers

        source = loopback_source(default_speakers)
        super().__init__(source=source, source_name=source_name, audio_file_name=audio_file_name)
        print(f'[INFO] Listening to sound from Speaker: {self.get_name()} ')
        # On some devices, speaker adjustment is very slow unless some noise is
        # made from the speakers, though capturing of speaker output is very
        # good in almost all instances I have seen thus far.
        # self.adjust_for_noise("Default Speaker",
        #                       "Please play sound from Default Speaker...")

    def get_name(self):
        return f'#{self.device_index} - {self.device_info["name"]}'

    def set_device(self, index: int):
        """Set active device based on index.
        """
        logger.info(SpeakerRecorder.set_device.__name__)
        with pyaudio.PyAudio() as p:
            self.device_index = index
            speakers = p.get_device_info_by_index(self.device_index)

            if not speakers["isLoopbackDevice"]:
                for loopback in p.get_loopback_device_info_generator():
                    if speakers["name"] in loopback["name"]:
                        speakers = loopback
                        break
                else:
                    print("[ERROR] No loopback device found.")

        # Stop the current stream
        if self.stop_record_func is not None:
            self.stop_record_func(wait_for_stop=True)
            self.stop_record_func = None

        self.device_info = speakers
        self.source = loopback_source(speakers)

        print(f'[INFO] Listening to sound from Speaker: {self.get_name()}')
        # self.adjust_for_noise("Speaker",
        #                       f"Please play sound from selected Speakers {self.get_name()}...")


class OutputsRecorder:
    """What the speakers play: the default output and, when it is a different one, the output
    Windows uses for calls (for example a Bluetooth headset in hands-free mode, which meeting
    apps play to). Other outputs are left alone: virtual devices that copy the sound to a real
    output would otherwise give every sentence twice. The default output comes first."""

    def __init__(self, source_name='Speaker', audio_file_name: str = None, only_default: bool = False,
                 keep_audio_file: bool = False, call_output_name=None):
        if call_output_name is None:
            from sdk.audio_devices import windows_call_output_name  # pylint: disable=import-outside-toplevel
            call_output_name = windows_call_output_name
        with pyaudio.PyAudio() as p:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_name = p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"]
            loopbacks = list(p.get_loopback_device_info_generator())
        wanted = [default_name]
        call_name = None if only_default else call_output_name()
        if call_name:
            wanted.append(call_name)
        devices = []
        for name in wanted:
            for device in loopbacks:
                # (Windows gives call device names cut to 31 characters)
                if device["name"].lower().startswith(name.lower()) and device not in devices:
                    devices.append(device)
                    break
        if not devices and loopbacks:
            devices = loopbacks[:1]
        self.source_name = source_name
        self.members = []
        for device in devices:
            try:
                first = not self.members
                self.members.append(SpeakerRecorder(source_name=source_name, device=device,
                                                    audio_file_name=audio_file_name if first else None,
                                                    keep_audio_file=keep_audio_file))
            except Exception as exception:  # pylint: disable=broad-except
                print(f'[WARN] Output "{device.get("name")}" cannot be captured: {exception}')
        if not self.members:
            raise OSError("no output device can be captured")
        self._enabled = True
        self.stop_record_func = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        self._enabled = bool(value)
        for member in self.members:
            member.enabled = self._enabled

    @property
    def source(self):
        return self.members[0].source

    @property
    def device_info(self):
        return self.members[0].device_info

    def get_name(self):
        return ", ".join(member.device_info["name"] for member in self.members)

    def record_audio(self, audio_queue):
        for member in self.members:
            member.stop_record_func = member.record_audio(audio_queue)

        def stop_all(wait_for_stop=True):
            for member in self.members:
                if member.stop_record_func is not None:
                    member.stop_record_func(wait_for_stop=False)
            if wait_for_stop:
                for member in self.members:
                    thread = getattr(member.stop_record_func, "thread", None)
                    if thread is not None:
                        thread.join(timeout=6)
        return stop_all

    def capture_alive(self) -> bool:
        """False when the default output stopped (capture is then opened again). A stopped call
        output is dropped and the default output goes on."""
        if not self.members[0].capture_alive():
            return False
        for member in self.members[1:]:
            if not member.capture_alive():
                print(f'[WARN] Output "{member.device_info.get("name")}" stopped; the default output goes on.')
        self.members = [self.members[0]] + [m for m in self.members[1:] if m.capture_alive()]
        return True

    def wait_until_capturing(self, timeout: float = 3.0) -> bool:
        return self.members[0].wait_until_capturing(timeout)

    def write_wav_data_to_file(self):
        return self.members[0].write_wav_data_to_file()

    def set_device(self, index: int):
        """A chosen output instead of the default and call outputs."""
        self.members = self.members[:1]
        self.members[0].set_device(index)


def device_identity(index: int) -> tuple:
    """(name, host API) of a device by its current index: what stays the same when indexes change."""
    with pyaudio.PyAudio() as p:
        info = p.get_device_info_by_index(index)
        return info["name"], info.get("hostApi")


def device_index_by_name(name: str, host_api: int = None):
    """The current index of the device with this name and host API (None when not connected).
    Matches microphones, outputs and their loopback devices alike."""
    with pyaudio.PyAudio() as p:
        for index in range(p.get_device_count()):
            info = p.get_device_info_by_index(index)
            if info.get("name") == name and (host_api is None or info.get("hostApi") == host_api):
                return index
    return None


if __name__ == "__main__":
    print_detailed_audio_info()
    # Below statements are useful to view all available fields in the
    # default Input Device.
    # Do not delete these lines
    with pyaudio.PyAudio() as p:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        print(wasapi_info)

    with pyaudio.PyAudio() as p:
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_mic = p.get_device_info_by_index(wasapi_info["defaultInputDevice"])
        print(default_mic)
