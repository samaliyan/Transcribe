"""Core application state shared across the desktop runtime."""

from __future__ import annotations

import os
import queue
import datetime
from typing import TYPE_CHECKING

from .conversation import Conversation

from tsutils import task_queue, utilities

from sdk import audio_recorder as ar

if TYPE_CHECKING:
    from ..audio_player import AudioPlayer
    from ..transcriber import TranscriberInterface


class AppRuntime:
    """Explicit runtime state for a single application instance."""

    audio_queue: queue.Queue = None
    user_audio_recorder: ar.MicRecorder = None
    speaker_audio_recorder: ar.OutputsRecorder = None
    device_watcher = None
    audio_player_var: "AudioPlayer" = None
    transcriber: "TranscriberInterface" = None
    responder = None
    update_response_now: bool = False
    read_response: bool = False
    previous_response: str = None
    start: datetime.datetime = None
    task_worker = None
    main_window = None
    data_dir = None
    db_file_path: str = None
    current_working_dir: str = None
    db_context: dict = None
    convo: Conversation = None
    _initialized: bool = False

    def __init__(self):
        if self._initialized:
            return

        self._chosen_devices = {}       # devices chosen in the settings, by name
        if self.audio_queue is None:
            self.audio_queue = queue.Queue()

        self.convo = Conversation(self)
        self.start = datetime.datetime.now()
        self.task_worker = task_queue.TaskQueue()
        self.data_dir = utilities.get_data_path(app_name="Transcribe")

        zip_file_name = utilities.incrementing_filename(
            filename=f"{self.data_dir}/logs/transcript",
            extension="zip",
        )
        zip_params = {
            "task_type": task_queue.TaskQueueEnum.ZIP_TASK,
            "folder_path": f"{self.data_dir}/logs",
            "zip_file_name": zip_file_name,
            "skip_zip_files": True,
        }
        self.task_worker.add(**zip_params)
        self.current_working_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        utilities.ensure_directory_exists(f"{self.data_dir}/logs")
        db_log_file = utilities.incrementing_filename(
            filename=f"{self.data_dir}/logs/db",
            extension="log",
        )
        self.db_file_path = f"{self.data_dir}/logs/app.db"
        self.db_context = {
            "db_file_path": self.db_file_path,
            "current_working_dir": self.current_working_dir,
            "db_log_file": db_log_file,
        }
        self._initialized = True

    def set_transcriber(self, transcriber):
        """Set the transcriber used across the application."""
        self.transcriber = transcriber

    def initiate_audio_devices(self, config: dict):
        """Initialize the active microphone and speaker devices."""
        self.user_audio_recorder, self.speaker_audio_recorder = self.open_audio_devices(config)

    def open_audio_devices(self, config: dict, reopen: bool = False):
        """Open the microphone and the speakers as they are now (also after a device change).

        A missing microphone or speaker (or one that fails to open) does not stop the program:
        it continues with the other source. A device chosen by number in the settings is
        remembered by its name, because numbers change when devices come and go.
        """
        general = config["General"]
        data_dir = utilities.get_data_path(app_name="Transcribe")
        realtime = str(general.get("stt", "")).lower() == "openai-realtime"
        # The log recordings go on after a device change (not for the realtime engine, whose
        # recordings are in the device's own format, which may differ on the new device).
        reopen = reopen and not realtime
        print("[INFO] Using default microphone.")
        mic = ar.open_recorder(ar.MicRecorder, "You", audio_file_name=f"{data_dir}/logs/mic.wav",
                               keep_audio_file=reopen)
        if getattr(mic, "available", True) and not general["disable_mic"] and general["mic_device_index"] != -1:
            print("[INFO] Override default microphone with device specified in parameters file.")
            self._use_chosen_device(mic, "mic", int(general["mic_device_index"]))

        fixed = not general["disable_speaker"] and general["speaker_device_index"] != -1
        one_output = fixed or realtime
        print("[INFO] Using default speaker.")
        speaker = ar.open_recorder(ar.OutputsRecorder, "Speaker", audio_file_name=f"{data_dir}/logs/speaker.wav",
                                   only_default=one_output, keep_audio_file=reopen)
        if getattr(speaker, "available", True) and fixed:
            print("[INFO] Override default speaker with device specified in parameters file.")
            self._use_chosen_device(speaker, "speaker", int(general["speaker_device_index"]))
        return mic, speaker

    def _use_chosen_device(self, recorder, kind: str, configured_index: int):
        """Use the device chosen in the settings: by its number the first time, then by its name
        and host API (numbers change when devices come and go)."""
        chosen = self._chosen_devices.get(kind)
        try:
            if chosen is None:
                self._chosen_devices[kind] = ar.device_identity(configured_index)
                index = configured_index
            else:
                index = ar.device_index_by_name(*chosen)
                if index is None:
                    print(f'[WARN] "{chosen[0]}" is not connected; using the default device.')
                    return
        except Exception as exception:  # pylint: disable=broad-except
            print(f"[WARN] Device {configured_index} could not be used ({exception}); keeping the default device.")
            return
        self._use_device(recorder, index)

    @staticmethod
    def _use_device(recorder, index: int):
        """Switch to the device chosen in the settings; keep the default one if that fails."""
        try:
            recorder.set_device(index=index)
        except Exception as exception:  # pylint: disable=broad-except
            print(f"[WARN] Device {index} could not be used ({exception}); keeping the default device.")

    def set_read_response(self, value: bool):
        """Signal that the current response should be read aloud."""
        self.read_response = value
        if self.audio_player_var is not None:
            self.audio_player_var.read_response = value


def create_app_runtime() -> AppRuntime:
    """Create a new application runtime instance."""
    return AppRuntime()


class TranscriptionGlobals(AppRuntime):
    """Compatibility singleton wrapper for legacy imports."""

    _default_instance = None

    def __new__(cls, *args, **kwargs):
        if cls._default_instance is None:
            cls._default_instance = super().__new__(cls)
        return cls._default_instance


T_GLOBALS = TranscriptionGlobals()
