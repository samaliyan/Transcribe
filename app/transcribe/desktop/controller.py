"""Desktop controller: what happens when the user presses something."""

from __future__ import annotations

import datetime
import os
import subprocess  # nosec
import sys
from typing import TYPE_CHECKING

import customtkinter as ctk

from .. import constants, gpt_responder
from ..core.state import AppRuntime
from ..uicomp import theme
from .presenter import DesktopPresenter
from .services import (
    ActionLogService,
    BrowserService,
    ConversationInsightsService,
    SettingsService,
    TranscriptIOService,
    format_transcript,
)
from .workflows import DesktopWorkflowService

from tsutils import app_logging as al
from tsutils import utilities

if TYPE_CHECKING:
    from ..appui import AppUI


logger = al.get_module_logger(al.UI_LOGGER)
PROJECT_URL = "https://github.com/samaliyan/transcribe"


class DesktopController:
    """Coordinates desktop actions between background services and the UI."""

    def __init__(
        self,
        config: dict,
        runtime: AppRuntime,
        action_log_service: ActionLogService = None,
        settings_service: SettingsService = None,
        transcript_io_service: TranscriptIOService = None,
        browser_service: BrowserService = None,
        insights_service: ConversationInsightsService = None,
        workflow_service: DesktopWorkflowService = None,
        presenter: DesktopPresenter = None,
    ):
        self.config = config
        self.global_vars = runtime
        self.ui: AppUI | None = None
        self.presenter = presenter or DesktopPresenter()
        self.action_log_service = action_log_service or ActionLogService()
        self.settings_service = settings_service or SettingsService()
        self.transcript_io_service = transcript_io_service or TranscriptIOService()
        self.browser_service = browser_service or BrowserService()
        self.insights_service = insights_service or ConversationInsightsService()
        self.workflow_service = workflow_service or DesktopWorkflowService(
            runtime=self.global_vars,
            insights_service=self.insights_service,
        )

    def bind_ui(self, ui: "AppUI"):
        """Attach the active UI instance to the controller."""
        self.ui = ui
        self.presenter.bind_ui(ui)

    def _status(self, text: str, colour: str | None = None):
        if self.ui is not None:
            self.ui.set_status(text, colour)

    def set_audio_device_menus(self):
        """Apply configured audio-device startup state to the UI."""
        if self.config["General"]["disable_speaker"]:
            print("[INFO] Disabling Speaker")
            self.enable_disable_speaker()

        if self.config["General"]["disable_mic"]:
            print("[INFO] Disabling Microphone")
            self.enable_disable_microphone()

    # ------------------------------------------------------------------ transcript
    def _transcript_text(self) -> str:
        return format_transcript(self.global_vars.transcriber.get_transcript())

    def copy_to_clipboard(self):
        """Copy the current transcript to the clipboard."""
        self.capture_action("Copy transcript to clipboard")
        try:
            text = self._transcript_text()
            if not text:
                self._status("The transcript is still empty.")
                return
            self.transcript_io_service.copy_transcript(text)
            self._status("Transcript copied.")
        except Exception as exception:
            logger.error(f"Error copying to clipboard: {exception}")
            self._status(f"Copying failed: {exception}", theme.DANGER)

    def save_file(self):
        """Save transcript text to a file."""
        logger.info(f"{self.__class__.__name__}.save_file")
        filename = ctk.filedialog.asksaveasfilename(
            defaultextension=".txt",
            title="Save the transcript",
            initialfile=f"transcript-{datetime.datetime.now():%Y-%m-%d-%H%M}.txt",
            filetypes=[("Text Files", "*.txt")],
        )
        self.capture_action("Save transcript to file")
        if not filename:
            return
        try:
            self.transcript_io_service.save_transcript(transcript=self._transcript_text(), filename=filename)
            self._status(f"Saved: {filename}")
        except Exception as exception:
            logger.error(f"Error saving file {filename}: {exception}")
            self._status(f"The file could not be saved: {exception}", theme.DANGER)

    def set_transcript_state(self):
        """Pause or resume transcription."""
        logger.info(f"{self.__class__.__name__}.set_transcript_state")
        try:
            enabled = not self.global_vars.transcriber.transcribe
            self.global_vars.transcriber.set_transcription_enabled(enabled)
            self.capture_action(f'{"Enabled" if enabled else "Disabled"} transcription.')
            self._status("Transcription resumed." if enabled else
                         "Transcription is paused: nothing is being written down.")
        except Exception as exception:
            logger.error(f"Error setting transcript state: {exception}")

    def _toggle_source(self, recorder, persona: str, name: str):
        if recorder is None or not getattr(recorder, "available", True):
            return
        try:
            recorder.enabled = not recorder.enabled
            self.global_vars.transcriber.set_source_enabled(persona, recorder.enabled)
            self.capture_action(f'{"Enabled" if recorder.enabled else "Disabled"} {name} input')
            self._status(f'{name.capitalize()} is {"on" if recorder.enabled else "off"}.')
        except Exception as exception:
            logger.error(f"Error toggling {name} state: {exception}")

    def enable_disable_speaker(self):
        """Toggle speaker capture state."""
        self._toggle_source(self.global_vars.speaker_audio_recorder, constants.PERSONA_SPEAKER, "speaker")

    def enable_disable_microphone(self):
        """Toggle microphone capture state."""
        self._toggle_source(self.global_vars.user_audio_recorder, constants.PERSONA_YOU, "microphone")

    # ------------------------------------------------------------------ replies
    def _replies_available(self) -> bool:
        """False (and the settings open) while there is no API key."""
        responder = self.global_vars.responder
        if responder is not None and responder.llm_client is not None:
            return True
        self._status("Replies need an API key: enter it in Settings.", theme.WARN)
        if self.ui is not None:
            self.ui.open_settings()
        return False

    def set_continuous_response(self, enabled: bool):
        """Turn automatic suggestions on or off (and remember the choice)."""
        try:
            self.global_vars.responder.enabled = bool(enabled)
            self.settings_service.save_value("General", "continuous_response", bool(enabled))
            self.capture_action(f'{"Enabled" if enabled else "Disabled"} continuous LLM responses')
            self._status("Replies are suggested automatically." if enabled else
                         "Automatic replies are off: press F2 when you want one.")
        except Exception as exception:
            logger.error(f"Error toggling responder state: {exception}")

    def freeze_unfreeze(self):
        """Toggle continuous LLM responses."""
        self.set_continuous_response(not self.global_vars.responder.enabled)
        self.presenter.set_continuous_response_enabled(self.global_vars.responder.enabled)

    def get_response_now(self):
        """Generate an LLM response from the current transcript."""
        self.capture_action("Get LLM response now")
        if not self._replies_available():
            return
        if not self.global_vars.convo.has_spoken_rows():
            self._status(gpt_responder.NOTHING_TO_ANSWER_MESSAGE)
            return
        if self.ui is not None:
            self.ui.back_to_live()
        self.workflow_service.start_response_workflow(
            response_generator=self.global_vars.responder.generate_response_from_transcript_no_check,
            on_response=self._enqueue_response_text,
            thread_name="GetResponseNow",
        )

    def get_response_selected_now(self):
        """Generate an LLM response for the selected transcript text."""
        self.capture_action("Get LLM response selected now")
        selected_text = self.presenter.get_selected_transcript_text()
        if not selected_text.strip():
            self._status("Select some text in the transcript (or click a line) first.")
            return
        if not self._replies_available():
            return
        if self.ui is not None:
            self.ui.back_to_live()
        self.workflow_service.start_response_workflow(
            response_generator=lambda: self.global_vars.responder.generate_response_for_selected_text(selected_text),
            on_response=self._enqueue_response_text,
            thread_name="GetResponseSelectedNow",
        )

    def summarize(self):
        """Start the summary worker thread."""
        self.capture_action("Get summary from LLM")
        if not self._replies_available():
            return
        self.workflow_service.start_summary_workflow(
            summarize_fn=self.global_vars.responder.summarize,
            on_loading=self._enqueue_loading_popup,
            on_close=self._enqueue_close_popup,
            on_message=self._enqueue_message_popup,
            thread_name="Summarize",
        )

    def word_cloud(self):
        """Start the word cloud worker thread."""
        self.capture_action("Generate word cloud")
        self.workflow_service.start_word_cloud_workflow(
            words_provider=self._get_word_cloud_words,
            on_close=self._enqueue_close_popup,
            on_word_cloud=self._enqueue_word_cloud_popup,
            thread_name="WordCloud",
            on_error=self.presenter.queue_status,
        )

    def update_response_ui_and_read_now(self):
        """Generate a response and signal the audio player to read it aloud."""
        self.capture_action("Get LLM response now and read aloud")
        if not self._replies_available():
            return
        self.global_vars.set_read_response(True)
        self.get_response_now()

    def read_response_aloud(self):
        """Read the reply that is on screen aloud (it was already generated)."""
        self.capture_action("Read response aloud")
        player = self.global_vars.audio_player_var
        if player is None or not self.global_vars.convo.transcript_data[constants.PERSONA_ASSISTANT]:
            self._status("There is no reply to read yet.")
            return
        self.global_vars.set_read_response(True)
        player.speech_text_available.set()
        self._status("Reading the reply aloud (this needs the internet)…")

    # ------------------------------------------------------------------ settings
    def set_audio_language(self, lang: str):
        """Persist the STT audio language and update the live model."""
        try:
            self.settings_service.save_audio_language(lang=lang, transcriber=self.global_vars.transcriber)
            self._status(f"Speech is now understood as {lang}.")
        except Exception as exception:
            logger.error(f"Error setting audio language: {exception}")
            self._status(f"The language could not be changed: {exception}", theme.DANGER)

    def set_response_language(self, lang: str):
        """Persist the response language and update the system prompt."""
        try:
            self.settings_service.save_response_language(lang=lang, convo=self.global_vars.convo)
            self._status(f"Replies will be in {lang}.")
        except Exception as exception:
            logger.error(f"Error setting response language: {exception}")
            self._status(f"The language could not be changed: {exception}", theme.DANGER)

    def settings_saved(self, changes: dict, restart: bool) -> bool:
        """Apply saved settings to the running program. Returns True when a restart is still
        needed for some of them."""
        self.capture_action("Settings saved")
        if not changes:
            return False
        try:
            responder = self.global_vars.responder
            if responder is not None:
                responder.configure()
                responder.update_response_interval(self.config["General"]["llm_response_interval"])
                responder.save_response_to_file = bool(self.config["General"]["save_llm_response_to_file"])
            self.global_vars.convo.refresh_system_prompt()
        except Exception as exception:
            logger.error(f"Error applying settings: {exception}")
            self._status(f"The settings were saved, but could not be applied: {exception}", theme.DANGER)
            return restart
        self._status("Settings saved." if not restart else "Settings saved. Restart to use the new engine.")
        return restart

    def restart_program(self):
        """Start the program again and close this one."""
        self.capture_action("Restart")
        command = [sys.executable] + (sys.argv[1:] if getattr(sys, "frozen", False) else sys.argv)
        try:
            subprocess.Popen(command, cwd=os.getcwd(), close_fds=True)  # nosec  pylint: disable=consider-using-with
        except OSError as exception:
            self._status(f"Could not restart by itself ({exception}): close and open the program.", theme.DANGER)
            return
        self.ui.close_window()

    def open_data_folder(self):
        """Show the folder with transcripts, logs and the database in the file manager."""
        folder = os.path.join(utilities.get_data_path(app_name="Transcribe"), "logs")
        try:
            os.makedirs(folder, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(folder)  # nosec  pylint: disable=no-member
            else:
                subprocess.Popen(["xdg-open", folder])  # nosec  pylint: disable=consider-using-with
        except OSError as exception:
            self._status(f"The folder is {folder} ({exception})")

    def open_link(self, url: str):
        """Open a URL in the default browser."""
        self.capture_action(f"Navigate to {url}.")
        try:
            self.browser_service.open_link(url)
        except Exception as exception:
            logger.error(f"Error opening URL {url}: {exception}")

    def open_github(self):
        """Open the project page."""
        self.open_link(PROJECT_URL)

    def open_support(self):
        """Open the issue tracker."""
        self.open_link(f"{PROJECT_URL}/issues/new")

    def capture_action(self, action_text: str):
        """Append a UI action to the session log."""
        try:
            self.action_log_service.capture(action_text)
        except Exception as exception:
            logger.error(f"Error capturing action {action_text}: {exception}")

    # ------------------------------------------------------------------ queued UI updates
    def _enqueue_response_text(self, response_string: str):
        """Queue response text rendering on the UI thread."""
        self.presenter.queue_response_text(response_string)

    def _enqueue_loading_popup(self, title: str, message: str):
        """Queue a loading popup on the UI thread."""
        self.presenter.queue_loading_popup(title, message)

    def _enqueue_close_popup(self):
        """Queue popup closure on the UI thread."""
        self.presenter.queue_close_popup()

    def _enqueue_message_popup(self, title: str, message: str):
        """Queue a text popup on the UI thread."""
        self.presenter.queue_message_popup(title, message)

    def _enqueue_word_cloud_popup(self, title: str, word_cloud):
        """Queue a word cloud popup on the UI thread."""
        self.presenter.queue_word_cloud_popup(title, word_cloud)

    def _get_word_cloud_words(self):
        """Return the transcript text used for word cloud generation."""
        return self.global_vars.convo.get_conversation(
            sources=[
                constants.PERSONA_YOU,
                constants.PERSONA_SPEAKER,
            ],
            length=0,
        )
