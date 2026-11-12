"""The real window, built with Tk: layout, status texts, the settings window and the transcript.

Skipped where no display is available (for example a Linux machine without an X server).
"""

import copy
import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

from app.transcribe import gpt_responder
from app.transcribe.core.conversation import APP_DIR
from app.transcribe.desktop import DesktopController, settings_dialog
from app.transcribe.desktop.display import render_reply, source_button_state
from app.transcribe.desktop.services import SettingsService
from app.transcribe.uicomp import theme
from app.transcribe.uicomp.transcript_view import TranscriptView

try:
    _probe = tk.Tk()
    _probe.destroy()
    HAVE_DISPLAY = True
except tk.TclError:
    HAVE_DISPLAY = False


class FakeConfig:
    """Stands in for tsutils.configuration.Config: keeps the changes in memory."""

    def __init__(self, data):
        self.data = data
        self.default_data = copy.deepcopy(data)
        self.saved = []

    def add_override_value(self, values):
        self.saved.append(copy.deepcopy(values))
        for section, items in values.items():
            self.data.setdefault(section, {}).update(items)


def default_settings() -> dict:
    with open(APP_DIR / "parameters.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@unittest.skipUnless(HAVE_DISPLAY, "no display")
class TestTranscriptView(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.view = TranscriptView(self.root)
        self.addCleanup(self.root.destroy)

    def test_rows_show_who_spoke_without_the_storage_brackets(self):
        self.assertEqual(self.view.get_plain_text(), "")            # (only the grey hint is shown)
        self.view.upsert_keyed_row("Speaker-0", "Speaker: [How are you?]\n")
        self.view.upsert_keyed_row("You-1", "You: [Fine, thanks.]\n")
        lines = self.view.get_plain_text().split("\n")
        self.assertTrue(lines[0].startswith("Speaker"))
        self.assertEqual(lines[1], "How are you?")
        self.assertTrue(lines[2].startswith("You"))
        self.assertEqual(lines[3], "Fine, thanks.")

    def test_a_live_update_replaces_its_row_where_it_is(self):
        self.view.upsert_keyed_row("Speaker-0", "Speaker: [How]\n")
        self.view.upsert_keyed_row("You-1", "You: [Fine]\n")
        self.view.upsert_keyed_row("Speaker-0", "Speaker: [How are you today?]\n")
        lines = self.view.get_plain_text().split("\n")
        self.assertEqual((lines[1], lines[3]), ("How are you today?", "Fine"))
        self.assertEqual(self.view.row_count(), 2)

    def test_right_to_left_text_is_aligned_right(self):
        self.view.upsert_keyed_row("Speaker-0", "Speaker: [سلام، حال شما چطوره؟]\n")
        self.view.upsert_keyed_row("You-1", "You: [Hello]\n")
        widget = self.view.text_widget
        self.assertIn("rtl", widget.tag_names("2.0"))
        self.assertNotIn("rtl", widget.tag_names("4.0"))
        self.assertTrue(theme.is_rtl("«سلام» hello"))
        self.assertFalse(theme.is_rtl("hello سلام"))

    def test_the_highlighted_row_can_be_read_and_edited(self):
        self.view.upsert_keyed_row("Speaker-0", "Speaker: [helo wrld]\n")
        self.view._highlight(self.view._tag_at("2.1"))
        self.assertEqual(self.view.selected_row()[1:], ("Speaker", "helo wrld"))
        self.assertEqual(self.view.selection_get(), "helo wrld")
        self.view.replace_selected_text("hello world")
        self.assertEqual(self.view.selected_row()[1:], ("Speaker", "hello world"))
        self.assertIn("hello world", self.view.get_plain_text())

    def test_clear_empties_the_view(self):
        self.view.upsert_keyed_row("You-0", "You: [one]\n")
        self.view.clear_all_text()
        self.assertEqual((self.view.get_plain_text(), self.view.row_count()), ("", 0))
        self.assertIsNone(self.view.selected_row())
        self.view.upsert_keyed_row("You-1", "You: [two]\n")
        self.assertIn("two", self.view.get_plain_text())


@unittest.skipUnless(HAVE_DISPLAY, "no display")
class TestReplyRendering(unittest.TestCase):
    def test_simple_markdown_becomes_readable_text(self):
        root = tk.Tk()
        root.withdraw()
        self.addCleanup(root.destroy)
        box = tk.Text(root)
        render_reply(box, "## Plan\n- **First** do this\n* then that\nسلام **دنیا**")
        self.assertEqual(box.get("1.0", "end-1c").rstrip("\n").split("\n"),
                         ["Plan", "•  First do this", "•  then that", "سلام دنیا"])
        self.assertIn("bold", box.tag_names("1.0"))
        self.assertIn("bold", box.tag_names("2.3"))
        self.assertNotIn("bold", box.tag_names("2.10"))
        self.assertIn("rtl", box.tag_names("4.0"))
        self.assertNotIn("bold", box.tag_names("4.6"))         # (a right-to-left line is one run)
        self.assertEqual(str(box.cget("state")), "disabled")


class FakeRecorder:
    def __init__(self, name, available=True):
        self.enabled, self.available, self._name = True, available, name

    def get_name(self):
        return self._name


@unittest.skipUnless(HAVE_DISPLAY, "no display")
class TestMainWindow(unittest.TestCase):
    def setUp(self):
        from app.transcribe.appui import AppUI
        self.settings = default_settings()
        self.config = FakeConfig(self.settings)
        patcher = patch.object(settings_dialog.configuration, "Config", return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)

        convo = MagicMock()
        convo.has_spoken_rows.return_value = True
        convo.get_response_for_row.return_value = None
        self.responder = SimpleNamespace(
            enabled=True, llm_client=None, status=gpt_responder.STATUS_NO_KEY, response="",
            last_error=gpt_responder.NO_KEY_MESSAGE, configure=MagicMock(),
            update_response_interval=MagicMock(), save_response_to_file=True,
            generate_response_from_transcript_no_check=MagicMock(return_value="ok"))
        self.model = SimpleNamespace(load_state="loading")
        transcriber = SimpleNamespace(
            stt_model=self.model, last_error=None, transcribe=True,
            get_transcript=lambda: "You: [hello]\n\nSpeaker: [hi there]\n\n",
            set_source_enabled=MagicMock(), set_language=MagicMock(), clear_transcriber_context=MagicMock())
        transcriber.set_transcription_enabled = lambda on: setattr(transcriber, "transcribe", on)
        self.runtime = SimpleNamespace(
            transcriber=transcriber, responder=self.responder, convo=convo, audio_queue=None,
            user_audio_recorder=FakeRecorder("Mic"), speaker_audio_recorder=FakeRecorder("", available=False),
            main_window=None, audio_player_var=None, update_response_now=False, read_response=False,
            set_read_response=MagicMock())
        self.workflows = MagicMock()
        controller = DesktopController(
            config=self.settings, runtime=self.runtime, action_log_service=MagicMock(),
            settings_service=SettingsService(config_factory=lambda: self.config),
            transcript_io_service=MagicMock(), browser_service=MagicMock(), workflow_service=self.workflows)
        self.ui = AppUI(config=self.settings, runtime=self.runtime, controller=controller)
        self.addCleanup(self.destroy)
        self.ui.update_initial_transcripts()
        self.ui.update()

    def destroy(self):
        for name in self.ui.tk.call("after", "info"):
            self.ui.after_cancel(name)
        self.ui.destroy()

    def refresh(self):
        self.ui.display_manager.refresh(self.runtime)
        self.ui.update()

    def test_status_texts_follow_the_program(self):
        ui = self.ui
        self.assertIn("Loading", ui.stt_status_label.cget("text"))
        self.assertIn("loading", ui.title())
        self.assertEqual(ui.speaker_button.cget("text"), "No speaker found")
        self.assertEqual(ui.mic_button.cget("text"), "●  Microphone")
        self.assertTrue(ui.ai_banner_visible)                       # no API key yet
        self.assertEqual(ui.ai_banner_button.cget("text"), "Add API key")

        self.model.load_state = "ready"
        self.responder.status, self.responder.last_error = gpt_responder.STATUS_READY, None
        self.responder.response = "Say **yes**."
        self.refresh()
        self.assertEqual((ui.stt_status_label.cget("text"), ui.title()), ("Listening", "Transcribe"))
        self.assertFalse(ui.ai_banner_visible)
        self.assertEqual(ui.response_textbox.get("1.0", "end-1c").strip(), "Say yes.")
        self.assertEqual(ui.display_manager.current_reply_text(), "Say **yes**.")

        self.runtime.transcriber.last_error = "Speech to text: the API key was rejected. Check it in Settings."
        self.responder.status, self.responder.last_error = gpt_responder.STATUS_ERROR, "No credit left."
        self.refresh()
        self.assertTrue(ui.stt_notice_visible)
        self.assertIn("rejected", ui.stt_notice_label.cget("text"))
        self.assertEqual(ui.ai_banner_label.cget("text"), "No credit left.")

    def test_buttons_do_what_they_say(self):
        ui = self.ui
        ui.pause_button.invoke()
        self.refresh()
        self.assertFalse(self.runtime.transcriber.transcribe)
        self.assertEqual(ui.pause_button.cget("text"), "Resume")

        ui.mic_button.invoke()
        self.refresh()
        self.assertFalse(self.runtime.user_audio_recorder.enabled)
        self.assertEqual(ui.mic_button.cget("text"), "○  Microphone off")

        ui.auto_switch.toggle()
        self.assertFalse(self.responder.enabled)
        self.assertIn({"General": {"continuous_response": False}}, self.config.saved)

        ui.change_text_size(+2)
        self.assertEqual(ui.text_size, self.settings["General"]["ui_font_size"])
        self.assertIn({"General": {"ui_font_size": ui.text_size}}, self.config.saved)

        ui.audio_lang_combobox.set("Persian")
        ui.set_audio_language("Persian")
        self.runtime.transcriber.set_language.assert_called_with("Persian")
        self.assertEqual(self.settings["OpenAI"]["audio_lang"], "Persian")

    def test_asking_for_a_reply_without_a_key_opens_the_settings(self):
        self.ui.get_response_now()
        self.ui.update()
        self.assertIsNotNone(self.ui._settings_window)
        self.workflows.start_response_workflow.assert_not_called()
        self.ui._settings_window.destroy()

        self.responder.llm_client = object()
        self.ui.get_response_now()
        self.workflows.start_response_workflow.assert_called_once()

    def test_settings_are_saved_and_applied(self):
        self.ui.open_settings()
        self.ui.update()
        dialog = self.ui._settings_window
        dialog.api_key_entry.insert(0, "sk-new")
        dialog.model_box.set("my-model")
        dialog.base_url_entry.delete(0, "end")
        dialog.base_url_entry.insert(0, "https://openrouter.ai/api/v1")
        dialog.preset_menu.set("Job interview (I am the candidate)")
        dialog._apply_preset("Job interview (I am the candidate)")
        restarts = []
        self.ui.ask_restart = lambda: restarts.append(True)
        dialog.save()
        self.ui.update()

        self.assertEqual(self.settings["OpenAI"]["api_key"], "sk-new")
        self.assertEqual(self.settings["OpenAI"]["ai_model"], "my-model")
        self.assertEqual(self.settings["OpenAI"]["base_url"], "https://openrouter.ai/api/v1")
        self.assertIn("job interview", self.settings["General"]["system_prompt"])
        saved = self.config.saved[-1]
        self.assertEqual(set(saved["OpenAI"]), {"api_key", "ai_model", "base_url"})     # only what changed
        self.responder.configure.assert_called_once()
        self.runtime.convo.refresh_system_prompt.assert_called_once()
        self.assertEqual(restarts, [])                                                # nothing needed one

    def test_changing_the_engine_asks_for_a_restart_and_needs_its_key(self):
        self.ui.open_settings("Speech to text")
        self.ui.update()
        dialog = self.ui._settings_window
        online = next(label for label, value in dialog.engine_choices.items() if value == ("whisper", True))
        dialog.engine_menu.set(online)
        dialog.save()
        self.assertIn("needs the API key", dialog.message_label.cget("text"))
        self.assertTrue(dialog.winfo_exists())

        dialog.api_key_entry.insert(0, "sk-new")
        small = next(label for label, value in dialog.model_choices.items() if value == "small")
        dialog.local_model_menu.set(small)
        restarts = []
        self.ui.ask_restart = lambda: restarts.append(True)
        dialog.save()
        self.assertEqual(restarts, [True])
        self.assertTrue(self.settings["General"]["use_api"])
        self.assertEqual(self.settings["OpenAI"]["local_transcripton_model_file"], "small")

    def test_copy_and_save_use_readable_text(self):
        self.ui.copy_to_clipboard()
        io_service = self.ui.controller.transcript_io_service
        io_service.copy_transcript.assert_called_once_with("You: hello\n\nSpeaker: hi there\n")
        self.assertEqual(self.ui.status_label.cget("text"), "Transcript copied.")

    def test_closing_remembers_the_window_size(self):
        self.ui.quit = MagicMock()
        self.ui.close_window()
        self.ui.quit.assert_called_once()
        self.assertTrue(any("window_geometry" in saved.get("General", {}) for saved in self.config.saved))


class TestSmallHelpers(unittest.TestCase):
    def test_source_states(self):
        self.assertEqual(source_button_state(None)[0], "missing")
        self.assertEqual(source_button_state(FakeRecorder("x", available=False))[0], "missing")
        recorder = FakeRecorder("Headset")
        self.assertEqual(source_button_state(recorder), ("on", "Headset"))
        recorder.enabled = False
        self.assertEqual(source_button_state(recorder)[0], "off")

    def test_only_real_changes_are_saved(self):
        current = {"General": {"stt": "whisper", "use_api": False, "proxy": ""}, "OpenAI": {"api_key": "k"}}
        wanted = {"General": {"stt": "whisper", "use_api": True, "proxy": ""}, "OpenAI": {"api_key": "k"}}
        changes = settings_dialog.diff_settings(current, wanted)
        self.assertEqual(changes, {"General": {"use_api": True}})
        self.assertTrue(settings_dialog.needs_restart(changes))
        self.assertFalse(settings_dialog.needs_restart({"OpenAI": {"api_key": "x"}, "General": {"system_prompt": "p"}}))


if __name__ == "__main__":
    unittest.main()
