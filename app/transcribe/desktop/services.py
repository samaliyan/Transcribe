"""Desktop service helpers that do not depend on Tk widgets."""

from __future__ import annotations

import datetime
import re
import webbrowser

import pyperclip

from .. import prompts

from tsutils import configuration, utilities


class ActionLogService:
    """Persist user actions for the current desktop session."""

    def __init__(
        self,
        app_name: str = "Transcribe",
        get_data_path=utilities.get_data_path,
        incrementing_filename=utilities.incrementing_filename,
        open_file=open,
        now_provider=datetime.datetime.now,
    ):
        self.app_name = app_name
        self.get_data_path = get_data_path
        self.incrementing_filename = incrementing_filename
        self.open_file = open_file
        self.now_provider = now_provider
        self.log_filename: str | None = None

    def capture(self, action_text: str):
        """Append a UI action to the session log."""
        if not self.log_filename:
            data_dir = self.get_data_path(app_name=self.app_name)
            self.log_filename = self.incrementing_filename(
                filename=f"{data_dir}/logs/ui",
                extension="txt",
            )

        with self.open_file(self.log_filename, mode="a", encoding="utf-8") as ui_file:
            ui_file.write(f"{self.now_provider()}: {action_text}\n")


class SettingsService:
    """Persist mutable desktop settings and apply them to runtime state."""

    def __init__(self, config_factory=configuration.Config, utcnow_provider=datetime.datetime.utcnow):
        self.config_factory = config_factory
        self.utcnow_provider = utcnow_provider

    def save_value(self, section: str, key: str, value):
        """Persist one setting."""
        self.config_factory().add_override_value({section: {key: value}})

    def save_llm_response_interval(self, slider_value) -> int:
        """Persist the LLM response interval."""
        interval = int(slider_value)
        self.save_value("General", "llm_response_interval", interval)
        return interval

    def save_audio_language(self, lang: str, transcriber):
        """Persist the STT language and update the live model."""
        transcriber.set_language(lang)
        self.save_value("OpenAI", "audio_lang", lang)

    def save_response_language(self, lang: str, convo):
        """Persist the response language and update the active system prompt."""
        self.save_value("OpenAI", "response_lang", lang)
        convo.refresh_system_prompt()


def format_transcript(raw_transcript: str) -> str:
    """'You: [text]' rows (as the conversation stores them) -> readable 'You: text' lines."""
    lines = []
    for row in raw_transcript.split("\n"):
        if not row.strip():
            continue
        persona, text = prompts.split_row(row)
        lines.append(f"{persona}: {text}" if persona else text)
    return "\n\n".join(lines) + ("\n" if lines else "")


class TranscriptIOService:
    """Handle transcript copy/save actions without depending on Tk."""

    def __init__(self, clipboard_copy=pyperclip.copy, open_file=open):
        self.clipboard_copy = clipboard_copy
        self.open_file = open_file

    def copy_transcript(self, transcript: str):
        """Copy transcript text to the clipboard."""
        self.clipboard_copy(transcript)

    def save_transcript(self, transcript: str, filename: str):
        """Save transcript text to a file."""
        with self.open_file(filename, mode="w", encoding="utf-8") as file_handle:
            file_handle.write(transcript)


class BrowserService:
    """Open external browser links."""

    def __init__(self, browser_open=webbrowser.open):
        self.browser_open = browser_open

    def open_link(self, url: str):
        """Open a URL in the default browser."""
        self.browser_open(url=url, new=2)


class ConversationInsightsService:
    """Create derived desktop artifacts from conversation text."""

    def __init__(self, word_cloud_factory=None):
        # (wordcloud is imported when first used: it loads slowly and most sessions never need it)
        self.word_cloud_factory = word_cloud_factory

    def normalize_word_cloud_text(self, words: str) -> str:
        """Strip persona labels before generating a word cloud."""
        return re.sub(r"^(You|Speaker|assistant):\s*\[?|\]$", "", words, flags=re.MULTILINE)

    def build_word_cloud(self, words: str):
        """Create a word cloud from conversation text."""
        processed_text = self.normalize_word_cloud_text(words)
        if self.word_cloud_factory is None:
            from wordcloud import WordCloud  # pylint: disable=import-outside-toplevel
            self.word_cloud_factory = WordCloud
        word_cloud = self.word_cloud_factory(
            background_color="white",
            colormap="binary",
            width=500,
            height=500,
        )
        return word_cloud.generate(processed_text)
