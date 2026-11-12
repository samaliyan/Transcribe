"""Thread-safe requests to the window: everything here only queues work for the Tk thread."""

from __future__ import annotations


class DesktopPresenter:
    """Queue UI updates from worker threads, and read simple values from widgets."""

    def __init__(self):
        self.ui = None

    def bind_ui(self, ui):
        """Attach the presenter to the active UI instance."""
        self.ui = ui

    def get_selected_transcript_text(self) -> str:
        """The text selected in the transcript (or the highlighted row)."""
        return self.ui.transcript_text.selection_get()

    def set_continuous_response_enabled(self, enabled: bool):
        """Show whether automatic suggestions are on."""
        if enabled:
            self.ui.auto_switch.select()
        else:
            self.ui.auto_switch.deselect()

    def queue_response_text(self, response_string: str):
        """Queue response text rendering on the Tk thread."""
        self.ui.enqueue_ui_action(self.ui.write_response_text, response_string)

    def queue_loading_popup(self, title: str, message: str):
        """Queue a loading popup on the Tk thread."""
        self.ui.enqueue_ui_action(self.ui.show_loading_popup, title, message)

    def queue_close_popup(self):
        """Queue popup closure on the Tk thread."""
        self.ui.enqueue_ui_action(self.ui.close_popup)

    def queue_message_popup(self, title: str, message: str):
        """Queue a text popup on the Tk thread."""
        self.ui.enqueue_ui_action(self.ui.show_message_popup, title, message)

    def queue_word_cloud_popup(self, title: str, word_cloud):
        """Queue a word-cloud popup on the Tk thread."""
        self.ui.enqueue_ui_action(self.ui.show_word_cloud_popup, title, word_cloud)

    def queue_status(self, text: str, colour: str | None = None):
        """Queue a status bar message on the Tk thread."""
        self.ui.enqueue_ui_action(self.ui.set_status, text, colour)
