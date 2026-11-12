"""What the window shows: transcript rows, the suggested reply, status texts and popups."""

from __future__ import annotations

import itertools
import re
import tkinter as tk
from io import BytesIO

import customtkinter as ctk

from .. import gpt_responder, prompts
from ..uicomp import theme
from .view import primary_button, quiet_button

REFRESH_INTERVAL_MS = 150
STATUS_SECONDS = 6          # how long a passing message stays in the status bar
_BOLD = re.compile(r"\*\*(.+?)\*\*")


def stt_status(runtime) -> tuple[str, str, str | None]:
    """(short text, colour, notice or None) describing speech to text right now."""
    transcriber = getattr(runtime, "transcriber", None)
    if transcriber is None:
        return "", theme.MUTED, None
    state = getattr(getattr(transcriber, "stt_model", None), "load_state", "ready")
    if state == "loading":
        return "Loading the speech to text model…", theme.WARN, None
    if state == "downloading":
        return "Downloading the speech to text model (first use only)…", theme.WARN, None
    if state != "ready":
        return "Speech to text is not working", theme.DANGER, f"{state} It is tried again every 30 seconds."
    error = getattr(transcriber, "last_error", None)
    if error:
        return "Speech to text is failing", theme.DANGER, str(error)
    if not getattr(transcriber, "transcribe", True):
        return "Paused", theme.WARN, None
    return "Listening", theme.OK, None


def ai_status(responder) -> tuple[str, str, str | None]:
    """(short text, colour, notice or None) describing the suggested replies right now."""
    if responder is None:
        return "", theme.MUTED, None
    status = getattr(responder, "status", gpt_responder.STATUS_READY)
    if status == gpt_responder.STATUS_NO_KEY:
        return "off", theme.MUTED, ("Replies need the API key of an AI provider. Transcription works "
                                    "without it.")
    if status == gpt_responder.STATUS_WORKING:
        return "thinking…", theme.ACCENT, None
    if status == gpt_responder.STATUS_ERROR:
        error = getattr(responder, "last_error", None) or "The last request failed."
        if error == gpt_responder.NOTHING_TO_ANSWER_MESSAGE:
            return "", theme.MUTED, None
        return "error", theme.DANGER, error
    return "", theme.MUTED, None


def source_button_state(recorder) -> tuple[str, str]:
    """('on' | 'off' | 'missing', device name) for the microphone or speaker button."""
    if recorder is None or not getattr(recorder, "available", True):
        return "missing", ""
    name = ""
    try:
        name = str(recorder.get_name() or "")
    except Exception:  # pylint: disable=broad-except
        info = getattr(recorder, "device_info", None)
        name = str(info.get("name", "")) if isinstance(info, dict) else ""
    return ("on" if getattr(recorder, "enabled", True) else "off"), name


def render_reply(widget: tk.Text, text: str):
    """Show a reply in a text widget: **bold**, headings and list items of simple Markdown, and
    right-aligned paragraphs for right-to-left languages."""
    top = widget.yview()[0]
    widget.configure(state="normal")
    widget.delete("1.0", tk.END)
    for line in text.split("\n"):
        tags: tuple = ()
        stripped = line.strip()
        heading = re.match(r"^#{1,6}\s+(.*)$", stripped)
        if heading:
            stripped, tags = heading.group(1), ("bold",)
        elif re.match(r"^[-*•]\s+", stripped):
            stripped = "•  " + re.sub(r"^[-*•]\s+", "", stripped)
            tags = ("item",)
        if theme.is_rtl(stripped):
            # One run per line: Windows orders and joins right-to-left text within a run, and
            # a line cut into normal and bold runs would show them in the wrong order.
            widget.insert(tk.END, _BOLD.sub(r"\1", stripped) + "\n", tags + ("rtl",))
            continue
        position = 0
        for match in _BOLD.finditer(stripped):
            widget.insert(tk.END, stripped[position:match.start()], tags)
            widget.insert(tk.END, match.group(1), tags + ("bold",))
            position = match.end()
        widget.insert(tk.END, stripped[position:] + "\n", tags)
    widget.configure(state="disabled")
    widget.yview_moveto(top)


class DesktopDisplayManager:
    """Encapsulate widget-facing display updates for the desktop UI."""

    def __init__(self, ui_font_size: int = theme.DEFAULT_TEXT_SIZE):
        self.ui = None
        self.ui_font_size = ui_font_size
        self.popup_window = None
        # Each speaker's newest transcript row is addressed by a key, so a live update replaces
        # that row in place instead of searching and reflowing the whole transcript.
        self._open_rows: dict[str, str] = {}
        self._row_numbers = itertools.count()
        # Bumped when the transcript is cleared: row updates queued before that are skipped.
        self._epoch = 0
        self._last_response_text: str | None = None
        self._history_text: str | None = None       # an earlier reply being shown instead of the live one
        self._status_until = 0.0
        self._last_states: dict = {}

    def bind_ui(self, ui):
        """Attach the active UI instance."""
        self.ui = ui

    # ------------------------------------------------------------------ transcript rows
    def update_initial_transcripts(self, runtime):
        """Connect the conversation to the window and start the periodic refresh."""
        runtime.convo.set_handlers(
            self.queue_update_last_row,
            self.queue_add_transcript_line,
            partial=self.queue_upsert_partial_row,
            finalize_partial=self.queue_finalize_partial_row,
        )
        self.apply_text_size(self.ui.text_size)
        self.refresh(runtime)

    def update_last_row(self, speaker: str, input_text: str):
        """Replace the latest transcript row for a speaker."""
        row_key = self._open_rows.get(speaker)
        if row_key is None:
            self.add_transcript_line(input_text, speaker)
            return
        self.ui.transcript_text.upsert_keyed_row(row_key, input_text)

    def add_transcript_line(self, input_text: str, persona: str | None = None):
        """Append a transcript row; a speaker's row stays addressable until their next one."""
        transcript_text = self.ui.transcript_text
        if persona is None:
            transcript_text.add_text_to_bottom(input_text)
            return
        previous = self._open_rows.get(persona)
        if previous is not None:
            transcript_text.release_keyed_row(previous)
        row_key = f"{persona}-{next(self._row_numbers)}"
        self._open_rows[persona] = row_key
        transcript_text.upsert_keyed_row(row_key, input_text)

    def reset_rows(self):
        """Forget open rows (the transcript widget was cleared)."""
        self._open_rows.clear()
        self._epoch += 1

    def _if_current(self, epoch: int, action, *args):
        if epoch == self._epoch:
            action(*args)

    def queue_update_last_row(self, speaker: str, input_text: str):
        """Queue transcript row replacement on the Tk thread."""
        self.ui.enqueue_ui_action(self._if_current, self._epoch, self.update_last_row, speaker, input_text)

    def queue_add_transcript_line(self, input_text: str, persona: str | None = None):
        """Queue transcript insertion on the Tk thread."""
        self.ui.enqueue_ui_action(self._if_current, self._epoch, self.add_transcript_line, input_text, persona)

    def queue_upsert_partial_row(self, item_id: str, input_text: str):
        """Queue insertion or replacement of one provider-addressed partial row."""
        self.ui.enqueue_ui_action(self._if_current, self._epoch,
                                  self.ui.transcript_text.upsert_keyed_row, item_id, input_text)

    def queue_finalize_partial_row(self, item_id: str, input_text: str):
        """Queue replacement of a provider partial with its durable final text."""
        self.ui.enqueue_ui_action(self._if_current, self._epoch,
                                  self.ui.transcript_text.finalize_keyed_row, item_id, input_text)

    # ------------------------------------------------------------------ reply panel
    def apply_text_size(self, size: int):
        """Use one text size for the transcript and the reply."""
        self.ui.text_size = size
        self.ui.transcript_text.set_text_size(size)
        box = self.ui.response_textbox
        box.configure(font=theme.font(size + 1))
        box.tag_configure("bold", font=theme.font(size + 1, "bold"))
        box.tag_configure("item", lmargin1=6, lmargin2=26)
        box.tag_configure("rtl", justify="right")
        box.tag_configure("hint", foreground=theme.MUTED, font=theme.font(max(10, size - 1)))

    def write_response_text(self, response_string: str):
        """Render a reply in the reply panel."""
        if not response_string:
            return
        text = prompts.strip_brackets(response_string)
        if text == self._last_response_text and self._history_text is None:
            return
        render_reply(self.ui.response_textbox, text)
        self._last_response_text = text

    def show_reply_hint(self, hint: str):
        """Grey guidance in the empty reply panel."""
        box = self.ui.response_textbox
        box.configure(state="normal")
        box.delete("1.0", tk.END)
        box.insert("1.0", hint, ("hint",))
        box.configure(state="disabled")
        self._last_response_text = None

    def show_history(self, text: str):
        """Show the reply that belongs to an earlier transcript row until 'Back to live'."""
        self._history_text = text
        render_reply(self.ui.response_textbox, prompts.strip_brackets(text))
        self._last_response_text = None
        self.ui.live_button.pack(side="right", padx=(0, 6))

    def back_to_live(self):
        self._history_text = None
        self._last_response_text = None
        self.ui.live_button.pack_forget()

    def current_reply_text(self) -> str:
        """The reply as shown (without the grey hint)."""
        if self._history_text is not None:
            return prompts.strip_brackets(self._history_text)
        return self._last_response_text or ""

    # ------------------------------------------------------------------ periodic refresh
    def refresh(self, runtime):
        """Bring every status text and the reply panel up to date; runs on a timer."""
        ui = self.ui
        try:
            self._refresh_sources(runtime)
            self._set_cached("stt", stt_status(runtime), self._show_stt_status)
            responder = getattr(runtime, "responder", None)
            label, colour, notice = ai_status(responder)
            self._set_cached("ai", (label, colour),
                             lambda value: ui.ai_status_label.configure(text=value[0], text_color=value[1]))
            self._set_cached("banner", (notice, getattr(responder, "status", None)), self._show_banner)
            self._set_cached("paused", getattr(getattr(runtime, "transcriber", None), "transcribe", True),
                             lambda on: ui.pause_button.configure(
                                 text="Pause" if on else "Resume",
                                 fg_color=theme.SURFACE_2 if on else theme.WARN,
                                 hover_color=theme.SURFACE_3 if on else theme.WARN,
                                 text_color=theme.TEXT if on else "#1a1a1a"))
            if responder is not None and self._history_text is None:
                response = getattr(responder, "response", "")
                if response:
                    self.write_response_text(response)
                elif self._last_response_text is None and not self._last_states.get("hinted"):
                    self._last_states["hinted"] = True
                    self.show_reply_hint(
                        "Suggested replies appear here while the conversation goes on.\n\n"
                        "Press F2 (or the button below) to ask for one at any moment.")
            if self._status_until and ui.tk_now() > self._status_until:
                self._status_until = 0.0
                ui.status_label.configure(text="")
        finally:
            ui.after(REFRESH_INTERVAL_MS, self.refresh, runtime)

    def _set_cached(self, name: str, value, apply):
        """Touch a widget only when what it shows changes."""
        if self._last_states.get(name, object()) != value:
            self._last_states[name] = value
            apply(value)

    @staticmethod
    def _fit(text: str, limit: int = 110) -> str:
        return text if len(text) <= limit else text[:limit - 1] + "…"

    def _refresh_sources(self, runtime):
        ui = self.ui
        for name, button, recorder in (
                ("Microphone", ui.mic_button, getattr(runtime, "user_audio_recorder", None)),
                ("Speaker", ui.speaker_button, getattr(runtime, "speaker_audio_recorder", None))):
            state, _device = source_button_state(recorder)

            def apply(value, button=button, name=name):
                if value == "on":
                    button.configure(text=f"●  {name}", text_color=theme.OK, state="normal")
                elif value == "off":
                    button.configure(text=f"○  {name} off", text_color=theme.MUTED, state="normal")
                else:
                    button.configure(text=f"No {name.lower()} found", text_color=theme.WARN, state="disabled")

            self._set_cached(f"source-{name}", state, apply)

    def _show_stt_status(self, value):
        text, colour, notice = value
        ui = self.ui
        ui.stt_status_label.configure(text=text, text_color=colour)
        # The title says the same for the taskbar (and the package test waits for "Transcribe").
        if "model" in text and notice is None:
            ui.title("Transcribe - loading the speech to text model...")
        else:
            ui.title("Transcribe" if notice is None else "Transcribe - speech to text is not working")
        if notice is None:
            if ui.stt_notice_visible:
                ui.stt_notice.grid_forget()
                ui.stt_notice_visible = False
            return
        ui.stt_notice_label.configure(text=notice)
        if not ui.stt_notice_visible:
            ui.stt_notice.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 8))
            ui.stt_notice_visible = True

    def _show_banner(self, value):
        notice, status = value
        ui = self.ui
        if notice is None:
            if ui.ai_banner_visible:
                ui.ai_banner.pack_forget()
                ui.ai_banner_visible = False
            return
        ui.ai_banner_label.configure(
            text=notice, text_color=theme.TEXT if status == gpt_responder.STATUS_NO_KEY else theme.DANGER)
        ui.ai_banner_button.configure(
            text="Add API key" if status == gpt_responder.STATUS_NO_KEY else "Open Settings")
        if not ui.ai_banner_visible:
            ui.ai_banner.pack(fill="x", padx=12, pady=(6, 2), before=ui.response_frame)
            ui.ai_banner_visible = True

    def set_status(self, text: str, colour: str | None = None, keep: bool = False):
        """A short message in the status bar (it disappears after a few seconds unless keep)."""
        self.ui.status_label.configure(text=text, text_color=colour or theme.MUTED)
        self._status_until = 0.0 if keep else self.ui.tk_now() + STATUS_SECONDS

    # ------------------------------------------------------------------ popups
    def close_popup(self):
        """Close the active popup, if present."""
        if self.popup_window is None:
            return
        try:
            self.popup_window.destroy()
        except tk.TclError:
            pass
        finally:
            self.popup_window = None

    def _popup(self, title: str, size: str):
        self.close_popup()
        popup = ctk.CTkToplevel(self.ui, fg_color=theme.BG)
        popup.title(title)
        popup.geometry(size)
        popup.transient(self.ui)
        popup.after(120, popup.lift)
        self.popup_window = popup
        return popup

    def show_loading_popup(self, title: str, msg: str):
        """Create a temporary loading popup."""
        popup = self._popup(title, "300x90")
        ctk.CTkLabel(popup, text=msg, font=theme.font(13), text_color=theme.TEXT).pack(
            expand=True, fill="both", padx=16, pady=16)

    def show_message_popup(self, title: str, msg: str):
        """Create a text popup with copy and close actions."""
        popup = self._popup(title, "560x520")
        box = tk.Text(popup, wrap="word", borderwidth=0, highlightthickness=0, background=theme.SURFACE,
                      foreground=theme.TEXT, selectbackground=theme.SELECT, padx=14, pady=12,
                      font=theme.font(self.ui.text_size), spacing3=4)
        box.tag_configure("bold", font=theme.font(self.ui.text_size, "bold"))
        box.tag_configure("item", lmargin1=6, lmargin2=26)
        box.tag_configure("rtl", justify="right")
        buttons = ctk.CTkFrame(popup, fg_color="transparent")
        buttons.pack(side="bottom", fill="x", padx=12, pady=10)
        box.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        render_reply(box, msg)

        def copy_text():
            self.ui.copy_text(box.get("1.0", "end-1c"))

        primary_button(buttons, "Copy", width=100, command=copy_text).pack(side="right")
        quiet_button(buttons, "Close", width=90, height=36, command=self.close_popup).pack(
            side="right", padx=(0, 8))

    def show_word_cloud_popup(self, title: str, word_cloud):
        """Create a word-cloud popup."""
        from PIL import Image, ImageTk  # pylint: disable=import-outside-toplevel
        popup = self._popup(title, "540x600")
        buffer = BytesIO()
        word_cloud.to_image().save(buffer, format="PNG")
        buffer.seek(0)
        image = ImageTk.PhotoImage(Image.open(buffer))
        label = tk.Label(popup, image=image, borderwidth=0, background=theme.BG)
        label.image = image
        label.pack(padx=12, pady=12)
        quiet_button(popup, "Close", width=90, height=34, command=self.close_popup).pack(pady=(0, 12))

    def edit_current_line(self, runtime):
        """Open an editor for the highlighted transcript row."""
        row = self.ui.transcript_text.selected_row()
        if row is None:
            self.set_status("Click a line of the transcript first, then choose Edit.")
            return
        _tag, persona, old_text = row
        popup = self._popup("Edit line", "520x260")
        box = ctk.CTkTextbox(popup, font=theme.font(self.ui.text_size), fg_color=theme.SURFACE,
                             text_color=theme.TEXT, wrap="word")
        buttons = ctk.CTkFrame(popup, fg_color="transparent")
        buttons.pack(side="bottom", fill="x", padx=12, pady=10)
        box.pack(fill="both", expand=True, padx=12, pady=(12, 0))
        box.insert("1.0", old_text)

        def save_edit():
            new_text = box.get("1.0", "end-1c").strip()
            if new_text and new_text != old_text:
                self.ui.transcript_text.replace_selected_text(new_text)
                convo_id = runtime.convo.get_convo_id(persona=persona, input_text=old_text)
                runtime.convo.update_conversation_by_id(persona=persona, convo_id=convo_id, text=new_text)
            self.close_popup()

        primary_button(buttons, "Save", width=100, command=save_edit).pack(side="right")
        quiet_button(buttons, "Cancel", width=90, height=36, command=self.close_popup).pack(
            side="right", padx=(0, 8))
