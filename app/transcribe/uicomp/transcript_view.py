"""The transcript panel: one block per spoken row, coloured by who spoke.

A row is addressed by a key, so the live update of a row replaces that block in place
instead of searching the text. Rows are given as the conversation stores them
('You: [text]'); the label and the text are shown separately, without the brackets.
"""

from __future__ import annotations

import datetime
import tkinter as tk

import customtkinter as ctk

from . import theme
from .. import constants, prompts

LABELS = {constants.PERSONA_YOU: "You", constants.PERSONA_SPEAKER: "Speaker"}
PLACEHOLDER = ("Listening…\n\nWhat you say (microphone) and what the others say (the sound of this "
               "computer) appears here.")


class TranscriptView(ctk.CTkFrame):
    """Read-only transcript with selectable rows and a right-click menu."""

    def __init__(self, master=None, text_size: int = theme.DEFAULT_TEXT_SIZE, **kwargs):
        super().__init__(master, fg_color=theme.SURFACE, corner_radius=0, **kwargs)
        self.text_widget = tk.Text(
            self, wrap="word", width=30, height=5, borderwidth=0, highlightthickness=0,
            background=theme.SURFACE, foreground=theme.TEXT, insertbackground=theme.SURFACE,
            selectbackground=theme.SELECT, selectforeground=theme.TEXT, inactiveselectbackground=theme.SELECT,
            padx=14, pady=10, spacing1=1, spacing3=1, cursor="arrow", state="disabled")
        self.scrollbar = ctk.CTkScrollbar(self, command=self.text_widget.yview, fg_color=theme.SURFACE,
                                          button_color=theme.SURFACE_3, button_hover_color=theme.MUTED)
        self.text_widget.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.text_widget.pack(side="left", fill="both", expand=True)

        self._rows: dict[str, tuple[str, str]] = {}      # tag -> (persona, text)
        self._stamps: dict[str, str] = {}               # tag -> time the row started
        self._selected_tag: str | None = None
        self._has_placeholder = False
        self.on_row_selected = None                     # callback(persona, text)
        self.context_menu = tk.Menu(self, tearoff=0)
        self.set_text_size(text_size)
        self.text_widget.tag_configure("selected", background=theme.SURFACE_2)
        self.text_widget.tag_raise("sel")
        self.text_widget.bind("<ButtonRelease-1>", self._on_click)
        self.text_widget.bind("<Button-3>", self._show_context_menu)
        self._show_placeholder()

    # ------------------------------------------------------------------ appearance
    def set_text_size(self, size: int):
        widget = self.text_widget
        widget.configure(font=theme.font(size))
        label_size = max(9, size - 4)
        widget.tag_configure("label-You", foreground=theme.YOU, font=theme.font(label_size, "bold"), spacing1=10)
        widget.tag_configure("label-Speaker", foreground=theme.SPEAKER, font=theme.font(label_size, "bold"),
                             spacing1=10)
        widget.tag_configure("label-other", foreground=theme.MUTED, font=theme.font(label_size, "bold"), spacing1=10)
        widget.tag_configure("time", foreground=theme.MUTED, font=theme.font(label_size))
        widget.tag_configure("rtl", justify="right")
        widget.tag_configure("placeholder", foreground=theme.MUTED, font=theme.font(max(10, size - 2)))

    # ------------------------------------------------------------------ menu
    def add_right_click_menu(self, label, command):
        self.context_menu.add_command(label=label, command=command)

    def add_right_menu_separator(self):
        self.context_menu.add_separator()

    def _show_context_menu(self, event):
        self._select_at(event)
        try:
            self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    # ------------------------------------------------------------------ content
    @staticmethod
    def _row_tag(row_key: str) -> str:
        return f"transcript-row-{row_key}"

    def _show_placeholder(self):
        self._edit(lambda: self.text_widget.insert("1.0", PLACEHOLDER, ("placeholder",)))
        self._has_placeholder = True

    def _edit(self, action):
        self.text_widget.configure(state="normal")
        try:
            return action()
        finally:
            self.text_widget.configure(state="disabled")

    def _near_bottom(self) -> bool:
        return self.text_widget.yview()[1] >= 0.98

    def _insert_block(self, index, tag: str, persona: str, text: str, stamp: str):
        """Insert 'label  time / text' at index; every character carries the row's tag."""
        widget = self.text_widget
        label_tag = f"label-{persona}" if persona in LABELS else "label-other"
        direction = ("rtl",) if theme.is_rtl(text) else ()
        widget.mark_set("row-insert", index)
        widget.mark_gravity("row-insert", "right")
        widget.insert("row-insert", LABELS.get(persona, persona or "?"), (tag, label_tag) + direction)
        widget.insert("row-insert", f"   {stamp}\n", (tag, "time") + direction)
        widget.insert("row-insert", f"{text}\n", (tag,) + direction)

    def upsert_keyed_row(self, row_key: str, input_text: str):
        """Insert a row, or replace it where it is. The view follows new text only while it
        is scrolled to the end, so reading earlier rows is not interrupted."""
        tag = self._row_tag(row_key)
        persona, text = prompts.split_row(input_text)
        follow = self._near_bottom()

        def action():
            widget = self.text_widget
            if self._has_placeholder:
                widget.delete("1.0", tk.END)
                self._has_placeholder = False
            ranges = widget.tag_ranges(tag)
            if ranges:
                start = widget.index(ranges[0])
                stamp = self._stamps.get(tag, "")
                selected = tag == self._selected_tag
                widget.delete(ranges[0], ranges[-1])
                self._insert_block(start, tag, persona, text, stamp)
                if selected:
                    self._highlight(tag)
            else:
                stamp = datetime.datetime.now().strftime("%H:%M")
                self._stamps[tag] = stamp
                self._insert_block(tk.END, tag, persona, text, stamp)
            self._rows[tag] = (persona, text)

        self._edit(action)
        if follow:
            self.scroll_to_bottom()

    def release_keyed_row(self, row_key: str):
        """The row will not be updated through this key again (its text stays where it is)."""

    def finalize_keyed_row(self, row_key: str, input_text: str):
        """Replace a provisional row with its final text."""
        self.upsert_keyed_row(row_key, input_text)

    def add_text_to_bottom(self, input_text: str):
        """Append a row that will not be updated later."""
        self.upsert_keyed_row(f"static-{len(self._rows)}-{id(input_text)}", input_text)

    def clear_all_text(self):
        self._edit(lambda: self.text_widget.delete("1.0", tk.END))
        for tag in list(self._rows):
            self.text_widget.tag_delete(tag)
        self._rows.clear()
        self._stamps = {}
        self._selected_tag = None
        self._show_placeholder()

    def scroll_to_bottom(self):
        self.text_widget.yview_moveto(1)

    def row_count(self) -> int:
        return len(self._rows)

    # ------------------------------------------------------------------ selection
    def _tag_at(self, index) -> str | None:
        for tag in self.text_widget.tag_names(index):
            if tag in self._rows:
                return tag
        return None

    def _highlight(self, tag: str | None):
        widget = self.text_widget
        widget.tag_remove("selected", "1.0", tk.END)
        self._selected_tag = tag
        if tag is not None:
            ranges = widget.tag_ranges(tag)
            if ranges:
                widget.tag_add("selected", ranges[0], ranges[-1])

    def _select_at(self, event) -> str | None:
        tag = self._tag_at(self.text_widget.index(f"@{event.x},{event.y}"))
        if tag is not None:
            self._highlight(tag)
        return tag

    def _on_click(self, event):
        if self.text_widget.tag_ranges("sel"):
            return                  # the user is selecting text with the mouse
        tag = self._select_at(event)
        if tag is not None and self.on_row_selected is not None:
            self.on_row_selected(*self._rows[tag])

    def selected_row(self):
        """(key tag, persona, text) of the highlighted row, or None."""
        if self._selected_tag is None or self._selected_tag not in self._rows:
            return None
        return (self._selected_tag,) + self._rows[self._selected_tag]

    def replace_selected_text(self, text: str):
        """Show an edited text in the highlighted row."""
        row = self.selected_row()
        if row is not None:
            tag, persona, _ = row
            self.upsert_keyed_row(tag[len("transcript-row-"):], f"{persona}: [{text}]")

    def selection_get(self) -> str:      # pylint: disable=arguments-differ
        """The text selected with the mouse; without one, the highlighted row's text."""
        try:
            return self.text_widget.get("sel.first", "sel.last")
        except tk.TclError:
            row = self.selected_row()
            return row[2] if row else ""

    def get_plain_text(self) -> str:
        return "" if self._has_placeholder else self.text_widget.get("1.0", "end-1c")
