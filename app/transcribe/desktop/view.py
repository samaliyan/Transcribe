"""Layout of the main window. Only builds widgets; commands are attached by bindings.py."""

from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from ..uicomp import theme
from ..uicomp.transcript_view import TranscriptView

from tsutils.language import LANGUAGES_DICT

COMMON_LANGUAGES = ["english", "persian", "arabic", "turkish", "german", "french", "spanish",
                    "russian", "chinese", "hindi", "urdu", "italian", "portuguese", "japanese", "korean"]


def language_names() -> list[str]:
    """Language names for the menus: the common ones first, then the rest in alphabetical order."""
    known = set(LANGUAGES_DICT.values())
    first = [name for name in COMMON_LANGUAGES if name in known]
    rest = sorted(known - set(first))
    return [name.title() for name in first + rest]


def quiet_button(master, text: str, width: int = 0, **kwargs):
    """A low-key button for toolbars."""
    options = {"height": 28, "corner_radius": 6, "fg_color": theme.SURFACE_2, "hover_color": theme.SURFACE_3,
               "text_color": theme.TEXT, "font": theme.font(12), "border_width": 0}
    options.update(kwargs)
    return ctk.CTkButton(master, text=text, width=width or 10, **options)


def primary_button(master, text: str, **kwargs):
    options = {"height": 36, "corner_radius": 8, "fg_color": theme.ACCENT, "hover_color": theme.ACCENT_HOVER,
               "text_color": "#ffffff", "font": theme.font(13, "bold")}
    options.update(kwargs)
    return ctk.CTkButton(master, text=text, **options)


class DesktopViewBuilder:
    """Build the desktop window layout and widgets."""

    def build(self, ui, config: dict):
        """Create all widgets required by the desktop window."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        ui.title("Transcribe")
        ui.configure(fg_color=theme.BG)
        ui.geometry(self._geometry(ui, config))
        ui.minsize(760, 460)
        ui.text_size = theme.clamp_text_size(config["General"].get("ui_font_size", theme.DEFAULT_TEXT_SIZE))

        ui.grid_columnconfigure(0, weight=1)
        ui.grid_rowconfigure(2, weight=1)
        self.create_header(ui)
        self.create_panels(ui)
        self.create_action_bar(ui, config)
        self.create_status_bar(ui)
        self.create_more_menu(ui)

    @staticmethod
    def _geometry(ui, config: dict) -> str:
        """The size the window had last time, if it still fits the screen."""
        saved = str(config["General"].get("window_geometry") or "")
        try:
            size = saved.split("+")[0]
            width, height = (int(part) for part in size.split("x"))
            if 760 <= width <= ui.winfo_screenwidth() and 460 <= height <= ui.winfo_screenheight():
                return size
        except ValueError:
            pass
        width = min(1180, ui.winfo_screenwidth() - 80)
        height = min(760, ui.winfo_screenheight() - 120)
        return f"{width}x{height}"

    # ------------------------------------------------------------------ header
    def create_header(self, ui):
        header = ctk.CTkFrame(ui, fg_color=theme.BG, corner_radius=0, height=52)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        ui.header = header

        ctk.CTkLabel(header, text="Transcribe", font=theme.font(17, "bold"), text_color=theme.TEXT).pack(
            side="left", padx=(2, 16))

        ui.settings_button = quiet_button(header, "Settings", height=30)
        ui.settings_button.pack(side="right")
        ui.more_button = quiet_button(header, "More ▾", height=30)
        ui.more_button.pack(side="right", padx=(0, 6))
        ui.pause_button = quiet_button(header, "Pause", height=30)
        ui.pause_button.pack(side="right", padx=(0, 6))

        ui.mic_button = quiet_button(header, "●  Microphone", height=30, text_color_disabled=theme.WARN)
        ui.mic_button.pack(side="left", padx=(0, 6))
        ui.speaker_button = quiet_button(header, "●  Speaker", height=30, text_color_disabled=theme.WARN)
        ui.speaker_button.pack(side="left", padx=(0, 6))
        ui.stt_status_label = ctk.CTkLabel(header, text="", font=theme.font(12), text_color=theme.MUTED,
                                           anchor="w")
        ui.stt_status_label.pack(side="left", padx=(8, 0))

        # A full-width notice under the header, shown while speech to text cannot work.
        ui.stt_notice = ctk.CTkFrame(ui, fg_color="#3a2326", corner_radius=8)
        ui.stt_notice_label = ctk.CTkLabel(ui.stt_notice, text="", font=theme.font(12), text_color="#ffd9da",
                                           justify="left", anchor="w", wraplength=1000)
        ui.stt_notice_label.pack(fill="x", padx=12, pady=7)
        ui.stt_notice.bind("<Configure>", lambda event: ui.stt_notice_label.configure(
            wraplength=max(200, event.width - 30)))
        ui.stt_notice_visible = False

    # ------------------------------------------------------------------ panels
    def create_panels(self, ui):
        ui.panes = tk.PanedWindow(ui, orient="horizontal", sashwidth=8, bd=0, bg=theme.BG,
                                  sashrelief="flat", opaqueresize=True)
        ui.panes.grid(row=2, column=0, sticky="nsew", padx=14, pady=0)

        # --- transcript
        left = ctk.CTkFrame(ui.panes, fg_color=theme.SURFACE, corner_radius=10, border_width=1,
                            border_color=theme.BORDER)
        bar = ctk.CTkFrame(left, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(bar, text="TRANSCRIPT", font=theme.font(11, "bold"), text_color=theme.MUTED).pack(side="left")
        ui.clear_button = quiet_button(bar, "Clear")
        ui.clear_button.pack(side="right")
        ui.save_button = quiet_button(bar, "Save…")
        ui.save_button.pack(side="right", padx=(0, 6))
        ui.copy_transcript_button = quiet_button(bar, "Copy")
        ui.copy_transcript_button.pack(side="right", padx=(0, 6))
        ui.transcript_text = TranscriptView(left, text_size=ui.text_size)
        ui.transcript_text.pack(fill="both", expand=True, padx=2, pady=(2, 8))

        # --- suggested reply
        right = ctk.CTkFrame(ui.panes, fg_color=theme.SURFACE, corner_radius=10, border_width=1,
                             border_color=theme.BORDER)
        bar = ctk.CTkFrame(right, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=(10, 2))
        ctk.CTkLabel(bar, text="SUGGESTED REPLY", font=theme.font(11, "bold"), text_color=theme.MUTED).pack(
            side="left")
        ui.ai_status_label = ctk.CTkLabel(bar, text="", font=theme.font(11), text_color=theme.MUTED)
        ui.ai_status_label.pack(side="left", padx=(10, 0))
        ui.read_button = quiet_button(bar, "Read aloud")
        ui.read_button.pack(side="right")
        ui.copy_response_button = quiet_button(bar, "Copy")
        ui.copy_response_button.pack(side="right", padx=(0, 6))
        ui.live_button = quiet_button(bar, "Back to live", fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                                      text_color="#ffffff")
        # (packed only while an earlier reply is shown)

        # A notice with one action, shown when replies cannot work (no key, an error).
        ui.ai_banner = ctk.CTkFrame(right, fg_color=theme.SURFACE_2, corner_radius=8)
        ui.ai_banner_button = quiet_button(ui.ai_banner, "Open Settings", width=110, fg_color=theme.ACCENT,
                                           hover_color=theme.ACCENT_HOVER, text_color="#ffffff")
        ui.ai_banner_button.pack(side="right", padx=10, pady=8)
        ui.ai_banner_label = ctk.CTkLabel(ui.ai_banner, text="", font=theme.font(12), text_color=theme.TEXT,
                                          justify="left", anchor="w", wraplength=300)
        ui.ai_banner_label.pack(side="left", fill="x", expand=True, padx=12, pady=8)
        ui.ai_banner.bind("<Configure>", lambda event: ui.ai_banner_label.configure(
            wraplength=max(120, event.width - 170)))
        ui.ai_banner_visible = False

        ui.response_frame = ctk.CTkFrame(right, fg_color="transparent")
        ui.response_frame.pack(fill="both", expand=True, padx=2, pady=(2, 8))
        ui.response_textbox = tk.Text(
            ui.response_frame, wrap="word", width=30, height=5, borderwidth=0, highlightthickness=0,
            background=theme.SURFACE, foreground=theme.TEXT, insertbackground=theme.SURFACE,
            selectbackground=theme.SELECT, selectforeground=theme.TEXT, inactiveselectbackground=theme.SELECT,
            padx=14, pady=10, spacing1=2, spacing3=4, cursor="arrow", state="disabled")
        response_scroll = ctk.CTkScrollbar(ui.response_frame, command=ui.response_textbox.yview,
                                           fg_color=theme.SURFACE, button_color=theme.SURFACE_3,
                                           button_hover_color=theme.MUTED)
        ui.response_textbox.configure(yscrollcommand=response_scroll.set)
        response_scroll.pack(side="right", fill="y")
        ui.response_textbox.pack(side="left", fill="both", expand=True)

        ui.panes.add(left, minsize=260, stretch="always")
        ui.panes.add(right, minsize=260, stretch="always")
        ui.left_panel, ui.right_panel = left, right

    # ------------------------------------------------------------------ action bar
    def create_action_bar(self, ui, config: dict):
        bar = ctk.CTkFrame(ui, fg_color=theme.BG, corner_radius=0)
        bar.grid(row=3, column=0, sticky="ew", padx=14, pady=(8, 4))
        ui.action_bar = bar

        ui.response_now_button = primary_button(bar, "Suggest reply now   (F2)", width=190)
        ui.response_now_button.pack(side="left")
        ui.auto_switch = ctk.CTkSwitch(bar, text="Suggest automatically", font=theme.font(12),
                                       text_color=theme.TEXT, progress_color=theme.ACCENT,
                                       button_color="#d7dbe3", button_hover_color="#ffffff",
                                       fg_color=theme.SURFACE_3)
        if config["General"]["continuous_response"]:
            ui.auto_switch.select()
        ui.auto_switch.pack(side="left", padx=(14, 0))
        ui.summarize_button = quiet_button(bar, "Summarize", height=36, font=theme.font(12))
        ui.summarize_button.pack(side="left", padx=(14, 0))

        ui.font_plus_button = quiet_button(bar, "A+", width=34, height=30)
        ui.font_plus_button.pack(side="right")
        ui.font_minus_button = quiet_button(bar, "A−", width=34, height=30)
        ui.font_minus_button.pack(side="right", padx=(0, 4))

        names = language_names()
        menu_style = {"width": 118, "height": 30, "font": theme.font(12), "dropdown_font": theme.font(12),
                      "fg_color": theme.SURFACE_2, "button_color": theme.SURFACE_2,
                      "button_hover_color": theme.SURFACE_3, "text_color": theme.TEXT,
                      "dropdown_fg_color": theme.SURFACE_2, "dropdown_hover_color": theme.SURFACE_3,
                      "dropdown_text_color": theme.TEXT, "dynamic_resizing": False}
        ui.response_lang_combobox = ctk.CTkOptionMenu(bar, values=names, **menu_style)
        ui.response_lang_combobox.set(str(config["OpenAI"]["response_lang"]).title())
        ui.response_lang_combobox.pack(side="right", padx=(0, 14))
        ctk.CTkLabel(bar, text="Reply in", font=theme.font(12), text_color=theme.MUTED).pack(
            side="right", padx=(0, 6))
        ui.audio_lang_combobox = ctk.CTkOptionMenu(bar, values=names, **menu_style)
        ui.audio_lang_combobox.set(str(config["OpenAI"]["audio_lang"]).title())
        ui.audio_lang_combobox.pack(side="right", padx=(0, 12))
        ctk.CTkLabel(bar, text="Spoken language", font=theme.font(12), text_color=theme.MUTED).pack(
            side="right", padx=(0, 6))

    # ------------------------------------------------------------------ status bar
    def create_status_bar(self, ui):
        bar = ctk.CTkFrame(ui, fg_color=theme.BG, corner_radius=0)
        bar.grid(row=4, column=0, sticky="ew", padx=16, pady=(0, 8))
        ui.status_label = ctk.CTkLabel(bar, text="", font=theme.font(11), text_color=theme.MUTED, anchor="w")
        ui.status_label.pack(side="left", fill="x", expand=True)
        ui.version_label = ctk.CTkLabel(bar, text="", font=theme.font(11), text_color=theme.MUTED)
        ui.version_label.pack(side="right")

    def create_more_menu(self, ui):
        ui.more_menu = tk.Menu(ui, tearoff=False, bg=theme.SURFACE_2, fg=theme.TEXT,
                               activebackground=theme.ACCENT, activeforeground="#ffffff", bd=0)
