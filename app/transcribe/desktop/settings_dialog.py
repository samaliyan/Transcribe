"""The settings window: API key and model, speech to text engine, the assistant's prompt.

Everything is stored in override.yaml (next to the program) through tsutils.configuration.
"""

from __future__ import annotations

import threading
import tkinter as tk

import customtkinter as ctk

from .. import gpt_responder, prompts
from ..uicomp import theme
from .view import primary_button, quiet_button

from tsutils import configuration

ENGINES = [
    # label, (General.stt, General.use_api)
    ("On this computer: free and private", ("whisper", False)),
    ("OpenAI API: more accurate, uses the API key (paid)", ("whisper", True)),
    ("OpenAI Realtime: streaming, uses the API key (paid)", ("openai-realtime", False)),
    ("Deepgram API: needs a Deepgram key (paid)", ("deepgram", False)),
]
LOCAL_MODELS = [
    ("tiny: fastest, least accurate", "tiny"),
    ("base: fast, comes with the program", "base"),
    ("small: more accurate, needs a good CPU", "small"),
    ("medium: accurate, slow without a GPU", "medium"),
    ("large-v3-turbo: very accurate, GPU recommended", "large-v3-turbo"),
    ("large-v3: most accurate, GPU needed", "large-v3"),
]
CUSTOM_PROMPT = "Custom"
# Changing these needs a restart (the engine is built once, at start-up).
RESTART_KEYS = {("General", "stt"), ("General", "use_api"), ("OpenAI", "local_transcripton_model_file"),
                ("Deepgram", "api_key"), ("General", "proxy"), ("General", "save_audio_log"),
                ("OpenAI", "stt_model")}


def diff_settings(current: dict, wanted: dict) -> dict:
    """The part of ``wanted`` ({section: {key: value}}) that differs from the current settings."""
    changes: dict = {}
    for section, values in wanted.items():
        for key, value in values.items():
            if (current.get(section) or {}).get(key) != value:
                changes.setdefault(section, {})[key] = value
    return changes


def needs_restart(changes: dict) -> bool:
    """True when one of the changed settings only takes effect at the next start."""
    return any((section, key) in RESTART_KEYS for section, values in changes.items() for key in values)


class SettingsDialog(ctk.CTkToplevel):
    """Modal settings window. ``on_saved(changes, restart)`` is called after a save."""

    def __init__(self, master, config: dict, on_saved, open_data_folder=None, focus_tab: str | None = None):
        super().__init__(master, fg_color=theme.BG)
        self.config_data = config
        self.on_saved = on_saved
        self.provider = "Together" if config["General"].get("chat_inference_provider") == "together" else "OpenAI"
        self.title("Settings")
        self.geometry("640x640")
        self.minsize(560, 520)
        self.transient(master)

        self.tabs = ctk.CTkTabview(self, fg_color=theme.SURFACE, segmented_button_fg_color=theme.SURFACE_2,
                                   segmented_button_selected_color=theme.ACCENT,
                                   segmented_button_selected_hover_color=theme.ACCENT_HOVER,
                                   segmented_button_unselected_color=theme.SURFACE_2,
                                   segmented_button_unselected_hover_color=theme.SURFACE_3,
                                   text_color=theme.TEXT)
        self.tabs.pack(fill="both", expand=True, padx=14, pady=(12, 6))
        for name in ("AI replies", "Speech to text", "Assistant", "General"):
            self.tabs.add(name)
        self._build_ai_tab(self.tabs.tab("AI replies"))
        self._build_stt_tab(self.tabs.tab("Speech to text"))
        self._build_assistant_tab(self.tabs.tab("Assistant"))
        self._build_general_tab(self.tabs.tab("General"), open_data_folder)
        if focus_tab:
            self.tabs.set(focus_tab)

        buttons = ctk.CTkFrame(self, fg_color="transparent")
        buttons.pack(fill="x", padx=14, pady=(4, 12))
        self.message_label = ctk.CTkLabel(buttons, text="", font=theme.font(12), text_color=theme.MUTED,
                                          anchor="w", justify="left", wraplength=330)
        self.message_label.pack(side="left", fill="x", expand=True)
        primary_button(buttons, "Save", width=110, command=self.save).pack(side="right")
        quiet_button(buttons, "Cancel", width=90, height=36, command=self.destroy).pack(side="right", padx=(0, 8))

        self.bind("<Escape>", lambda _event: self.destroy())
        self.after(120, self._take_focus)

    def _take_focus(self):
        try:
            self.lift()
            self.focus_force()
            self.grab_set()
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ small builders
    @staticmethod
    def _label(parent, text: str, muted: bool = False, **pack):
        label = ctk.CTkLabel(parent, text=text, font=theme.font(11 if muted else 12),
                             text_color=theme.MUTED if muted else theme.TEXT, anchor="w", justify="left",
                             wraplength=560)
        options = {"fill": "x", "padx": 12, "pady": (10, 2) if not muted else (0, 2)}
        options.update(pack)
        label.pack(**options)
        return label

    @staticmethod
    def _entry(parent, value, show: str = "", placeholder: str = ""):
        entry = ctk.CTkEntry(parent, height=32, font=theme.font(12), fg_color=theme.SURFACE_2,
                             border_color=theme.BORDER, text_color=theme.TEXT, show=show,
                             placeholder_text=placeholder)
        if value not in (None, ""):
            entry.insert(0, str(value))
        return entry

    @staticmethod
    def _menu(parent, values: list[str], current: str, command=None):
        menu = ctk.CTkOptionMenu(parent, values=values, command=command, height=32, font=theme.font(12),
                                 dropdown_font=theme.font(12), fg_color=theme.SURFACE_2,
                                 button_color=theme.SURFACE_2, button_hover_color=theme.SURFACE_3,
                                 text_color=theme.TEXT, dropdown_fg_color=theme.SURFACE_2,
                                 dropdown_hover_color=theme.SURFACE_3, dropdown_text_color=theme.TEXT,
                                 dynamic_resizing=False)
        menu.set(current)
        return menu

    @staticmethod
    def _textbox(parent, text: str, height: int):
        box = ctk.CTkTextbox(parent, height=height, font=theme.font(12), fg_color=theme.SURFACE_2,
                             border_color=theme.BORDER, border_width=1, text_color=theme.TEXT, wrap="word")
        box.insert("1.0", text or "")
        return box

    # ------------------------------------------------------------------ tabs
    def _build_ai_tab(self, tab):
        section = self.config_data[self.provider]
        self._label(tab, "API key")
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=12)
        key = section.get("api_key")
        self.api_key_entry = self._entry(row, key if gpt_responder.has_api_key(key) else "", show="•",
                                         placeholder="sk-…")
        self.api_key_entry.pack(side="left", fill="x", expand=True)
        self.show_key_button = quiet_button(row, "Show", width=60, height=32, command=self._toggle_key)
        self.show_key_button.pack(side="left", padx=(6, 0))
        self._label(tab, "The key of your AI provider. It is stored on this computer only, in override.yaml "
                         "next to the program.", muted=True)

        self._label(tab, "Base URL")
        self.base_url_entry = self._entry(tab, section.get("base_url"), placeholder="https://api.openai.com/v1")
        self.base_url_entry.pack(fill="x", padx=12)
        self._label(tab, "The address of the provider. Any OpenAI-compatible service works: enter the "
                         "address it gives you (for example OpenRouter, Together, a local server).", muted=True)

        self._label(tab, "Model")
        default_model = (configuration.Config().default_data.get(self.provider) or {}).get("ai_model", "")
        models = list(dict.fromkeys(m for m in (section.get("ai_model"), default_model) if m))
        self.model_box = ctk.CTkComboBox(tab, values=models or [""], height=32, font=theme.font(12),
                                         dropdown_font=theme.font(12), fg_color=theme.SURFACE_2,
                                         border_color=theme.BORDER, button_color=theme.SURFACE_3,
                                         button_hover_color=theme.MUTED, text_color=theme.TEXT,
                                         dropdown_fg_color=theme.SURFACE_2, dropdown_text_color=theme.TEXT)
        self.model_box.set(str(section.get("ai_model") or ""))
        self.model_box.pack(fill="x", padx=12)
        self._label(tab, "The model name exactly as the provider writes it. A small, fast model keeps "
                         "the replies quick.", muted=True)

        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(14, 0))
        self.test_button = quiet_button(row, "Test the connection", height=32, command=self._test_connection)
        self.test_button.pack(side="left")
        self.test_label = ctk.CTkLabel(row, text="", font=theme.font(12), text_color=theme.MUTED, anchor="w",
                                       justify="left", wraplength=380)
        self.test_label.pack(side="left", fill="x", expand=True, padx=(10, 0))

        self._label(tab, "Automatic replies: at most one every", pady=(18, 2))
        row = ctk.CTkFrame(tab, fg_color="transparent")
        row.pack(fill="x", padx=12)
        self.interval_label = ctk.CTkLabel(row, text="", font=theme.font(12), text_color=theme.TEXT, width=90,
                                           anchor="w")
        self.interval_slider = ctk.CTkSlider(row, from_=1, to=30, number_of_steps=29,
                                             progress_color=theme.ACCENT, button_color=theme.ACCENT,
                                             button_hover_color=theme.ACCENT_HOVER,
                                             command=self._show_interval)
        self.interval_slider.set(int(self.config_data["General"].get("llm_response_interval", 5)))
        self.interval_slider.pack(side="left", fill="x", expand=True)
        self.interval_label.pack(side="left", padx=(10, 0))
        self._show_interval(self.interval_slider.get())
        self._label(tab, "A lower number gives newer replies while people talk, and more requests (cost).",
                    muted=True, pady=(4, 2))

    def _build_stt_tab(self, tab):
        general = self.config_data["General"]
        current = (str(general.get("stt", "whisper")).lower(), bool(general.get("use_api")))
        if current[0] != "whisper":
            current = (current[0], False)
        self.engine_choices = dict(ENGINES)
        if current not in self.engine_choices.values():      # an engine set by hand in the yaml file
            self.engine_choices[f"{current[0]} (set in the settings file)"] = current
        engine_label = next(label for label, value in self.engine_choices.items() if value == current)
        self._label(tab, "Engine")
        self.engine_menu = self._menu(tab, list(self.engine_choices), engine_label)
        self.engine_menu.pack(fill="x", padx=12)
        self._label(tab, "On this computer nothing leaves the machine and no internet is needed. For "
                         "languages other than English (for example Persian), the online engines are much "
                         "more accurate than the small local models.", muted=True, pady=(4, 2))

        self._label(tab, "Model on this computer")
        self.model_choices = dict(LOCAL_MODELS)
        model = str(self.config_data["OpenAI"].get("local_transcripton_model_file") or "base")
        if model not in self.model_choices.values():
            self.model_choices[model] = model
        model_label = next(label for label, value in self.model_choices.items() if value == model)
        self.local_model_menu = self._menu(tab, list(self.model_choices), model_label)
        self.local_model_menu.pack(fill="x", padx=12)
        self._label(tab, "A model other than base is downloaded once, when it is first used "
                         "(tiny 75 MB, small 480 MB, medium 1.5 GB, large 1.6 to 3 GB).", muted=True, pady=(4, 2))

        self._label(tab, "Model for the OpenAI API engine")
        self.stt_model_entry = self._entry(tab, self.config_data["OpenAI"].get("stt_model"),
                                           placeholder="whisper-1")
        self.stt_model_entry.pack(fill="x", padx=12)

        self._label(tab, "Deepgram API key")
        key = self.config_data["Deepgram"].get("api_key")
        self.deepgram_entry = self._entry(tab, key if gpt_responder.has_api_key(key) else "", show="•")
        self.deepgram_entry.pack(fill="x", padx=12)

        self._label(tab, "Changes on this page take effect after the program restarts; it offers to do "
                         "that when you save.", muted=True, pady=(16, 2))

    def _build_assistant_tab(self, tab):
        general = self.config_data["General"]
        prompt = str(general.get("system_prompt") or "")
        preset = next((name for name, text in prompts.PROMPT_PRESETS.items() if text == prompt), CUSTOM_PROMPT)
        self._label(tab, "Role of the assistant")
        self.preset_menu = self._menu(tab, list(prompts.PROMPT_PRESETS) + [CUSTOM_PROMPT], preset,
                                      command=self._apply_preset)
        self.preset_menu.pack(fill="x", padx=12)
        self._label(tab, "Instructions (you can edit them, or add facts about you and the meeting: your "
                         "name, your job, the subject, what you want from the conversation)", pady=(12, 2))
        self.prompt_box = self._textbox(tab, prompt, 210)
        self.prompt_box.pack(fill="both", expand=True, padx=12)
        self.prompt_box.bind("<KeyRelease>", lambda _event: self.preset_menu.set(self._preset_of_box()))
        self._label(tab, "Instruction for Summarize")
        self.summary_entry = self._entry(tab, general.get("summary_prompt"))
        self.summary_entry.pack(fill="x", padx=12, pady=(0, 8))

    def _build_general_tab(self, tab, open_data_folder):
        general = self.config_data["General"]
        self._label(tab, "Proxy (optional)")
        self.proxy_entry = self._entry(tab, general.get("proxy"), placeholder="http://127.0.0.1:8080")
        self.proxy_entry.pack(fill="x", padx=12)
        self._label(tab, "Used for the AI provider and for model downloads. Leave it empty to use the "
                         "Windows proxy settings (a VPN usually needs nothing here).", muted=True, pady=(4, 2))

        self.save_audio_var = tk.BooleanVar(value=bool(general.get("save_audio_log", False)))
        ctk.CTkCheckBox(tab, text="Keep a sound recording of each session (large files)",
                        variable=self.save_audio_var, font=theme.font(12), text_color=theme.TEXT,
                        fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                        border_color=theme.MUTED).pack(anchor="w", padx=12, pady=(18, 4))
        self.save_responses_var = tk.BooleanVar(value=bool(general.get("save_llm_response_to_file", True)))
        ctk.CTkCheckBox(tab, text="Keep the suggested replies in a text file",
                        variable=self.save_responses_var, font=theme.font(12), text_color=theme.TEXT,
                        fg_color=theme.ACCENT, hover_color=theme.ACCENT_HOVER,
                        border_color=theme.MUTED).pack(anchor="w", padx=12, pady=4)
        self._label(tab, "The transcript of every session is saved automatically when the program closes.",
                    muted=True, pady=(8, 2))
        if open_data_folder is not None:
            quiet_button(tab, "Open the folder with transcripts and logs", height=32,
                         command=open_data_folder).pack(anchor="w", padx=12, pady=(12, 0))

    # ------------------------------------------------------------------ behaviour
    def _toggle_key(self):
        hidden = self.api_key_entry.cget("show") != ""
        self.api_key_entry.configure(show="" if hidden else "•")
        self.show_key_button.configure(text="Hide" if hidden else "Show")

    def _show_interval(self, value):
        seconds = int(round(float(value)))
        self.interval_label.configure(text=f"{seconds} second" + ("" if seconds == 1 else "s"))

    def _preset_of_box(self) -> str:
        text = self.prompt_box.get("1.0", "end-1c").strip()
        return next((name for name, preset in prompts.PROMPT_PRESETS.items() if preset == text), CUSTOM_PROMPT)

    def _apply_preset(self, name: str):
        if name in prompts.PROMPT_PRESETS:
            self.prompt_box.delete("1.0", "end")
            self.prompt_box.insert("1.0", prompts.PROMPT_PRESETS[name])

    def _test_connection(self):
        api_key = self.api_key_entry.get().strip()
        base_url = self.base_url_entry.get().strip()
        model = self.model_box.get().strip()
        self.test_button.configure(state="disabled")
        self.test_label.configure(text="Testing…", text_color=theme.MUTED)

        def run():
            worked, message = gpt_responder.test_connection(api_key, base_url, model)
            self._from_thread(lambda: self._show_test_result(worked, message))

        threading.Thread(target=run, name="TestConnection", daemon=True).start()

    def _from_thread(self, action):
        try:
            self.after(0, action)
        except (tk.TclError, RuntimeError):
            pass                        # the window was closed meanwhile

    def _show_test_result(self, worked: bool, message: str):
        if not self.winfo_exists():
            return
        self.test_button.configure(state="normal")
        self.test_label.configure(text=message, text_color=theme.OK if worked else theme.DANGER)

    def collect(self) -> dict:
        """The settings as the window shows them now, as {section: {key: value}}."""
        stt, use_api = self.engine_choices[self.engine_menu.get()]
        api_key = self.api_key_entry.get().strip()
        deepgram_key = self.deepgram_entry.get().strip()
        base_url = self.base_url_entry.get().strip()
        wanted = {
            "General": {
                "llm_response_interval": int(round(float(self.interval_slider.get()))),
                "stt": stt,
                "use_api": use_api,
                "system_prompt": self.prompt_box.get("1.0", "end-1c").strip(),
                "summary_prompt": self.summary_entry.get().strip(),
                "proxy": self.proxy_entry.get().strip(),
                "save_audio_log": bool(self.save_audio_var.get()),
                "save_llm_response_to_file": bool(self.save_responses_var.get()),
            },
            "OpenAI": {
                "local_transcripton_model_file": self.model_choices[self.local_model_menu.get()],
                "stt_model": self.stt_model_entry.get().strip() or "whisper-1",
            },
            "Deepgram": {"api_key": deepgram_key or "API_KEY"},
        }
        # (the provider's section is "OpenAI" itself unless Together is the chat provider)
        wanted.setdefault(self.provider, {}).update({
            "api_key": api_key or "API_KEY",
            "base_url": base_url or configuration.Config().default_data[self.provider].get("base_url"),
            "ai_model": self.model_box.get().strip(),
        })
        return wanted

    def validate(self, wanted: dict) -> str | None:
        """Why these settings cannot be saved (None: they can)."""
        if not wanted[self.provider]["ai_model"]:
            return "Enter a model name on the AI replies page."
        if not wanted["General"]["system_prompt"]:
            return "The instructions on the Assistant page are empty."
        stt, use_api = wanted["General"]["stt"], wanted["General"]["use_api"]
        openai_key = wanted["OpenAI"].get("api_key", self.config_data["OpenAI"].get("api_key"))
        if (use_api or stt == "openai-realtime") and not gpt_responder.has_api_key(openai_key):
            return "This speech to text engine needs the API key (AI replies page)."
        if stt == "deepgram" and not gpt_responder.has_api_key(wanted["Deepgram"]["api_key"]):
            return "The Deepgram engine needs a Deepgram API key."
        proxy = wanted["General"]["proxy"]
        if proxy and "://" not in proxy:
            return "The proxy must look like http://address:port"
        return None

    def save(self):
        wanted = self.collect()
        problem = self.validate(wanted)
        if problem:
            self.message_label.configure(text=problem, text_color=theme.DANGER)
            return
        changes = diff_settings(self.config_data, wanted)
        restart = needs_restart(changes)
        online_stt = wanted["General"]["use_api"] or wanted["General"]["stt"] == "openai-realtime"
        if online_stt and any(key in changes.get("OpenAI", {}) for key in ("api_key", "base_url")):
            restart = True              # the online speech to text engine uses them too
        if changes:
            try:
                configuration.Config().add_override_value(changes)
            except OSError as exception:
                self.message_label.configure(
                    text=f"The settings could not be written ({exception}).", text_color=theme.DANGER)
                return
        on_saved = self.on_saved
        self.destroy()
        on_saved(changes, restart)
