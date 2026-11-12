"""The main window of Transcribe."""
import queue
import time
import tkinter as tk

import customtkinter as ctk

from .desktop import DesktopCommandBinder, DesktopController, DesktopDisplayManager, DesktopViewBuilder
from .desktop.settings_dialog import SettingsDialog
from .global_vars import AppRuntime, create_app_runtime
from .uicomp import theme
from tsutils import app_logging as al


logger = al.get_module_logger(al.UI_LOGGER)
UI_POLL_INTERVAL_MS = 50


class AppUI(ctk.CTk):
    """Encapsulates all UI functionality for the app
    """
    global_vars: AppRuntime
    text_size: int = theme.DEFAULT_TEXT_SIZE

    def __init__(
        self,
        config: dict,
        runtime: AppRuntime = None,
        controller: DesktopController = None,
        view_builder: DesktopViewBuilder = None,
        command_binder: DesktopCommandBinder = None,
        display_manager: DesktopDisplayManager = None,
    ):
        super().__init__()
        self.config_data = config
        self.global_vars = runtime or create_app_runtime()
        self.controller = controller or DesktopController(config=config, runtime=self.global_vars)
        self.view_builder = view_builder or DesktopViewBuilder()
        self.command_binder = command_binder or DesktopCommandBinder()
        self.display_manager = display_manager or DesktopDisplayManager()
        self._ui_action_queue: queue.Queue = queue.Queue()
        self._settings_window = None
        self._closing = False
        self.topmost_var = tk.BooleanVar(value=bool(config["General"].get("always_on_top", False)))

        self.global_vars.main_window = self
        self.controller.bind_ui(self)
        self.display_manager.bind_ui(self)
        self.create_ui_components(config=config)
        self.set_audio_device_menus(config=config)
        self.toggle_topmost(save=False)
        self.after(UI_POLL_INTERVAL_MS, self.process_ui_actions)

    def start(self):
        """Start showing the UI
        """
        self.mainloop()

    def report_callback_exception(self, exc, val, tb):  # pylint: disable=arguments-differ
        """An error in a button or timer: log it and say so, the window keeps working."""
        logger.error("Error in the window", exc_info=(exc, val, tb))
        try:
            self.set_status(f"Something went wrong: {val}", theme.DANGER)
        except Exception:  # pylint: disable=broad-except
            pass

    @staticmethod
    def tk_now() -> float:
        """A clock for timed things in the window."""
        return time.monotonic()

    # ------------------------------------------------------------------ work from other threads
    def enqueue_ui_action(self, callback, *args, **kwargs):
        """Queue UI work for the Tk main loop."""
        self._ui_action_queue.put((callback, args, kwargs))

    def process_ui_actions(self):
        """Run queued UI actions on the Tk main loop."""
        try:
            while True:
                callback, args, kwargs = self._ui_action_queue.get_nowait()
                try:
                    callback(*args, **kwargs)
                except Exception as exception:
                    logger.error(f"Error processing queued UI action: {exception}")
        except queue.Empty:
            pass

        if not self._closing:
            self.after(UI_POLL_INTERVAL_MS, self.process_ui_actions)

    # ------------------------------------------------------------------ building
    def create_ui_components(self, config: dict):
        """Create all UI components."""
        self.view_builder.build(self, config)
        self.command_binder.bind(self, config)

    def set_audio_device_menus(self, config):
        del config
        self.controller.set_audio_device_menus()

    def update_initial_transcripts(self):
        """Connect the conversation to the window and start refreshing it."""
        return self.display_manager.update_initial_transcripts(self.global_vars)

    def set_version(self, version: str):
        """Show the program version in the corner."""
        self.version_label.configure(text=version)

    # ------------------------------------------------------------------ display (also called from queued actions)
    def update_last_row(self, speaker: str, input_text: str):
        return self.display_manager.update_last_row(speaker, input_text)

    def queue_update_last_row(self, speaker: str, input_text: str):
        return self.display_manager.queue_update_last_row(speaker, input_text)

    def queue_add_transcript_line(self, input_text: str, persona: str = None):
        return self.display_manager.queue_add_transcript_line(input_text, persona)

    def write_response_text(self, response_string: str):
        return self.display_manager.write_response_text(response_string)

    def close_popup(self):
        return self.display_manager.close_popup()

    def show_loading_popup(self, title: str, msg: str):
        return self.display_manager.show_loading_popup(title, msg)

    def show_message_popup(self, title: str, msg: str):
        return self.display_manager.show_message_popup(title, msg)

    def show_word_cloud_popup(self, title: str, word_cloud):
        return self.display_manager.show_word_cloud_popup(title, word_cloud)

    def set_status(self, text: str, colour: str = None, keep: bool = False):
        return self.display_manager.set_status(text, colour, keep)

    def back_to_live(self):
        """Leave an earlier reply and show the newest one again."""
        return self.display_manager.back_to_live()

    def show_reply_for_row(self, persona: str, text: str):
        """A transcript row was clicked: show the reply that was suggested at that point, if any."""
        try:
            reply = self.global_vars.convo.get_response_for_row(persona, text)
        except Exception as exception:  # pylint: disable=broad-except
            logger.error(f"Error looking up the reply of a row: {exception}")
            return
        if reply:
            self.display_manager.show_history(reply)
            self.set_status("Showing the reply suggested at that point. 'Back to live' returns to the newest.")

    def change_text_size(self, step: int):
        """Make the transcript and reply text larger or smaller, and remember it."""
        size = theme.clamp_text_size(self.text_size + step)
        if size == self.text_size:
            return
        self.display_manager.apply_text_size(size)
        self.controller.settings_service.save_value("General", "ui_font_size", size)

    def copy_text(self, text: str):
        """Put text on the clipboard."""
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update_idletasks()

    def copy_response(self):
        """Copy the reply that is on screen."""
        text = self.display_manager.current_reply_text()
        if not text:
            self.set_status("There is no reply to copy yet.")
            return
        self.copy_text(text)
        self.set_status("Reply copied.")

    def show_more_menu(self):
        """Open the 'More' menu under its button."""
        button = self.more_button
        try:
            self.more_menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height() + 2)
        finally:
            self.more_menu.grab_release()

    def toggle_topmost(self, save: bool = True):
        """Keep the window above other windows (useful next to a meeting)."""
        on = bool(self.topmost_var.get())
        try:
            self.attributes("-topmost", on)
        except tk.TclError:
            pass
        if save:
            self.controller.settings_service.save_value("General", "always_on_top", on)

    def clear_transcript(self):
        """Clear transcript from all places where it exists.
        """
        self.global_vars.transcriber.clear_transcriber_context(self.global_vars.audio_queue)
        self.transcript_text.clear_all_text()
        self.display_manager.reset_rows()     # (row updates still queued from before are skipped)
        self.back_to_live()
        self.set_status("The transcript was cleared.")

    def edit_current_line(self):
        """Edit the highlighted transcript row."""
        try:
            return self.display_manager.edit_current_line(self.global_vars)
        except tk.TclError:
            return None

    # ------------------------------------------------------------------ settings and closing
    def open_settings(self, tab: str = None):
        """Open the settings window (once)."""
        if self._settings_window is not None and self._settings_window.winfo_exists():
            self._settings_window.lift()
            self._settings_window.focus_force()
            return
        self._settings_window = SettingsDialog(
            self, self.config_data, on_saved=self._settings_saved,
            open_data_folder=self.controller.open_data_folder, focus_tab=tab)

    def _settings_saved(self, changes: dict, restart: bool):
        if self.controller.settings_saved(changes, restart):
            self.ask_restart()

    def ask_restart(self):
        """Offer to restart now for settings that need it."""
        popup = ctk.CTkToplevel(self, fg_color=theme.BG)
        popup.title("Restart needed")
        popup.geometry("420x150")
        popup.transient(self)
        ctk.CTkLabel(popup, text="The new speech to text settings are used after a restart.\n"
                                 "The transcript so far is saved automatically.",
                     font=theme.font(12), text_color=theme.TEXT, justify="left").pack(padx=18, pady=(20, 12))
        row = ctk.CTkFrame(popup, fg_color="transparent")
        row.pack(fill="x", padx=18, pady=(0, 14))
        ctk.CTkButton(row, text="Restart now", width=120, height=34, fg_color=theme.ACCENT,
                      hover_color=theme.ACCENT_HOVER, font=theme.font(12, "bold"),
                      command=self.controller.restart_program).pack(side="right")
        ctk.CTkButton(row, text="Later", width=80, height=34, fg_color=theme.SURFACE_2,
                      hover_color=theme.SURFACE_3, font=theme.font(12),
                      command=popup.destroy).pack(side="right", padx=(0, 8))
        popup.after(120, popup.lift)

    def close_window(self):
        """Remember the window size and end the program."""
        if self._closing:
            return
        self._closing = True
        try:
            self.controller.settings_service.save_value("General", "window_geometry", self.geometry())
        except Exception as exception:  # pylint: disable=broad-except
            logger.error(f"Could not save the window size: {exception}")
        self.quit()

    # ------------------------------------------------------------------ commands (see bindings.py)
    def copy_to_clipboard(self):
        return self.controller.copy_to_clipboard()

    def save_file(self):
        return self.controller.save_file()

    def freeze_unfreeze(self):
        """The 'Suggest automatically' switch was clicked."""
        return self.controller.set_continuous_response(bool(self.auto_switch.get()))

    def freeze_unfreeze_from_key(self):
        """The keyboard shortcut for the same switch."""
        return self.controller.freeze_unfreeze()

    def enable_disable_speaker(self):
        return self.controller.enable_disable_speaker()

    def enable_disable_microphone(self):
        return self.controller.enable_disable_microphone()

    def get_response_now(self):
        return self.controller.get_response_now()

    def get_response_selected_now(self):
        return self.controller.get_response_selected_now()

    def summarize(self):
        return self.controller.summarize()

    def word_cloud(self):
        return self.controller.word_cloud()

    def update_response_ui_and_read_now(self):
        return self.controller.update_response_ui_and_read_now()

    def read_response_aloud(self):
        return self.controller.read_response_aloud()

    def set_transcript_state(self):
        return self.controller.set_transcript_state()

    def open_data_folder(self):
        return self.controller.open_data_folder()

    def open_github(self):
        return self.controller.open_github()

    def open_support(self):
        return self.controller.open_support()

    def capture_action(self, action_text: str):
        return self.controller.capture_action(action_text)

    def set_audio_language(self, lang: str):
        return self.controller.set_audio_language(lang)

    def set_response_language(self, lang: str):
        return self.controller.set_response_language(lang)
