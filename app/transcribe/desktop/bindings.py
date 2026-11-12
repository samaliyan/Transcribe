"""Attach commands, menus and keyboard shortcuts to the widgets built by view.py."""

from __future__ import annotations


class DesktopCommandBinder:
    """Bind widget commands, the menus and the keyboard shortcuts."""

    def bind(self, ui, config: dict):
        """Bind everything for the main window."""
        del config
        self.bind_buttons(ui)
        self.bind_menus(ui)
        self.bind_context_menu(ui)
        self.bind_shortcuts(ui)

    def bind_buttons(self, ui):
        """Bind button and selection widget callbacks."""
        ui.mic_button.configure(command=ui.enable_disable_microphone)
        ui.speaker_button.configure(command=ui.enable_disable_speaker)
        ui.pause_button.configure(command=ui.set_transcript_state)
        ui.settings_button.configure(command=ui.open_settings)
        ui.more_button.configure(command=ui.show_more_menu)
        ui.copy_transcript_button.configure(command=ui.copy_to_clipboard)
        ui.save_button.configure(command=ui.save_file)
        ui.clear_button.configure(command=ui.clear_transcript)
        ui.copy_response_button.configure(command=ui.copy_response)
        ui.read_button.configure(command=ui.read_response_aloud)
        ui.live_button.configure(command=ui.back_to_live)
        ui.ai_banner_button.configure(command=ui.open_settings)
        ui.response_now_button.configure(command=ui.get_response_now)
        ui.auto_switch.configure(command=ui.freeze_unfreeze)
        ui.summarize_button.configure(command=ui.summarize)
        ui.font_minus_button.configure(command=lambda: ui.change_text_size(-1))
        ui.font_plus_button.configure(command=lambda: ui.change_text_size(+1))
        ui.audio_lang_combobox.configure(command=ui.set_audio_language)
        ui.response_lang_combobox.configure(command=ui.set_response_language)
        ui.transcript_text.on_row_selected = ui.show_reply_for_row
        ui.protocol("WM_DELETE_WINDOW", ui.close_window)

    def bind_menus(self, ui):
        """Fill the 'More' menu."""
        menu = ui.more_menu
        menu.add_checkbutton(label="Keep the window on top", variable=ui.topmost_var,
                             command=ui.toggle_topmost)
        menu.add_separator()
        menu.add_command(label="Reply to the selected text", command=ui.get_response_selected_now)
        menu.add_command(label="Suggest a reply and read it aloud", command=ui.update_response_ui_and_read_now)
        menu.add_command(label="Word cloud of the conversation", command=ui.word_cloud)
        menu.add_separator()
        menu.add_command(label="Open the folder with transcripts and logs", command=ui.open_data_folder)
        menu.add_command(label="Project page (help, new versions)", command=ui.open_github)
        menu.add_command(label="Report a problem", command=ui.open_support)

    def bind_context_menu(self, ui):
        """Populate transcript context-menu actions."""
        transcript = ui.transcript_text
        transcript.add_right_click_menu(label="Reply to the selected text", command=ui.get_response_selected_now)
        transcript.add_right_click_menu(label="Edit this line…", command=ui.edit_current_line)
        transcript.add_right_menu_separator()
        transcript.add_right_click_menu(label="Copy the whole transcript", command=ui.copy_to_clipboard)
        transcript.add_right_click_menu(label="Save the transcript…", command=ui.save_file)
        transcript.add_right_click_menu(label="Clear the transcript", command=ui.clear_transcript)

    def bind_shortcuts(self, ui):
        """Keyboard shortcuts (they work while the window has the focus)."""
        shortcuts = {
            "<F2>": ui.get_response_now,
            "<Control-Return>": ui.get_response_now,
            "<F3>": ui.freeze_unfreeze_from_key,
            "<F4>": ui.set_transcript_state,
            "<Control-s>": ui.save_file,
            "<Control-comma>": ui.open_settings,
            "<Control-plus>": lambda: ui.change_text_size(+1),
            "<Control-equal>": lambda: ui.change_text_size(+1),
            "<Control-minus>": lambda: ui.change_text_size(-1),
        }
        for sequence, command in shortcuts.items():
            ui.bind(sequence, lambda _event, command=command: command())
