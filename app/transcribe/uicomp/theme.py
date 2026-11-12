"""Colours, fonts and small text helpers shared by the desktop window."""

from __future__ import annotations

import sys
import unicodedata

BG = "#0f1115"            # window
SURFACE = "#171a21"       # panels
SURFACE_2 = "#20242e"     # inputs, quiet buttons
SURFACE_3 = "#2a2f3b"     # hover
BORDER = "#2a2f3b"
TEXT = "#e7e9ee"
MUTED = "#8d95a5"
ACCENT = "#4f8cff"
ACCENT_HOVER = "#3b76e6"
YOU = "#4fd1b5"           # the microphone's rows
SPEAKER = "#f5b942"       # the speakers' rows
OK = "#3ecf8e"
WARN = "#f5b942"
DANGER = "#f2686b"
SELECT = "#2b4a7d"

FONT = "Segoe UI" if sys.platform == "win32" else "DejaVu Sans"
MIN_TEXT_SIZE, DEFAULT_TEXT_SIZE, MAX_TEXT_SIZE = 9, 14, 30


def font(size: int = 12, weight: str = "normal"):
    return (FONT, size, weight)


def is_rtl(text: str) -> bool:
    """True when the first letter of the text belongs to a right-to-left script (Persian, Arabic, Hebrew)."""
    for char in text:
        direction = unicodedata.bidirectional(char)
        if direction in ("R", "AL"):
            return True
        if direction == "L":
            return False
    return False


def clamp_text_size(size) -> int:
    try:
        size = int(size)
    except (TypeError, ValueError):
        size = DEFAULT_TEXT_SIZE
    return max(MIN_TEXT_SIZE, min(MAX_TEXT_SIZE, size))
