"""
hotkeys.py — Global keyboard hotkeys via the `keyboard` library.
F9  →  toggle pause/resume for the active session.
F10 →  capture a slide screenshot (manual trigger).
"""

from typing import Callable, Optional

from config import SLIDE_CAPTURE_KEY

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

_registered: bool = False


def register_hotkeys(
    on_pause_resume:  Callable,
    on_capture_slide: Optional[Callable] = None,
) -> None:
    """
    Bind F9 (pause/resume) and F10 (capture slide) globally.
    Safe to call multiple times — re-registers once.
    Falls back gracefully if `keyboard` is not installed.
    """
    global _registered
    if not KEYBOARD_AVAILABLE:
        print("[Hotkeys] `keyboard` package not installed — hotkeys disabled.")
        return
    if _registered:
        return
    try:
        keyboard.add_hotkey("F9", on_pause_resume, suppress=False)
        print("[Hotkeys] F9 → pause/resume registered.")
        if on_capture_slide:
            keyboard.add_hotkey(SLIDE_CAPTURE_KEY, on_capture_slide, suppress=False)
            print(f"[Hotkeys] {SLIDE_CAPTURE_KEY} → capture slide registered.")
        _registered = True
    except Exception as e:
        print(f"[Hotkeys] Could not register hotkeys: {e}")
        print("         Try running as Administrator if hotkeys don't respond.")


def unregister_hotkeys() -> None:
    """Remove all registered hotkeys on shutdown."""
    global _registered
    if not KEYBOARD_AVAILABLE or not _registered:
        return
    try:
        keyboard.remove_all_hotkeys()
        _registered = False
    except Exception:
        pass
