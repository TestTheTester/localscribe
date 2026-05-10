"""
hotkeys.py — Global keyboard hotkeys via the `keyboard` library.
F9  →  toggle pause/resume for the active session.
"""

from typing import Callable, Optional

try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False

_registered: bool = False


def register_hotkeys(on_pause_resume: Callable) -> None:
    """
    Bind F9 globally.  Safe to call multiple times — re-registers once.
    Falls back gracefully if `keyboard` is not installed.
    """
    global _registered
    if not KEYBOARD_AVAILABLE:
        print("[Hotkeys] `keyboard` package not installed — F9 hotkey disabled.")
        return
    if _registered:
        return
    try:
        keyboard.add_hotkey("F9", on_pause_resume, suppress=False)
        print("[Hotkeys] F9 → pause/resume registered.")
        _registered = True
    except Exception as e:
        print(f"[Hotkeys] Could not register F9: {e}")
        print("         Try running as Administrator if F9 doesn't respond.")


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
