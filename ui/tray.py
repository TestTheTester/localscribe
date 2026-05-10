"""
ui/tray.py — System-tray icon and menu via pystray.
Runs in a daemon thread; communicates back to the main thread via root.after().
"""

import threading
from typing import TYPE_CHECKING, Callable, Optional

try:
    import pystray
    PYSTRAY_AVAILABLE = True
except ImportError:
    PYSTRAY_AVAILABLE = False

from config import TrayState
from ui.icon_factory import make_all_icons, make_icon_image

if TYPE_CHECKING:
    import tkinter as tk


class TrayManager:
    """
    Manages a pystray.Icon running in a daemon thread.
    Menu callbacks use root.after() to safely hand off work to the main thread.
    """

    def __init__(
        self,
        root:              "tk.Tk",
        on_start:          Callable,
        on_pause_resume:   Callable,
        on_stop:           Callable,
        on_open_chat:      Callable,
        on_open_folder:    Callable,
        on_quit:           Callable,
    ) -> None:
        self.root            = root
        self._on_start        = on_start
        self._on_pause_resume = on_pause_resume
        self._on_stop         = on_stop
        self._on_open_chat    = on_open_chat
        self._on_open_folder  = on_open_folder
        self._on_quit         = on_quit

        self._icon: Optional["pystray.Icon"] = None
        self._icons: dict = {}
        self._thread: Optional[threading.Thread] = None

    # ── Startup ────────────────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        if not PYSTRAY_AVAILABLE:
            print("[Tray] pystray not installed — system tray disabled.")
            return threading.Thread()   # dummy

        self._icons = make_all_icons(64)
        self._icon  = pystray.Icon(
            name    = "training_noter",
            icon    = self._icons[TrayState.IDLE],
            title   = "Training Noter — Idle",
            menu    = self._build_menu(),
        )
        self._thread = threading.Thread(
            target=self._icon.run, name="Tray", daemon=True
        )
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        if self._icon:
            try:
                self._icon.stop()
            except Exception:
                pass

    # ── Icon state updates (callable from any thread) ──────────────────────────

    def update_state(self, state: str, tooltip: Optional[str] = None) -> None:
        if not self._icon:
            return
        try:
            self._icon.icon  = self._icons.get(state, self._icons[TrayState.IDLE])
            self._icon.title = tooltip or f"Training Noter — {state.capitalize()}"
        except Exception:
            pass

    def notify(self, title: str, message: str) -> None:
        """Show a balloon/toast notification if the platform supports it."""
        if not self._icon:
            return
        try:
            self._icon.notify(message, title)
        except Exception:
            pass

    # ── Menu construction ──────────────────────────────────────────────────────

    def _build_menu(self) -> "pystray.Menu":
        def _defer(cb: Callable) -> Callable:
            """Wrap a callback so it runs on the main thread via root.after()."""
            def _handler(icon, item):
                self.root.after(0, cb)
            return _handler

        return pystray.Menu(
            pystray.MenuItem("▶ New Session",               _defer(self._on_start)),
            pystray.MenuItem("⏸ Pause / Resume  (F9)",      _defer(self._on_pause_resume)),
            pystray.MenuItem("⏹ Stop & Generate Notes",     _defer(self._on_stop)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("💬 Post-Session Q&A",         _defer(self._on_open_chat)),
            pystray.MenuItem("📂 Open Notes Folder",        _defer(self._on_open_folder)),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("✕ Quit",                      _defer(self._on_quit)),
        )
