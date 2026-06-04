"""
main.py — Training Video Note-Taker (multi-file edition)
Entry point: boots all subsystems then hands control to tkinter's mainloop.

Threading model
───────────────
  Main thread    : tkinter event loop (root.mainloop())
  Tray thread    : pystray icon  (daemon)
  Transcriber    : faster-whisper worker (daemon, per session)
  OCR worker     : screen capture + OCR   (daemon, per session)
  Browser poller : window title polling   (daemon, per session)
  Finalizer      : note generation + export (daemon, triggered on stop)
"""

import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

from config import OUTPUT_DIR, TrayState
from app_state import AppState
from audio.transcriber import load_whisper_model
from llm.ollama_client import OllamaClient
from session import SessionController
from ui.tray import TrayManager
from ui.session_dialog import show_session_dialog
from hotkeys import register_hotkeys, unregister_hotkeys
from browser.window_detector import infer_course_from_title, get_active_window_title


# ── Bootstrap ──────────────────────────────────────────────────────────────────

def build_app() -> tuple[tk.Tk, AppState, SessionController, TrayManager]:
    """
    Create every subsystem in dependency order.
    Returns (root, state, controller, tray) — the four objects main() needs.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Hidden tkinter root ────────────────────────────────────────────────────
    root = tk.Tk()
    root.withdraw()                 # Stay invisible; app lives in the tray
    root.title("Training Noter")

    # ── Shared state ───────────────────────────────────────────────────────────
    state = AppState()

    # ── Ollama health check ────────────────────────────────────────────────────
    print("Checking Ollama…", end=" ", flush=True)
    client = OllamaClient()
    if not client.health_check():
        messagebox.showerror(
            "Ollama unreachable",
            "Cannot connect to Ollama.\n\nPlease start it with:\n  ollama serve\n\n"
            "Then re-launch Training Noter."
        )
        sys.exit(1)
    print("OK")

    # ── Whisper model ──────────────────────────────────────────────────────────
    print("Loading Whisper model…")
    whisper = load_whisper_model()
    print("Whisper ready.")

    # ── Session controller ─────────────────────────────────────────────────────
    # We need a reference to tray before creating the controller,
    # but tray needs controller callbacks.  Wire via lambdas.
    tray_holder: list[TrayManager] = []
    ctrl_holder: list[SessionController] = []

    def _on_start() -> None:
        # Try to pre-fill course from the active browser tab
        suggested = infer_course_from_title(get_active_window_title())
        params = show_session_dialog(root, suggested_course=suggested)
        if params is None:
            return                  # User cancelled
        ctrl_holder[0].start_session(
            course     = params["course"],
            domain     = params["domain"],
            device_id  = params["device_id"],
            enable_ocr = params.get("enable_ocr", True),
        )

    def _on_pause_resume() -> None:
        if ctrl_holder:
            ctrl_holder[0].toggle_pause()

    def _on_stop() -> None:
        if ctrl_holder:
            ctrl_holder[0].stop_session()

    def _on_open_chat() -> None:
        if ctrl_holder:
            ctrl_holder[0].open_chat()

    def _on_open_folder() -> None:
        if ctrl_holder:
            ctrl_holder[0].open_output_folder()

    def _on_quit() -> None:
        _shutdown(root, ctrl_holder[0] if ctrl_holder else None)

    tray = TrayManager(
        root            = root,
        on_start        = _on_start,
        on_pause_resume = _on_pause_resume,
        on_stop         = _on_stop,
        on_open_chat    = _on_open_chat,
        on_open_folder  = _on_open_folder,
        on_quit         = _on_quit,
    )
    tray_holder.append(tray)

    def _on_session_end(export_paths: dict) -> None:
        """Called on main thread after finalize completes."""
        notes_path = export_paths.get("notes")
        anki_path  = export_paths.get("anki")
        msg = "Notes generated successfully!\n\n"
        if notes_path:
            msg += f"Notes:  {notes_path}\n"
        if anki_path:
            msg += f"Anki:   {anki_path}\n"
        msg += "\nRight-click the tray icon to open Q&A chat."
        messagebox.showinfo("Training Notes Ready", msg)

    ctrl = SessionController(
        state          = state,
        whisper_model  = whisper,
        ollama_client  = client,
        tray           = tray,
        root           = root,
        on_session_end = _on_session_end,
    )
    ctrl_holder.append(ctrl)

    return root, state, ctrl, tray


# ── Shutdown ───────────────────────────────────────────────────────────────────

def _shutdown(root: tk.Tk, ctrl: "SessionController | None") -> None:
    """Clean up before exit."""
    print("\n[Main] Shutting down…")
    unregister_hotkeys()
    if ctrl and ctrl.is_active:
        ctrl.stop_session()
    try:
        root.quit()
    except Exception:
        pass


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 56)
    print("  Training Video Note-Taker")
    print("  Local AI  ·  Ollama + faster-whisper")
    print("  RTX 4050  ·  Windows 11")
    print("=" * 56)
    print()

    root, state, ctrl, tray = build_app()

    # ── Start tray icon (daemon thread) ────────────────────────────────────────
    tray.start()
    print("[Main] System tray icon running. Right-click to start a session.")

    # ── Register global hotkeys ────────────────────────────────────────────────
    register_hotkeys(
        on_pause_resume  = ctrl.toggle_pause,
        on_capture_slide = ctrl.capture_slide,
    )

    # ── Handle window close (root is hidden but still has a WM) ───────────────
    root.protocol("WM_DELETE_WINDOW", lambda: _shutdown(root, ctrl))

    # ── Tkinter event loop ─────────────────────────────────────────────────────
    try:
        root.mainloop()
    except KeyboardInterrupt:
        _shutdown(root, ctrl)


if __name__ == "__main__":
    main()
