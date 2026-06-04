"""
session.py — SessionController orchestrates the full lifecycle of one recording session.
All public methods are thread-safe and callable from tray, hotkey, or UI threads.
Long-running work dispatches to daemon threads; results return via root.after().
"""

import datetime
import itertools
import os
import subprocess
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from config import (
    OUTPUT_DIR, TrayState, OCR_ENABLED, PREFER_VISION_OCR,
    ANKI_MAX_CARDS, ANKI_DECK_NAME,
)
from app_state import AppState, SessionMeta, SlideCapture
from audio.capture import build_input_stream, find_loopback_device
from audio.transcriber import transcription_worker
from browser.window_detector import WindowPollerWorker
from export.obsidian_exporter import export_session as obsidian_export
from export.course_aggregator import CourseAggregator
from export.anki_exporter import write_anki_csv
from llm.note_generator import NoteGenerator
from llm.anki_generator import AnkiGenerator
from llm.qa_chat import QASession
from llm.ollama_client import OllamaClient

if TYPE_CHECKING:
    import tkinter as tk
    from ui.tray import TrayManager
    from ui.chat_window import ChatWindow


class SessionController:
    def __init__(
        self,
        state:          AppState,
        whisper_model,
        ollama_client:  OllamaClient,
        tray:           "TrayManager",
        root:           "tk.Tk",
        on_session_end: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.state          = state
        self.whisper_model  = whisper_model
        self.ollama_client  = ollama_client
        self.tray           = tray
        self.root           = root
        self.on_session_end = on_session_end   # Called on main thread after finalize

        self._note_gen   = NoteGenerator(ollama_client)
        self._anki_gen   = AnkiGenerator(ollama_client)
        self._aggregator = CourseAggregator(OUTPUT_DIR)

        # Live handles
        self._stream             = None          # sd.InputStream
        self._transcriber_thread: Optional[threading.Thread] = None
        self._ocr_thread:         Optional[threading.Thread] = None
        self._browser_worker:     Optional[WindowPollerWorker] = None
        self._chunk_counter       = itertools.count(1)

        self._slides_dir: Optional[Path]          = None
        self._qa_session:   Optional[QASession]   = None
        self._chat_window:  Optional["ChatWindow"] = None
        self._lock = threading.Lock()

    # ── Session start ──────────────────────────────────────────────────────────

    def start_session(self, course: str, domain: str, device_id: int, enable_ocr: bool = True) -> None:
        """Called from the main thread after session dialog is confirmed."""
        with self._lock:
            if self.state.get_tray_state() == TrayState.RECORDING:
                print("[Session] Already recording.")
                return

        self.state.reset_for_new_session()
        self._chunk_counter = itertools.count(1)

        self.state.session_meta = SessionMeta(
            course_title = course or "Training Session",
            domain       = domain,
            started_at   = datetime.datetime.now(),
        )

        # ── Audio stream ──────────────────────────────────────────────────────
        self._stream = build_input_stream(device_id, self.state)
        self._stream.start()

        # ── Transcription worker ───────────────────────────────────────────────
        self._transcriber_thread = threading.Thread(
            target = transcription_worker,
            args   = (self.whisper_model, self.state, self._chunk_counter),
            name   = "Transcriber",
            daemon = True,
        )
        self._transcriber_thread.start()

        # ── Slides folder (used by both manual F10 capture and auto OCR) ────────
        date_str    = self.state.session_meta.started_at.strftime("%Y-%m-%d")
        safe_course = "".join(
            c if c.isalnum() or c in " _-" else "_"
            for c in (course or "session")
        ).strip()[:60]
        self._slides_dir = OUTPUT_DIR / date_str / safe_course / "slides"
        self._slides_dir.mkdir(parents=True, exist_ok=True)

        # ── Slide OCR worker (auto continuous capture — off by default) ────────
        if enable_ocr and OCR_ENABLED:
            self._start_ocr_worker(slides_dir=self._slides_dir)

        # ── Browser title poller ───────────────────────────────────────────────
        self._browser_worker = WindowPollerWorker(
            state             = self.state,
            on_course_change  = self._on_browser_course_change,
        )
        self._browser_worker.start()

        self.state.set_tray_state(TrayState.RECORDING)
        self.tray.update_state(TrayState.RECORDING,
                               f"Recording: {self.state.session_meta.course_title}")
        print(f"[Session] Started: {course} ({domain})")

    def _start_ocr_worker(self, slides_dir=None) -> None:
        try:
            from ocr.screen_capture import ScreenCaptureWorker
            from ocr.ocr_engine import extract_slide_text, is_tesseract_available

            tesseract_ok = is_tesseract_available()
            vision_ok    = self.ollama_client.resolve_vision_model()

            if not tesseract_ok and not vision_ok:
                print(
                    "[OCR] Slide OCR disabled — neither Tesseract nor a vision model is available.\n"
                    "      To enable: install Tesseract (https://github.com/UB-Mannheim/tesseract/wiki)\n"
                    "      OR pull a vision model:  ollama pull llava"
                )
                return

            method = "vision" if (vision_ok and PREFER_VISION_OCR) else \
                     "tesseract" if tesseract_ok else "vision"
            print(f"[OCR] Starting slide capture (method: {method})")

            def _ocr_func(img):
                return extract_slide_text(img, self.ollama_client, PREFER_VISION_OCR)

            worker = ScreenCaptureWorker(self.state, _ocr_func, slides_dir=slides_dir)
            self._ocr_thread = worker.start()
        except Exception as e:
            print(f"[Session] OCR worker failed to start: {e}")

    def _on_browser_course_change(self, course: str, domain: str) -> None:
        """Called from browser poller thread — updates session meta safely."""
        if self.state.session_meta and not self.state.session_meta.course_title:
            self.state.session_meta.course_title = course
        print(f"[Session] Browser: course='{course}' domain='{domain}'")

    # ── Manual slide capture (F10) ─────────────────────────────────────────────

    def capture_slide(self) -> None:
        """Called from the F10 hotkey — fire-and-forget on a daemon thread."""
        if self.state.get_tray_state() not in (TrayState.RECORDING, TrayState.PAUSED):
            return
        threading.Thread(target=self._do_capture_slide, name="SlideCapture", daemon=True).start()

    def _do_capture_slide(self) -> None:
        from ocr.screen_capture import capture_primary_screen
        from ocr.ocr_engine import extract_slide_text

        img = capture_primary_screen()
        if img is None:
            print("[Slide] Screenshot failed.")
            return

        ts         = datetime.datetime.now()
        image_path = self._slides_dir / f"slide_{ts.strftime('%H%M%S')}.png"
        try:
            img.save(str(image_path))
        except Exception as e:
            print(f"[Slide] Could not save image: {e}")
            return

        ocr_text = None
        try:
            ocr_text = extract_slide_text(img, self.ollama_client, PREFER_VISION_OCR)
        except Exception:
            pass

        self.state.add_slide_capture(SlideCapture(
            wall_clock = ts,
            image_path = image_path,
            ocr_text   = ocr_text,
        ))
        label = f"{image_path.name}" + (f" — {ocr_text[:60]}" if ocr_text else "")
        print(f"[Slide] Captured: {label}")
        self.tray.notify("Slide captured", image_path.name)

    # ── Pause / resume ─────────────────────────────────────────────────────────

    def toggle_pause(self) -> None:
        """Toggle pause state. Safe to call from any thread (F9 hotkey, tray)."""
        current = self.state.get_tray_state()
        if current not in (TrayState.RECORDING, TrayState.PAUSED):
            return

        paused = not self.state.is_paused()
        self.state.set_paused(paused)

        if paused:
            self.state.set_tray_state(TrayState.PAUSED)
            self.tray.update_state(TrayState.PAUSED, "Training Noter — Paused (F9 to resume)")
            print("[Session] Paused.")
        else:
            self.state.set_tray_state(TrayState.RECORDING)
            self.tray.update_state(TrayState.RECORDING,
                                   f"Recording: {self.state.session_meta.course_title if self.state.session_meta else ''}")
            print("[Session] Resumed.")

    # ── Session stop ───────────────────────────────────────────────────────────

    def stop_session(self) -> None:
        """Stop capture and kick off async note generation."""
        current = self.state.get_tray_state()
        if current not in (TrayState.RECORDING, TrayState.PAUSED):
            print("[Session] Nothing to stop.")
            return

        print("[Session] Stopping capture...")
        self.state.set_paused(False)        # Unpause so buffers drain
        self.state.stop_event.set()

        # Stop audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        self.state.set_tray_state(TrayState.PROCESSING)
        self.tray.update_state(TrayState.PROCESSING, "Training Noter — Generating notes…")
        self.tray.notify("Training Noter", "Recording stopped. Generating notes…")

        threading.Thread(
            target=self._finalize, name="Finalizer", daemon=True
        ).start()

    # ── Finalise (daemon thread) ───────────────────────────────────────────────

    def _finalize(self) -> None:
        """Run in a daemon thread: join workers, generate notes, export."""
        # Wait for transcription to finish draining
        if self._transcriber_thread:
            self._transcriber_thread.join(timeout=120)

        chunks = self.state.get_chunks()
        if not chunks:
            print("[Session] No transcript chunks — nothing to export.")
            self.root.after(0, lambda: self._finish({}))
            return

        meta = self.state.session_meta
        print(f"[Session] Finalising: {len(chunks)} chunks, avg_conf={self.state.avg_confidence():.2f}")

        # ── Note generation ────────────────────────────────────────────────────
        notes = ""
        try:
            notes = self._note_gen.generate(
                chunks          = chunks,
                slide_captures  = self.state.get_slide_captures(),
                domain          = meta.domain if meta else "general",
                stream_callback = lambda t: print(t, end="", flush=True),
            )
            self.state.final_notes = notes
        except Exception as e:
            print(f"[Session] Note generation failed: {e}")
            notes = "Note generation failed. See transcript.\n\n" + self.state.plain_transcript()

        # ── Obsidian export ────────────────────────────────────────────────────
        export_paths: dict[str, Path] = {}
        try:
            export_paths = obsidian_export(
                session_meta   = meta,
                notes          = notes,
                chunks         = chunks,
                slide_captures = self.state.get_slide_captures(),
                base_dir       = OUTPUT_DIR,
            )
        except Exception as e:
            print(f"[Session] Obsidian export failed: {e}")

        # ── Course index ───────────────────────────────────────────────────────
        if export_paths.get("notes") and meta:
            try:
                self._aggregator.register_session(
                    session_meta     = meta,
                    notes_path       = export_paths["notes"],
                    transcript_path  = export_paths.get("transcript", export_paths["notes"]),
                    avg_confidence   = self.state.avg_confidence(),
                    chunk_count      = len(chunks),
                )
            except Exception as e:
                print(f"[Session] Aggregator failed: {e}")

        # ── Anki flashcards ────────────────────────────────────────────────────
        if notes:
            try:
                pairs = self._anki_gen.generate_qa_pairs(
                    notes_text = notes,
                    domain     = meta.domain if meta else "general",
                    max_cards  = ANKI_MAX_CARDS,
                )
                if pairs:
                    write_anki_csv(
                        qa_pairs   = pairs,
                        output_dir = OUTPUT_DIR,
                        deck_name  = ANKI_DECK_NAME,
                        course     = meta.course_title if meta else "",
                    )
                    export_paths["anki"] = OUTPUT_DIR   # approximate
            except Exception as e:
                print(f"[Session] Anki export failed: {e}")

        # ── Build Q&A session for chat ─────────────────────────────────────────
        self._qa_session = QASession(
            client     = self.ollama_client,
            transcript = self.state.full_transcript(),
            notes      = notes,
        )

        self.root.after(0, lambda: self._finish(export_paths))

    def _finish(self, export_paths: dict) -> None:
        """Main-thread callback: update tray, notify, call on_session_end."""
        self.state.set_tray_state(TrayState.IDLE)
        self.tray.update_state(TrayState.IDLE, "Training Noter — Done ✓")
        self.tray.notify("Training Noter", "Notes ready! Right-click for Q&A chat.")

        print(f"\n[Session] Done. Exports: {export_paths}")

        if self.on_session_end:
            self.on_session_end(export_paths)

    # ── Post-session Q&A ───────────────────────────────────────────────────────

    def open_chat(self) -> None:
        """Open the Q&A chat window. Must be called from the main thread."""
        if self._qa_session is None:
            print("[Session] No session available for chat yet.")
            return
        if self._chat_window is None:
            from ui.chat_window import ChatWindow
            import tkinter as tk
            self._chat_window = ChatWindow(
                root       = self.root,
                qa_session = self._qa_session,
                title      = f"Q&A — {self.state.session_meta.course_title if self.state.session_meta else 'Session'}",
            )
        self._chat_window.show()

    # ── Helpers ────────────────────────────────────────────────────────────────

    def open_output_folder(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(OUTPUT_DIR)])

    @property
    def is_active(self) -> bool:
        return self.state.get_tray_state() in (TrayState.RECORDING, TrayState.PAUSED)
