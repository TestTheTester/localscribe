"""
app_state.py — Central shared state.  All worker threads read/write through here
via thread-safe accessors so no module needs to import another module's globals.
"""

import queue
import threading
import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import TrayState, CONFIDENCE_THRESHOLD


# ── Data records ───────────────────────────────────────────────────────────────

@dataclass
class TranscriptChunk:
    index: int
    text: str
    wall_clock: datetime.datetime
    confidence: float           # 0.0–1.0 derived from avg_logprob
    is_uncertain: bool          # confidence < CONFIDENCE_THRESHOLD
    ocr_text: Optional[str]        # Slide text captured near this chunk, if any
    ocr_image_path: Optional[Path] = None  # Saved screenshot for that slide

    def formatted(self) -> str:
        ts    = self.wall_clock.strftime("%H:%M:%S")
        flag  = " [uncertain]" if self.is_uncertain else ""
        slide = f"\n  [SLIDE] {self.ocr_text}" if self.ocr_text else ""
        return f"[{ts}]{flag} {self.text}{slide}"


@dataclass
class SlideCapture:
    wall_clock: datetime.datetime
    image_path: Path
    ocr_text:   Optional[str] = None


@dataclass
class SessionMeta:
    course_title: str
    domain: str
    started_at: datetime.datetime
    tags: list = field(default_factory=list)


# ── Main state container ───────────────────────────────────────────────────────

class AppState:
    """
    Single shared-state object.  Created once in main() and passed everywhere.
    All public methods are thread-safe.
    """

    def __init__(self) -> None:
        # ── Threading primitives ───────────────────────────────────────────────
        self.stop_event   = threading.Event()   # Set → all workers shut down
        self.pause_event  = threading.Event()   # Set → capture is paused

        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.ocr_queue:   queue.Queue[tuple[str, Optional[Path]]] = queue.Queue(maxsize=8)

        # ── Session data (protected by _chunks_lock) ───────────────────────────
        self._chunks_lock:  threading.Lock              = threading.Lock()
        self._chunks:       list[TranscriptChunk]       = []
        self.session_meta:  Optional[SessionMeta]       = None
        self.final_notes:   str                         = ""

        # ── Manual slide captures (protected by _slides_lock) ─────────────────
        self._slides_lock:    threading.Lock         = threading.Lock()
        self._slide_captures: list[SlideCapture]     = []

        # ── Tray state (protected by _state_lock) ─────────────────────────────
        self._state_lock: threading.Lock = threading.Lock()
        self._tray_state: TrayState      = TrayState.IDLE

    # ── Tray state ─────────────────────────────────────────────────────────────

    def set_tray_state(self, state: TrayState) -> None:
        with self._state_lock:
            self._tray_state = state

    def get_tray_state(self) -> TrayState:
        with self._state_lock:
            return self._tray_state

    # ── Pause / resume ─────────────────────────────────────────────────────────

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()

    def is_paused(self) -> bool:
        return self.pause_event.is_set()

    # ── Transcript chunks ──────────────────────────────────────────────────────

    def append_chunk(self, chunk: TranscriptChunk) -> None:
        with self._chunks_lock:
            self._chunks.append(chunk)

    def get_chunks(self) -> list[TranscriptChunk]:
        """Return a snapshot copy — safe to iterate without holding the lock."""
        with self._chunks_lock:
            return list(self._chunks)

    def clear_chunks(self) -> None:
        with self._chunks_lock:
            self._chunks.clear()
        with self._slides_lock:
            self._slide_captures.clear()
        self.audio_queue = queue.Queue()
        self.ocr_queue   = queue.Queue(maxsize=8)

    # ── Slide captures ─────────────────────────────────────────────────────────

    def add_slide_capture(self, capture: "SlideCapture") -> None:
        with self._slides_lock:
            self._slide_captures.append(capture)

    def get_slide_captures(self) -> "list[SlideCapture]":
        with self._slides_lock:
            return list(self._slide_captures)

    # ── Convenience properties ─────────────────────────────────────────────────

    def full_transcript(self) -> str:
        """Assembles chunks with timestamps and uncertainty flags."""
        return "\n\n".join(c.formatted() for c in self.get_chunks())

    def plain_transcript(self) -> str:
        """Plain text only — used as LLM input."""
        return "\n\n".join(c.text for c in self.get_chunks())

    def uncertain_chunks(self) -> list[TranscriptChunk]:
        return [c for c in self.get_chunks() if c.is_uncertain]

    def avg_confidence(self) -> float:
        chunks = self.get_chunks()
        if not chunks:
            return 0.0
        return sum(c.confidence for c in chunks) / len(chunks)

    def chunk_count(self) -> int:
        with self._chunks_lock:
            return len(self._chunks)

    def reset_for_new_session(self) -> None:
        """Prepare state for a fresh recording session."""
        self.stop_event.clear()
        self.pause_event.clear()
        self.clear_chunks()
        self.session_meta = None
        self.final_notes  = ""
        self.set_tray_state(TrayState.IDLE)
