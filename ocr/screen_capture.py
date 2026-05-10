"""
ocr/screen_capture.py — Periodic screenshot capture with change detection.
Sends changed frames to the OCR engine and pushes text into state.ocr_queue.
"""

import hashlib
import queue
import threading
import time
from typing import Optional

try:
    import mss
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from config import OCR_INTERVAL_SECONDS, OCR_CHANGE_THRESHOLD, PREFER_VISION_OCR


def capture_primary_screen() -> Optional["Image.Image"]:
    """Grab the primary monitor and return a PIL Image (RGB), or None on failure."""
    if not MSS_AVAILABLE or not PIL_AVAILABLE:
        return None
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]   # [0] = all monitors combined; [1] = primary
            raw     = sct.grab(monitor)
            return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    except Exception as e:
        print(f"[OCR] Screenshot failed: {e}")
        return None


def image_hash(image: "Image.Image", size: int = 8) -> str:
    """
    Compute a perceptual (difference) hash for quick change detection.
    Returns a hex string.
    """
    try:
        thumb  = image.convert("L").resize((size + 1, size), Image.LANCZOS)
        pixels = list(thumb.getdata())
        bits   = "".join(
            "1" if pixels[i] > pixels[i + 1] else "0"
            for i in range(size * size)
        )
        # Fold bits into bytes for a compact hash
        n   = int(bits, 2)
        raw = n.to_bytes((len(bits) + 7) // 8, "big")
        return raw.hex()
    except Exception:
        # Fallback: MD5 of raw pixel bytes
        buf = image.tobytes()
        return hashlib.md5(buf[:4096]).hexdigest()


def hamming_distance(h1: str, h2: str) -> float:
    """
    Fraction of differing bits between two hex hash strings.
    Returns a value in [0.0, 1.0].
    """
    if not h1 or not h2 or len(h1) != len(h2):
        return 1.0
    b1 = bin(int(h1, 16))[2:].zfill(len(h1) * 4)
    b2 = bin(int(h2, 16))[2:].zfill(len(h2) * 4)
    diff = sum(a != b for a, b in zip(b1, b2))
    return diff / max(len(b1), 1)


class ScreenCaptureWorker:
    """
    Daemon thread that periodically screenshots the primary monitor.
    When a significant visual change is detected (slide change), it calls
    the OCR engine and pushes the extracted text into state.ocr_queue.
    """

    def __init__(
        self,
        state,
        ocr_func,           # Callable[[PIL.Image], str]
        interval:  float    = OCR_INTERVAL_SECONDS,
        threshold: float    = OCR_CHANGE_THRESHOLD,
    ) -> None:
        self.state     = state
        self.ocr_func  = ocr_func
        self.interval  = interval
        self.threshold = threshold
        self._thread: Optional[threading.Thread] = None

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(
            target=self._run, name="ScreenCapture", daemon=True
        )
        self._thread.start()
        return self._thread

    def _run(self) -> None:
        if not MSS_AVAILABLE:
            print("[OCR] mss not installed — slide OCR disabled.")
            return

        print("[OCR] Screen capture worker started.")
        last_hash:  Optional[str] = None

        while not self.state.stop_event.is_set():
            if self.state.is_paused():
                time.sleep(self.interval)
                continue

            img = capture_primary_screen()
            if img is None:
                time.sleep(self.interval)
                continue

            current_hash = image_hash(img)
            changed = (
                last_hash is None
                or hamming_distance(last_hash, current_hash) > self.threshold
            )

            if changed:
                last_hash = current_hash
                try:
                    text = self.ocr_func(img)
                    if text and text.strip():
                        # Non-blocking put — drop if queue is full
                        try:
                            self.state.ocr_queue.put_nowait(text.strip())
                        except queue.Full:
                            pass
                except Exception as e:
                    print(f"[OCR] Engine error: {e}")

            time.sleep(self.interval)

        print("[OCR] Screen capture worker exited.")
