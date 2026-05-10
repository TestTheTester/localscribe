"""
browser/window_detector.py — Active-window title parser for course/domain detection.
Uses win32gui (pywin32) to read the foreground window title.
"""

import re
import time
import threading
from typing import Callable, Optional

from config import DOMAIN_KEYWORDS, BROWSER_POLL_INTERVAL, Domain

try:
    import win32gui
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False


# ── Low-level OS calls ─────────────────────────────────────────────────────────

def get_active_window_title() -> str:
    """Return the title text of the current foreground window, or empty string."""
    if not WIN32_AVAILABLE:
        return ""
    try:
        hwnd = win32gui.GetForegroundWindow()
        return win32gui.GetWindowText(hwnd)
    except Exception:
        return ""


# ── Heuristic parsers ──────────────────────────────────────────────────────────

# Patterns that identify common training/browser title formats
_BROWSER_SUFFIXES = re.compile(
    r"\s[-–—|]\s*(Google Chrome|Mozilla Firefox|Microsoft Edge|Opera|Brave"
    r"|Chromium|Safari|Internet Explorer)$",
    re.IGNORECASE,
)
_PLATFORM_SUFFIXES = re.compile(
    r"\s[-–—|]\s*(Udemy|Pluralsight|Coursera|LinkedIn Learning|YouTube"
    r"|A Cloud Guru|CBT Nuggets|SANS|Cybrary|O'Reilly|edX|Skillshare).*$",
    re.IGNORECASE,
)


def infer_course_from_title(title: str) -> str:
    """
    Strip browser and platform suffixes from a window title and return what
    remains as a candidate course/video title.  Returns empty string on failure.
    """
    if not title:
        return ""
    clean = _BROWSER_SUFFIXES.sub("", title)
    clean = _PLATFORM_SUFFIXES.sub("", clean).strip()
    # Drop the raw domain name or tab group label if nothing useful remains
    if len(clean) < 4 or clean.lower() in ("new tab", "home", "start"):
        return ""
    return clean


def infer_domain_from_title(title: str) -> str:
    """
    Match title tokens against DOMAIN_KEYWORDS to guess the content domain.
    Returns the best-matching domain key or Domain.GENERAL.
    """
    lower = title.lower()
    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hit = sum(1 for kw in keywords if kw in lower)
        if hit:
            scores[domain] = hit

    if not scores:
        return Domain.GENERAL
    return max(scores, key=scores.__getitem__)


# ── Polling worker ─────────────────────────────────────────────────────────────

class WindowPollerWorker:
    """
    Daemon thread that polls the active window title every BROWSER_POLL_INTERVAL
    seconds.  Calls on_course_change(new_course, new_domain) when the detected
    course or domain changes.
    """

    def __init__(
        self,
        state,
        on_course_change: Optional[Callable[[str, str], None]] = None,
        poll_interval: float = BROWSER_POLL_INTERVAL,
    ) -> None:
        self.state            = state
        self.on_course_change = on_course_change
        self.poll_interval    = poll_interval
        self._thread: Optional[threading.Thread] = None

    def start(self) -> threading.Thread:
        self._thread = threading.Thread(
            target=self._run, name="WindowPoller", daemon=True
        )
        self._thread.start()
        return self._thread

    def _run(self) -> None:
        if not WIN32_AVAILABLE:
            print("[Browser] pywin32 not installed — browser detection disabled.")
            return

        print("[Browser] Window poller started.")
        last_course = ""
        last_domain = Domain.GENERAL

        while not self.state.stop_event.is_set():
            title  = get_active_window_title()
            course = infer_course_from_title(title)
            domain = infer_domain_from_title(title)

            if course and (course != last_course or domain != last_domain):
                last_course = course
                last_domain = domain
                print(f"[Browser] Detected: '{course}' ({domain})")
                if self.on_course_change:
                    try:
                        self.on_course_change(course, domain)
                    except Exception as e:
                        print(f"[Browser] Callback error: {e}")

            time.sleep(self.poll_interval)

        print("[Browser] Window poller exited.")
