"""
ocr/ocr_engine.py — OCR text extraction from PIL Images.
Primary: pytesseract.  Fallback: Ollama vision model.

Availability is checked ONCE at startup — errors are never repeated per-frame.
"""

import shutil
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image
    from llm.ollama_client import OllamaClient


# ── One-time availability checks ───────────────────────────────────────────────

def _check_tesseract() -> bool:
    """
    True only when both the pytesseract wrapper AND the tesseract binary exist.
    The Python package can be installed without the binary, which causes the
    'tesseract is not installed or it's not in your PATH' runtime error.
    """
    try:
        import pytesseract
        # shutil.which checks PATH; pytesseract.get_tesseract_version() also works
        # but spawns a subprocess — which() is cheaper for a startup check.
        if shutil.which("tesseract") is None:
            # Also check the common Windows default install location
            import os
            default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.isfile(default):
                pytesseract.pytesseract.tesseract_cmd = default
                return True
            return False
        return True
    except ImportError:
        return False


TESSERACT_AVAILABLE: bool = _check_tesseract()

if TESSERACT_AVAILABLE:
    print("[OCR] Tesseract: available")
else:
    print("[OCR] Tesseract: not found — install from https://github.com/UB-Mannheim/tesseract/wiki")


# ── Extraction functions ────────────────────────────────────────────────────────

def is_tesseract_available() -> bool:
    return TESSERACT_AVAILABLE


def extract_text_tesseract(image: "Image.Image") -> str:
    """
    Run pytesseract on image with slide-optimised preprocessing.
    Returns empty string silently if tesseract is unavailable.
    """
    if not TESSERACT_AVAILABLE:
        return ""
    try:
        import pytesseract
        from PIL import ImageEnhance
        w, h   = image.size
        scaled = image.resize((w * 2, h * 2), resample=1)    # LANCZOS
        grey   = scaled.convert("L")
        sharp  = ImageEnhance.Sharpness(grey).enhance(2.0)
        return pytesseract.image_to_string(sharp, config="--psm 6").strip()
    except Exception:
        return ""


def extract_text_ollama_vision(
    image: "Image.Image",
    client: "OllamaClient",
    prompt: str = (
        "Look at this screenshot. Extract ALL visible text from the slide or screen. "
        "Return only the raw text — no commentary, no formatting."
    ),
) -> str:
    """
    Use the Ollama vision model to extract text.
    Returns empty string silently if no vision model is configured.
    """
    if not client or not client.vision_model:
        return ""
    try:
        b64 = client.pil_to_b64(image)
        return client.vision_chat(prompt, b64).strip()
    except Exception:
        return ""


def extract_slide_text(
    image:         "Image.Image",
    client:        Optional["OllamaClient"] = None,
    prefer_vision: bool                     = False,
) -> str:
    """
    Main entry point for ScreenCaptureWorker.

    prefer_vision=False (default): tesseract → vision fallback
    prefer_vision=True:            vision → tesseract fallback
    Returns empty string if both fail (no error printed here).
    """
    def _tess() -> str:
        return extract_text_tesseract(image)

    def _vis() -> str:
        return extract_text_ollama_vision(image, client) if client else ""

    primary, fallback = (_vis, _tess) if prefer_vision else (_tess, _vis)
    return primary() or fallback()
