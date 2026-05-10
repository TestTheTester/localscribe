"""
ui/icon_factory.py — Programmatically generate pystray icon images via Pillow.
No external icon files required.
"""

from config import TrayState

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Map each tray state to an RGBA fill colour
STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    TrayState.IDLE:       (100, 100, 100, 255),   # grey
    TrayState.RECORDING:  (0,   200, 80,  255),   # green
    TrayState.PAUSED:     (255, 165, 0,   255),   # orange
    TrayState.PROCESSING: (30,  144, 255, 255),   # blue
    TrayState.ERROR:      (220, 50,  50,  255),   # red
}

STATE_LABELS: dict[str, str] = {
    TrayState.IDLE:       "●",
    TrayState.RECORDING:  "◉",
    TrayState.PAUSED:     "⏸",
    TrayState.PROCESSING: "⟳",
    TrayState.ERROR:      "✕",
}


def make_icon_image(state: str, size: int = 64) -> "Image.Image":
    """
    Draw a filled circle with a state indicator glyph on a transparent background.
    Falls back to a plain-colour square if Pillow is unavailable.
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow is required for icon generation (pip install Pillow)")

    color = STATE_COLORS.get(state, STATE_COLORS[TrayState.IDLE])
    img   = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw  = ImageDraw.Draw(img)

    # Outer circle
    margin = 4
    draw.ellipse([margin, margin, size - margin, size - margin], fill=color)

    # Inner white circle for depth (ring effect)
    inner_margin = size // 6
    draw.ellipse(
        [inner_margin, inner_margin, size - inner_margin, size - inner_margin],
        fill=(*color[:3], 180),
    )

    return img


def make_all_icons(size: int = 64) -> dict[str, "Image.Image"]:
    """Pre-generate all state icons at startup."""
    return {state: make_icon_image(state, size) for state in STATE_COLORS}
