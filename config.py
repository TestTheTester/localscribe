"""
config.py — Single source of truth for all settings.
Edit this file to tune the app without touching logic.
"""

import enum
from pathlib import Path

# ── Audio ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000          # Hz — Whisper expects 16 kHz
CHANNELS           = 1
CHUNK_SECONDS      = 30             # Transcribe a new chunk every N seconds

# ── Whisper ────────────────────────────────────────────────────────────────────
WHISPER_MODEL_SIZE = "medium"       # tiny | base | small | medium | large-v3
WHISPER_DEVICE     = "cuda"         # cuda | cpu
WHISPER_COMPUTE    = "float16"      # float16 (GPU) | int8 (CPU)

# ── Ollama ─────────────────────────────────────────────────────────────────────
OLLAMA_MODEL        = "gemma3:12b"
OLLAMA_VISION_MODEL = "llava:13b"   # Used for slide OCR when prefer_vision=True

# ── Output paths ───────────────────────────────────────────────────────────────
OUTPUT_DIR    = Path.home() / "TrainingNotes"
OBSIDIAN_VAULT = Path.home() / "ObsidianVault" / "TrainingNotes"  # set to your vault

# ── Transcription / confidence ─────────────────────────────────────────────────
MAX_WORDS_PER_CHUNK  = 6000    # ~8k tokens; chunked strategy triggered above this
CONFIDENCE_THRESHOLD = 0.45   # Chunks below this are flagged [uncertain]
# avg_logprob → confidence mapping: clamped linear from [-2, 0] → [0, 1]

# ── Slide capture ──────────────────────────────────────────────────────────────
SLIDE_CAPTURE_KEY    = "F10"   # Hotkey to manually capture a slide screenshot
OCR_ENABLED          = False   # Auto continuous screen capture (disabled; use F10 instead)
OCR_INTERVAL_SECONDS = 8.0     # Seconds between auto screenshot grabs (if OCR_ENABLED=True)
OCR_CHANGE_THRESHOLD = 0.04    # Fraction of pixels that must differ to trigger auto OCR
PREFER_VISION_OCR    = False   # True = use Ollama vision instead of pytesseract

# ── Browser polling ────────────────────────────────────────────────────────────
BROWSER_POLL_INTERVAL = 3.0    # Seconds between active-window-title polls

# ── Anki ───────────────────────────────────────────────────────────────────────
ANKI_MAX_CARDS  = 20
ANKI_DECK_NAME  = "TrainingNotes"

# ── Domains ────────────────────────────────────────────────────────────────────
class Domain(str, enum.Enum):
    GENERAL     = "general"
    SECURITY    = "security"
    DEVOPS      = "devops"
    PROGRAMMING = "programming"
    DATA        = "data"
    NETWORKING  = "networking"
    BUSINESS    = "business"

DOMAIN_LABELS: dict[str, str] = {
    Domain.GENERAL:     "General / Other",
    Domain.SECURITY:    "Cybersecurity / InfoSec",
    Domain.DEVOPS:      "DevOps / Cloud / Infrastructure",
    Domain.PROGRAMMING: "Software Development / Programming",
    Domain.DATA:        "Data Science / ML / AI",
    Domain.NETWORKING:  "Networking / Systems",
    Domain.BUSINESS:    "Business / Management",
}

# Keywords used by the browser-title detector to guess a domain
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    Domain.SECURITY:    ["security", "pentest", "ctf", "hack", "cyber", "cve",
                         "infosec", "siem", "soc", "malware", "threat", "vuln"],
    Domain.DEVOPS:      ["devops", "docker", "kubernetes", "k8s", "terraform",
                         "ansible", "ci/cd", "pipeline", "jenkins", "helm",
                         "aws", "azure", "gcp", "cloud", "infra"],
    Domain.PROGRAMMING: ["python", "javascript", "java", "rust", "golang", "c++",
                         "coding", "algorithm", "software", "programming", "api",
                         "framework", "react", "django", "fastapi"],
    Domain.DATA:        ["machine learning", "deep learning", "neural", "pytorch",
                         "tensorflow", "pandas", "numpy", "data science",
                         "llm", "nlp", "ai", "model training", "kaggle"],
    Domain.NETWORKING:  ["networking", "tcp", "udp", "dns", "bgp", "ospf",
                         "firewall", "router", "switch", "vlan", "wireshark",
                         "packet", "subnet", "protocol"],
    Domain.BUSINESS:    ["management", "agile", "scrum", "product", "strategy",
                         "leadership", "marketing", "finance", "pmp"],
}

# ── Tray icon states ───────────────────────────────────────────────────────────
class TrayState(str, enum.Enum):
    IDLE       = "idle"
    RECORDING  = "recording"
    PAUSED     = "paused"
    PROCESSING = "processing"
    ERROR      = "error"
