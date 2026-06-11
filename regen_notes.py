"""
regen_notes.py — Regenerate notes from a saved *_transcript.md file.

Usage:
    python regen_notes.py path/to/session_transcript.md [--domain DOMAIN]

The regenerated notes are written alongside the transcript as
    <same_dir>/<same_stem>_regen.md
"""

import argparse
import datetime
import re
import sys
from pathlib import Path

# ── Bootstrap sys.path so local modules resolve ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from app_state import TranscriptChunk
from config import DOMAIN_LABELS, Domain
from llm.ollama_client import OllamaClient
from llm.note_generator import NoteGenerator


# ── Transcript parser ─────────────────────────────────────────────────────────

_CHUNK_RE = re.compile(
    r"^\[(\d{2}:\d{2}:\d{2})\](\s+\[uncertain\])?\s+(.*)"
)
_SLIDE_RE = re.compile(r"^\s+\[SLIDE\]\s+(.*)")
_DATE_RE  = re.compile(r"^\*(\d{4}-\d{2}-\d{2})")


def parse_transcript(path: Path) -> tuple[str, list[TranscriptChunk]]:
    """
    Returns (date_str, chunks).  date_str may be empty if the header is absent.
    """
    text    = path.read_text(encoding="utf-8")
    lines   = text.splitlines()
    chunks: list[TranscriptChunk] = []

    # Try to extract the session date from the header line  *YYYY-MM-DD HH:MM*
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    for line in lines[:5]:
        m = _DATE_RE.search(line)
        if m:
            date_str = m.group(1)
            break

    idx = 0
    pending_slide: str | None = None

    for line in lines:
        slide_m = _SLIDE_RE.match(line)
        if slide_m and chunks:
            chunks[-1].ocr_text = slide_m.group(1).strip()
            continue

        chunk_m = _CHUNK_RE.match(line)
        if chunk_m:
            time_str    = chunk_m.group(1)                  # HH:MM:SS
            uncertain   = bool(chunk_m.group(2))
            body        = chunk_m.group(3).strip()

            wall_clock = datetime.datetime.strptime(
                f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
            )
            chunks.append(TranscriptChunk(
                index       = idx,
                text        = body,
                wall_clock  = wall_clock,
                confidence  = 0.4 if uncertain else 1.0,
                is_uncertain= uncertain,
                ocr_text    = None,
            ))
            idx += 1

    return date_str, chunks


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate notes from a transcript file.")
    parser.add_argument("transcript", type=Path, help="Path to the *_transcript.md file")
    parser.add_argument(
        "--domain", "-d",
        default=Domain.GENERAL,
        choices=[d.value for d in Domain],
        help="Domain hint for the LLM (default: general)",
    )
    args = parser.parse_args()

    transcript_path: Path = args.transcript.resolve()
    if not transcript_path.exists():
        sys.exit(f"[Error] File not found: {transcript_path}")

    print(f"[Regen] Parsing transcript: {transcript_path}")
    date_str, chunks = parse_transcript(transcript_path)

    if not chunks:
        sys.exit("[Error] No transcript chunks found — is this the right file?")

    print(f"[Regen] Found {len(chunks)} chunks  (date={date_str}, domain={args.domain})")

    # ── Connect to Ollama ─────────────────────────────────────────────────────
    client   = OllamaClient()
    note_gen = NoteGenerator(client)

    # ── Generate ──────────────────────────────────────────────────────────────
    print("[Regen] Sending to Ollama…\n" + "─" * 60)
    notes = note_gen.generate(
        chunks          = chunks,
        domain          = args.domain,
        stream_callback = lambda t: print(t, end="", flush=True),
    )
    print("\n" + "─" * 60)

    # ── Write output ──────────────────────────────────────────────────────────
    out_path = transcript_path.with_name(
        transcript_path.stem.replace("_transcript", "") + "_notes_regen.md"
    )
    out_path.write_text(notes, encoding="utf-8")
    print(f"\n[Regen] Notes written → {out_path}")


if __name__ == "__main__":
    main()
