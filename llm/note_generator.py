"""
llm/note_generator.py — Drives single-pass or chunked-merge note generation.
Automatically chooses strategy based on transcript word count.
"""

from typing import Callable, Optional

from config import MAX_WORDS_PER_CHUNK, Domain, REVIEW_MODEL, ENABLE_NOTE_REVIEW
from app_state import TranscriptChunk
from llm.ollama_client import OllamaClient
from llm.prompts import (
    build_summary_prompt,
    build_chunk_summary_prompt,
    build_merge_prompt,
    build_review_prompt,
    build_revision_prompt,
)


class NoteGenerator:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    # ── Public entry point ─────────────────────────────────────────────────────

    def generate(
        self,
        chunks:          list[TranscriptChunk],
        slide_captures:  list                            = None,
        domain:          str                             = Domain.GENERAL,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate structured Markdown notes from a list of TranscriptChunks.

        - Injects slide OCR text as [SLIDE: ...] markers at the right position.
        - Chooses single-pass for short sessions, chunked-merge for long ones.
        - stream_callback receives tokens as they stream out of Ollama.
        """
        full_text  = self._inject_ocr(chunks, slide_captures or [])
        word_count = len(full_text.split())

        print(f"\n[Notes] {word_count:,} words, {len(chunks)} chunks, domain={domain}")

        if word_count <= MAX_WORDS_PER_CHUNK:
            print("[Notes] Strategy: single-pass")
            draft = self._single_pass(full_text, domain, stream_callback)
        else:
            total_parts = -(-word_count // MAX_WORDS_PER_CHUNK)   # ceiling division
            print(f"[Notes] Strategy: chunked-merge ({total_parts} parts)")
            draft = self._chunked_merge(full_text, domain, total_parts, stream_callback)

        if ENABLE_NOTE_REVIEW:
            return self._review_and_revise(draft, domain, stream_callback)
        return draft

    # ── Private helpers ────────────────────────────────────────────────────────

    def _review_and_revise(
        self,
        draft:           str,
        domain:          str,
        stream_callback: Optional[Callable[[str], None]],
    ) -> str:
        print(f"\n[Review] Sending notes to {REVIEW_MODEL} for critique...\n" + "─" * 60)
        review = self.client.chat_with_model(
            model=REVIEW_MODEL,
            prompt=build_review_prompt(draft, domain),
            ctx=16384,
            temperature=0.2,
            stream_callback=lambda t: print(t, end="", flush=True),
        )
        print("\n" + "─" * 60)

        if "no issues" in review.strip().lower()[:50]:
            print("[Review] No issues found — keeping original notes.")
            return draft

        print(f"[Review] Critique received. Sending back to {self.client.model} for revision...\n" + "─" * 60)
        revised = self.client.chat(
            build_revision_prompt(draft, review, domain),
            ctx=32768,
            stream_callback=stream_callback,
        )
        print("\n" + "─" * 60)
        return revised

    def _inject_ocr(self, chunks: list[TranscriptChunk], slide_captures: list) -> str:
        """
        Assemble the transcript, interleaving manual slide captures by wall-clock
        time.  Each slide gets a [SCREENSHOT:filename] marker so the LLM can echo
        it back as <!-- SCREENSHOT:filename --> for later post-processing.
        """
        events: list[tuple] = []
        for c in chunks:
            events.append(("chunk", c.wall_clock, c))
        for s in slide_captures:
            events.append(("slide", s.wall_clock, s))
        events.sort(key=lambda e: e[1])

        parts = []
        for kind, _, obj in events:
            if kind == "slide":
                lines = [f"[SCREENSHOT:{obj.image_path.name}]"]
                if obj.ocr_text and obj.ocr_text.strip():
                    lines.append(f"[SLIDE: {obj.ocr_text.strip()}]")
                parts.append("\n".join(lines))
            else:
                # legacy auto-capture path (OCR_ENABLED=True)
                if obj.ocr_image_path:
                    lines = [f"[SCREENSHOT:{obj.ocr_image_path.name}]"]
                    if obj.ocr_text and obj.ocr_text.strip():
                        lines.append(f"[SLIDE: {obj.ocr_text.strip()}]")
                    parts.append("\n".join(lines))
                elif obj.ocr_text and obj.ocr_text.strip():
                    parts.append(f"[SLIDE: {obj.ocr_text.strip()}]")
                flag = " [uncertain]" if obj.is_uncertain else ""
                parts.append(f"[{obj.wall_clock.strftime('%H:%M:%S')}]{flag} {obj.text}")

        return "\n\n".join(parts)

    def _single_pass(
        self,
        full_text:       str,
        domain:          str,
        stream_callback: Optional[Callable[[str], None]],
    ) -> str:
        prompt = build_summary_prompt(full_text, domain)
        print("[Notes] Sending to Ollama (single-pass)...\n" + "─" * 60)
        result = self.client.chat(
            prompt,
            ctx=32768,
            stream_callback=stream_callback,
        )
        print("\n" + "─" * 60)
        return result

    def _chunked_merge(
        self,
        full_text:       str,
        domain:          str,
        total_parts:     int,
        stream_callback: Optional[Callable[[str], None]],
    ) -> str:
        words          = full_text.split()
        section_points = []

        for i in range(total_parts):
            start        = i * MAX_WORDS_PER_CHUNK
            end          = start + MAX_WORDS_PER_CHUNK
            section_text = " ".join(words[start:end])

            print(f"\n[Notes] Extracting key points: part {i+1}/{total_parts}\n" + "─" * 60)
            points = self.client.chat(
                build_chunk_summary_prompt(section_text, i + 1, total_parts, domain),
                ctx=16384,
                stream_callback=stream_callback,
            )
            print("\n" + "─" * 60)
            section_points.append(f"### Section {i+1}/{total_parts}\n{points}")

        combined = "\n\n".join(section_points)
        print(f"\n[Notes] Merging {total_parts} sections into final notes...\n" + "─" * 60)
        result = self.client.chat(
            build_merge_prompt(combined, domain),
            ctx=32768,
            stream_callback=stream_callback,
        )
        print("\n" + "─" * 60)
        return result
