"""
llm/note_generator.py — Drives single-pass or chunked-merge note generation.
Automatically chooses strategy based on transcript word count.
"""

from typing import Callable, Optional

from config import MAX_WORDS_PER_CHUNK, Domain
from app_state import TranscriptChunk
from llm.ollama_client import OllamaClient
from llm.prompts import (
    build_summary_prompt,
    build_chunk_summary_prompt,
    build_merge_prompt,
)


class NoteGenerator:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    # ── Public entry point ─────────────────────────────────────────────────────

    def generate(
        self,
        chunks:          list[TranscriptChunk],
        domain:          str                             = Domain.GENERAL,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate structured Markdown notes from a list of TranscriptChunks.

        - Injects slide OCR text as [SLIDE: ...] markers at the right position.
        - Chooses single-pass for short sessions, chunked-merge for long ones.
        - stream_callback receives tokens as they stream out of Ollama.
        """
        full_text  = self._inject_ocr(chunks)
        word_count = len(full_text.split())

        print(f"\n[Notes] {word_count:,} words, {len(chunks)} chunks, domain={domain}")

        if word_count <= MAX_WORDS_PER_CHUNK:
            print("[Notes] Strategy: single-pass")
            return self._single_pass(full_text, domain, stream_callback)
        else:
            total_parts = -(-word_count // MAX_WORDS_PER_CHUNK)   # ceiling division
            print(f"[Notes] Strategy: chunked-merge ({total_parts} parts)")
            return self._chunked_merge(full_text, domain, total_parts, stream_callback)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _inject_ocr(self, chunks: list[TranscriptChunk]) -> str:
        """Assemble transcript text, inserting slide OCR at the chunk boundary."""
        parts = []
        for c in chunks:
            if c.ocr_text and c.ocr_text.strip():
                parts.append(f"[SLIDE: {c.ocr_text.strip()}]")
            flag = " [uncertain]" if c.is_uncertain else ""
            parts.append(f"[{c.wall_clock.strftime('%H:%M:%S')}]{flag} {c.text}")
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
