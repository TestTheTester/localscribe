"""
llm/anki_generator.py — Generate Anki-ready Q&A pairs from notes via Ollama.
"""

import re
from config import ANKI_MAX_CARDS, Domain
from llm.ollama_client import OllamaClient
from llm.prompts import build_anki_prompt


class AnkiGenerator:
    def __init__(self, client: OllamaClient) -> None:
        self.client = client

    def generate_qa_pairs(
        self,
        notes_text: str,
        domain:     str = Domain.GENERAL,
        max_cards:  int = ANKI_MAX_CARDS,
    ) -> list[tuple[str, str]]:
        """
        Ask Ollama to generate Q&A pairs from the given notes.
        Returns a list of (front, back) tuples.
        """
        print(f"[Anki] Generating up to {max_cards} flashcards (domain={domain})...")
        prompt = build_anki_prompt(notes_text, domain, max_cards)
        raw    = self.client.chat(prompt, ctx=8192, temperature=0.4)
        pairs  = self._parse_response(raw)
        print(f"[Anki] Parsed {len(pairs)} Q&A pairs.")
        return pairs

    def _parse_response(self, raw: str) -> list[tuple[str, str]]:
        """
        Parse LLM output in the form:
            Q: <question>
            A: <answer>
        Handles minor formatting variations.
        """
        pairs: list[tuple[str, str]] = []
        # Split on blank lines to get card blocks
        blocks = re.split(r"\n\s*\n", raw.strip())

        for block in blocks:
            lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
            q_lines: list[str] = []
            a_lines: list[str] = []
            current = None

            for line in lines:
                if re.match(r"^Q\s*[:：]", line, re.IGNORECASE):
                    current  = "q"
                    q_lines.append(re.sub(r"^Q\s*[:：]\s*", "", line, flags=re.IGNORECASE))
                elif re.match(r"^A\s*[:：]", line, re.IGNORECASE):
                    current = "a"
                    a_lines.append(re.sub(r"^A\s*[:：]\s*", "", line, flags=re.IGNORECASE))
                elif current == "q":
                    q_lines.append(line)
                elif current == "a":
                    a_lines.append(line)

            q = " ".join(q_lines).strip()
            a = " ".join(a_lines).strip()
            if q and a:
                pairs.append((q, a))

        return pairs
