"""
llm/ollama_client.py — Thin, retry-safe wrapper around the Ollama Python SDK.
"""

import base64
import io
from typing import Callable, Optional

import ollama

from config import OLLAMA_MODEL, OLLAMA_VISION_MODEL


class OllamaConnectionError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        model:        str = OLLAMA_MODEL,
        vision_model: str = OLLAMA_VISION_MODEL,
    ) -> None:
        self.model        = model
        self.vision_model = vision_model

    # ── Health check ──────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            ollama.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """Return list of model names currently pulled in Ollama."""
        try:
            result = ollama.list()
            # SDK returns an object with a .models attribute (list of Model objects)
            models = result.models if hasattr(result, "models") else result.get("models", [])
            return [m.model if hasattr(m, "model") else m.get("name", "") for m in models]
        except Exception:
            return []

    def find_vision_model(self) -> Optional[str]:
        """
        Return the name of the best available vision-capable model from Ollama's
        local model list, or None if none is found.

        Priority order — first match wins:
          gemma3 (supports vision natively), llava, llama3.2-vision,
          minicpm-v, moondream, bakllava, phi3-vision, qwen2-vl
        """
        # Keywords that identify vision-capable models (checked as substrings)
        VISION_KEYWORDS = [
            "gemma3",
            "llava",
            "llama3.2-vision",
            "minicpm-v",
            "moondream",
            "bakllava",
            "phi3-vision",
            "qwen2-vl",
        ]
        available = self.list_models()
        for keyword in VISION_KEYWORDS:
            for name in available:
                if keyword in name.lower():
                    return name
        return None

    def resolve_vision_model(self) -> bool:
        """
        Auto-detect the best vision model and update self.vision_model.
        Returns True if a usable vision model was found.
        """
        # If already configured and present, keep it
        if self.vision_model:
            available = self.list_models()
            if any(self.vision_model in m for m in available):
                return True

        found = self.find_vision_model()
        if found:
            self.vision_model = found
            print(f"[OCR] Vision model auto-selected: {found}")
            return True

        self.vision_model = ""
        print("[OCR] No vision-capable model found in Ollama — vision OCR disabled.")
        print("      Pull one with: ollama pull llava  (or gemma3:12b already supports vision)")
        return False

    # ── Text chat ─────────────────────────────────────────────────────────────

    def chat(
        self,
        prompt: str,
        ctx: int = 8192,
        temperature: float = 0.3,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a single-user-turn message to Ollama.
        If stream_callback is provided it is called for every token as it arrives.
        Returns the full response string.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_ctx": ctx},
                stream=True,
            )
            result = ""
            for chunk in response:
                token = chunk["message"]["content"]
                result += token
                if stream_callback:
                    stream_callback(token)
            return result
        except Exception as e:
            raise OllamaConnectionError(f"Ollama chat failed: {e}") from e

    def chat_messages(
        self,
        messages: list[dict],
        ctx: int = 8192,
        temperature: float = 0.3,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Multi-turn chat using a messages list (role/content dicts)."""
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": temperature, "num_ctx": ctx},
                stream=True,
            )
            result = ""
            for chunk in response:
                token = chunk["message"]["content"]
                result += token
                if stream_callback:
                    stream_callback(token)
            return result
        except Exception as e:
            raise OllamaConnectionError(f"Ollama multi-turn chat failed: {e}") from e

    # ── Vision chat ───────────────────────────────────────────────────────────

    def vision_chat(
        self,
        prompt: str,
        image_b64: str,
        ctx: int = 4096,
    ) -> str:
        """
        Send image+prompt to the vision model.
        image_b64 must be a base-64 encoded PNG/JPEG string.
        Raises OllamaConnectionError if no vision model is configured.
        """
        if not self.vision_model:
            raise OllamaConnectionError("No vision model configured.")
        try:
            response = ollama.chat(
                model=self.vision_model,
                messages=[{
                    "role":    "user",
                    "content": prompt,
                    "images":  [image_b64],
                }],
                options={"temperature": 0.1, "num_ctx": ctx},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            raise OllamaConnectionError(f"Ollama vision chat failed: {e}") from e

    # ── Convenience ───────────────────────────────────────────────────────────

    @staticmethod
    def pil_to_b64(image) -> str:
        """Convert a PIL Image to a base-64 PNG string."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")
