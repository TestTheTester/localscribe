"""
llm/qa_chat.py — Post-session Q&A chat logic (stateful, multi-turn).
UI is handled separately in ui/chat_window.py.
"""

from typing import Callable, Optional

from llm.ollama_client import OllamaClient
from llm.prompts import build_chat_messages


class QASession:
    """
    Maintains conversation history for a post-session Q&A chat.
    Each instance corresponds to one training session.
    """

    def __init__(
        self,
        client:     OllamaClient,
        transcript: str,
        notes:      str,
    ) -> None:
        self.client     = client
        self.transcript = transcript
        self.notes      = notes
        self.history:   list[dict] = []

    def ask(
        self,
        user_message:    str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a user message and return the assistant's reply.
        Conversation history is updated automatically.
        """
        messages = build_chat_messages(
            self.transcript,
            self.notes,
            self.history,
            user_message,
        )
        reply = self.client.chat_messages(
            messages,
            ctx=16384,
            temperature=0.4,
            stream_callback=stream_callback,
        )
        # Append this turn to history so context accumulates
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear history but keep transcript and notes."""
        self.history.clear()
