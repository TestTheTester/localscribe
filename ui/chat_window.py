"""
ui/chat_window.py — Post-session Q&A chat window (tkinter Toplevel).
Must be created and shown from the main thread (tkinter constraint).
"""

import threading
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Optional

from llm.qa_chat import QASession


class ChatWindow:
    """
    Modal-ish chat window.  Keeps itself open; can be re-summoned after close.
    Thread-safe: LLM calls happen in a daemon thread; UI updates via root.after().
    """

    def __init__(
        self,
        root:       tk.Tk,
        qa_session: QASession,
        title:      str = "Post-Session Q&A",
    ) -> None:
        self.root       = root
        self.qa_session = qa_session
        self.title_str  = title
        self._window:   Optional[tk.Toplevel] = None
        self._thinking  = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def show(self) -> None:
        """Create or raise the chat window. Must be called from the main thread."""
        if self._window and self._window.winfo_exists():
            self._window.lift()
            self._window.focus_force()
            return
        self._build_window()

    def schedule_show(self) -> None:
        """Thread-safe wrapper — schedules show() on the main thread."""
        self.root.after(0, self.show)

    # ── Window construction ────────────────────────────────────────────────────

    def _build_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title(self.title_str)
        win.geometry("900x620")
        win.minsize(600, 400)
        win.configure(bg="#1e1e2e")
        self._window = win

        # ── Layout: left = notes preview, right = chat ─────────────────────────
        pane = tk.PanedWindow(win, orient=tk.HORIZONTAL, bg="#1e1e2e", sashwidth=4)
        pane.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Left: notes/transcript read-only panel
        left_frame = tk.Frame(pane, bg="#1e1e2e")
        pane.add(left_frame, minsize=200)

        lbl_left = tk.Label(left_frame, text="Session Notes", bg="#1e1e2e",
                            fg="#cdd6f4", font=("Segoe UI", 10, "bold"))
        lbl_left.pack(anchor="w", padx=4, pady=(0, 2))

        self._notes_box = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#181825", fg="#cdd6f4", font=("Consolas", 9),
            relief=tk.FLAT, borderwidth=0,
        )
        self._notes_box.pack(fill=tk.BOTH, expand=True)

        # Right: chat panel
        right_frame = tk.Frame(pane, bg="#1e1e2e")
        pane.add(right_frame, minsize=300)

        lbl_right = tk.Label(right_frame, text="Ask a Question", bg="#1e1e2e",
                             fg="#cdd6f4", font=("Segoe UI", 10, "bold"))
        lbl_right.pack(anchor="w", padx=4, pady=(0, 2))

        self._chat_box = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD, state=tk.DISABLED,
            bg="#181825", fg="#cdd6f4", font=("Segoe UI", 10),
            relief=tk.FLAT, borderwidth=0,
        )
        self._chat_box.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        # Configure tags for role-coloured text
        self._chat_box.tag_configure("user",      foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        self._chat_box.tag_configure("assistant", foreground="#a6e3a1")
        self._chat_box.tag_configure("system",    foreground="#f38ba8", font=("Segoe UI", 9, "italic"))

        # Input row
        input_frame = tk.Frame(right_frame, bg="#1e1e2e")
        input_frame.pack(fill=tk.X)

        self._input_var = tk.StringVar()
        self._input_box = ttk.Entry(input_frame, textvariable=self._input_var,
                                    font=("Segoe UI", 10))
        self._input_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._input_box.bind("<Return>", lambda _: self._send())
        self._input_box.bind("<KP_Enter>", lambda _: self._send())

        self._send_btn = ttk.Button(input_frame, text="Send", command=self._send, width=8)
        self._send_btn.pack(side=tk.LEFT)

        self._status_lbl = tk.Label(right_frame, text="", bg="#1e1e2e",
                                    fg="#a6adc8", font=("Segoe UI", 8, "italic"))
        self._status_lbl.pack(anchor="w")

        # Bottom: clear history + close buttons
        btn_row = tk.Frame(win, bg="#1e1e2e")
        btn_row.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(btn_row, text="Clear History",
                   command=self._clear_history).pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(btn_row, text="Close",
                   command=win.destroy).pack(side=tk.RIGHT)

        win.protocol("WM_DELETE_WINDOW", win.destroy)

        # Populate notes panel
        self._populate_notes()
        self._append_message("system", "Session loaded. Ask me anything about the material.")
        self._input_box.focus_set()

    # ── Notes panel ───────────────────────────────────────────────────────────

    def _populate_notes(self) -> None:
        notes = self.qa_session.notes
        self._notes_box.configure(state=tk.NORMAL)
        self._notes_box.delete("1.0", tk.END)
        self._notes_box.insert(tk.END, notes or "(No notes available)")
        self._notes_box.configure(state=tk.DISABLED)

    # ── Chat logic ─────────────────────────────────────────────────────────────

    def _send(self) -> None:
        if self._thinking:
            return
        text = self._input_var.get().strip()
        if not text:
            return
        self._input_var.set("")
        self._append_message("user", text)
        self._set_thinking(True)
        threading.Thread(
            target=self._llm_call, args=(text,), daemon=True
        ).start()

    def _llm_call(self, user_text: str) -> None:
        try:
            # Accumulate tokens and push to UI in real-time
            self.root.after(0, lambda: self._start_assistant_bubble())
            reply_buf: list[str] = []

            def _stream(token: str) -> None:
                reply_buf.append(token)
                self.root.after(0, lambda t=token: self._append_token(t))

            self.qa_session.ask(user_text, stream_callback=_stream)
        except Exception as e:
            self.root.after(0, lambda: self._append_message("system", f"Error: {e}"))
        finally:
            self.root.after(0, lambda: self._finish_assistant_bubble())
            self.root.after(0, lambda: self._set_thinking(False))

    def _start_assistant_bubble(self) -> None:
        """Insert 'Assistant: ' label before streaming tokens."""
        self._chat_box.configure(state=tk.NORMAL)
        self._chat_box.insert(tk.END, "\nAssistant: ", "assistant")
        self._chat_box.configure(state=tk.DISABLED)
        self._chat_box.see(tk.END)
        self._assistant_start_idx = self._chat_box.index(tk.END)

    def _append_token(self, token: str) -> None:
        self._chat_box.configure(state=tk.NORMAL)
        self._chat_box.insert(tk.END, token, "assistant")
        self._chat_box.configure(state=tk.DISABLED)
        self._chat_box.see(tk.END)

    def _finish_assistant_bubble(self) -> None:
        self._chat_box.configure(state=tk.NORMAL)
        self._chat_box.insert(tk.END, "\n\n")
        self._chat_box.configure(state=tk.DISABLED)

    def _append_message(self, role: str, text: str) -> None:
        """Thread-safe helper — must be called via root.after() from non-main threads."""
        if not self._window or not self._window.winfo_exists():
            return
        self._chat_box.configure(state=tk.NORMAL)
        prefix = {"user": "You: ", "assistant": "Assistant: ", "system": "◆ "}.get(role, "")
        self._chat_box.insert(tk.END, f"{prefix}{text}\n\n", role)
        self._chat_box.configure(state=tk.DISABLED)
        self._chat_box.see(tk.END)

    def _set_thinking(self, thinking: bool) -> None:
        self._thinking = thinking
        state = tk.DISABLED if thinking else tk.NORMAL
        self._send_btn.configure(state=state)
        self._input_box.configure(state=state)
        self._status_lbl.configure(text="Thinking…" if thinking else "")

    def _clear_history(self) -> None:
        self.qa_session.reset()
        self._chat_box.configure(state=tk.NORMAL)
        self._chat_box.delete("1.0", tk.END)
        self._chat_box.configure(state=tk.DISABLED)
        self._append_message("system", "History cleared.")
