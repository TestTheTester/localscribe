"""
ui/session_dialog.py — tkinter dialog for configuring a new recording session.
Returns a dict with 'course', 'domain', and 'device_id' on OK; None on cancel.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional

from config import DOMAIN_LABELS, Domain
from audio.capture import list_input_devices


def show_session_dialog(root: tk.Tk, suggested_course: str = "") -> Optional[dict]:
    """
    Display a blocking modal dialog.  Returns:
        {"course": str, "domain": str, "device_id": int}
    or None if the user cancels.

    Must be called from the main thread.
    """
    result: dict[str, object] = {}

    dlg = tk.Toplevel(root)
    dlg.title("New Recording Session")
    dlg.resizable(False, False)
    dlg.grab_set()          # modal
    dlg.configure(bg="#1e1e2e")

    style = ttk.Style(dlg)
    style.theme_use("clam")
    style.configure("TLabel",  background="#1e1e2e", foreground="#cdd6f4",
                    font=("Segoe UI", 10))
    style.configure("TEntry",  fieldbackground="#313244", foreground="#cdd6f4",
                    font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 10))
    style.configure("TCombobox", fieldbackground="#313244", foreground="#cdd6f4")

    pad = {"padx": 12, "pady": 6}

    # ── Course title ───────────────────────────────────────────────────────────
    ttk.Label(dlg, text="Course / Video title:").grid(
        row=0, column=0, sticky="w", **pad)
    course_var = tk.StringVar(value=suggested_course)
    course_entry = ttk.Entry(dlg, textvariable=course_var, width=46)
    course_entry.grid(row=0, column=1, columnspan=2, sticky="ew", **pad)
    course_entry.focus_set()

    # ── Domain ────────────────────────────────────────────────────────────────
    ttk.Label(dlg, text="Domain:").grid(row=1, column=0, sticky="w", **pad)
    domain_options = list(DOMAIN_LABELS.values())
    domain_keys    = list(DOMAIN_LABELS.keys())
    domain_var     = tk.StringVar(value=DOMAIN_LABELS[Domain.GENERAL])
    domain_cb = ttk.Combobox(dlg, textvariable=domain_var,
                              values=domain_options, state="readonly", width=44)
    domain_cb.grid(row=1, column=1, columnspan=2, sticky="ew", **pad)

    # ── Audio device ──────────────────────────────────────────────────────────
    ttk.Label(dlg, text="Audio device:").grid(row=2, column=0, sticky="w", **pad)
    devices     = list_input_devices()
    dev_labels  = [f"[{d['index']}] {d['name']}" for d in devices]
    dev_var     = tk.StringVar(value=dev_labels[0] if dev_labels else "No devices found")
    dev_cb = ttk.Combobox(dlg, textvariable=dev_var,
                          values=dev_labels, state="readonly", width=44)
    dev_cb.grid(row=2, column=1, columnspan=2, sticky="ew", **pad)

    # Try to pre-select a loopback device
    from audio.capture import find_loopback_device
    loopback_id = find_loopback_device()
    if loopback_id is not None:
        for i, d in enumerate(devices):
            if d["index"] == loopback_id:
                dev_cb.current(i)
                break

    # ── Slide OCR toggle ──────────────────────────────────────────────────────
    ocr_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(dlg, text="Enable slide OCR (mss + tesseract)",
                    variable=ocr_var).grid(
        row=3, column=0, columnspan=3, sticky="w", **pad)

    # ── Buttons ───────────────────────────────────────────────────────────────
    def _ok() -> None:
        course = course_var.get().strip() or "Training Session"
        label  = domain_var.get()
        domain = domain_keys[domain_options.index(label)] if label in domain_options else Domain.GENERAL
        sel_idx = dev_cb.current()
        device_id = devices[sel_idx]["index"] if 0 <= sel_idx < len(devices) else 0
        result["course"]    = course
        result["domain"]    = domain
        result["device_id"] = device_id
        result["enable_ocr"] = ocr_var.get()
        dlg.destroy()

    def _cancel() -> None:
        dlg.destroy()

    btn_frame = tk.Frame(dlg, bg="#1e1e2e")
    btn_frame.grid(row=4, column=0, columnspan=3, pady=10)
    ttk.Button(btn_frame, text="Start Recording", command=_ok, width=18).pack(
        side=tk.LEFT, padx=6)
    ttk.Button(btn_frame, text="Cancel", command=_cancel, width=10).pack(
        side=tk.LEFT, padx=6)

    dlg.bind("<Return>", lambda _: _ok())
    dlg.bind("<Escape>", lambda _: _cancel())

    # Centre the dialog over the (hidden) root window
    dlg.update_idletasks()
    x = root.winfo_x() + (root.winfo_width()  - dlg.winfo_width())  // 2
    y = root.winfo_y() + (root.winfo_height() - dlg.winfo_height()) // 2
    dlg.geometry(f"+{max(0,x)}+{max(0,y)}")

    root.wait_window(dlg)
    return result if result else None
