"""
Training Video Note-Taker
Captures system audio in real-time, transcribes with faster-whisper,
then generates structured notes + summary via local Ollama (gemma3:12b).
"""

import sys
import os
import threading
import queue
import time
import wave
import tempfile
import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import ollama

# ── Configuration ──────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000          # Hz — Whisper expects 16kHz
CHANNELS           = 1
CHUNK_SECONDS      = 30             # Transcribe every N seconds
WHISPER_MODEL_SIZE = "medium"       # Options: tiny, base, small, medium, large-v3
WHISPER_DEVICE     = "cuda"         # Use "cpu" if GPU issues occur
WHISPER_COMPUTE    = "float16"      # float16 for GPU, int8 for CPU
OLLAMA_MODEL       = "gemma3:12b"   # Change to llama3.1:8b if preferred
OUTPUT_DIR         = Path.home() / "TrainingNotes"

OUTPUT_DIR.mkdir(exist_ok=True)

# ── Globals ────────────────────────────────────────────────────────────────────
audio_queue        = queue.Queue()
transcript_chunks  = []
stop_event         = threading.Event()


# ── Audio Capture ──────────────────────────────────────────────────────────────
def find_loopback_device():
    """Find WASAPI loopback device (system audio) automatically."""
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        # WASAPI loopback devices typically contain these keywords
        if dev["max_input_channels"] > 0 and (
            "loopback" in name or "stereo mix" in name or
            "what u hear" in name or "wave out mix" in name
        ):
            return i
    return None


def list_input_devices():
    """Print all input devices for manual selection."""
    devices = sd.query_devices()
    print("\n📋 Available input devices:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i:2d}] {dev['name']}")
    print()


def audio_callback(indata, frames, time_info, status):
    """Called by sounddevice for each audio chunk."""
    if status:
        pass  # Suppress WASAPI warnings
    audio_queue.put(indata.copy())


# ── Transcription ──────────────────────────────────────────────────────────────
def transcribe_chunk(model, audio_data: np.ndarray) -> str:
    """Transcribe a numpy audio array using faster-whisper."""
    # Write to temp WAV file (faster-whisper accepts file path)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())

    segments, _ = model.transcribe(
        tmp_path,
        beam_size=5,
        language="en",
        vad_filter=True,           # Skip silent sections
        vad_parameters={"min_silence_duration_ms": 500},
    )
    text = " ".join(seg.text.strip() for seg in segments)
    os.unlink(tmp_path)
    return text.strip()


def transcription_worker(model):
    """Continuously pull audio from queue, transcribe in chunks."""
    buffer = np.zeros((0,), dtype=np.float32)
    chunk_samples = SAMPLE_RATE * CHUNK_SECONDS
    chunk_num = 0

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            data = audio_queue.get(timeout=1.0)
            buffer = np.concatenate([buffer, data.flatten()])

            if len(buffer) >= chunk_samples:
                chunk_num += 1
                audio_chunk = buffer[:chunk_samples]
                buffer = buffer[chunk_samples:]

                print(f"\n🎙️  Transcribing chunk {chunk_num}...", end="", flush=True)
                text = transcribe_chunk(model, audio_chunk)

                if text:
                    transcript_chunks.append(text)
                    print(f" ✓ ({len(text)} chars)")
                    print(f"   💬 {text[:120]}{'...' if len(text) > 120 else ''}")
                else:
                    print(" (silence/no speech)")

        except queue.Empty:
            continue

    # Process any remaining audio
    if len(buffer) > SAMPLE_RATE * 2:  # At least 2 seconds
        print(f"\n🎙️  Transcribing final chunk...", end="", flush=True)
        text = transcribe_chunk(model, buffer)
        if text:
            transcript_chunks.append(text)
            print(f" ✓")


# ── LLM Note Generation ────────────────────────────────────────────────────────
SUMMARY_PROMPT = """You are an expert technical note-taker. Below is a transcript from a training video.

Your task:
1. Write a concise SUMMARY (3-5 sentences) covering the main topic and key takeaways.
2. Write DETAILED NOTES in this structure:
   - ## Key Concepts (bullet points of core ideas)
   - ## Important Details (technical specifics, definitions, examples mentioned)
   - ## Action Items / Things to Remember (practical takeaways)
   - ## Questions to Explore (gaps or things worth researching further)

Be specific. Use the actual terminology from the transcript. Do not pad with filler.

TRANSCRIPT:
{transcript}

---
Respond in clean Markdown format. Start with the Summary section."""

CHUNK_SUMMARY_PROMPT = """You are an expert technical note-taker. Below is PART {part} of {total} parts of a long training video transcript.

Extract and list only the KEY POINTS from this section as concise bullet points.
Focus on: concepts introduced, definitions, examples, and important details.
Do NOT write a summary yet — just bullet points of key information.

TRANSCRIPT SECTION:
{transcript}

---
Respond with bullet points only. Be specific and use exact terminology."""

MERGE_PROMPT = """You are an expert technical note-taker. Below are key points extracted from different sections of a long training video.

Synthesise these into a complete set of notes:
1. Write a SUMMARY (4-6 sentences) covering the full video's topic and key takeaways.
2. Write DETAILED NOTES in this structure:
   - ## Key Concepts (consolidated core ideas, remove duplicates)
   - ## Important Details (technical specifics, definitions, examples)
   - ## Action Items / Things to Remember (practical takeaways)
   - ## Questions to Explore (gaps or things worth researching further)

EXTRACTED POINTS FROM ALL SECTIONS:
{combined_points}

---
Respond in clean Markdown format. Start with the Summary section."""

# Token estimate: ~0.75 words per token, safe limit per chunk
MAX_WORDS_PER_CHUNK = 6000  # ~8000 tokens — safe for 8192 ctx window


def call_ollama(prompt: str, ctx: int = 8192) -> str:
    """Stream a response from Ollama and return full text."""
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.3, "num_ctx": ctx},
        stream=True,
    )
    result = ""
    for chunk in response:
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        result += token
    return result


def generate_notes(full_transcript: str, video_title: str) -> str:
    """Generate notes — uses chunked summarisation for long transcripts."""
    word_count = len(full_transcript.split())
    print(f"\n🤖 Sending transcript to {OLLAMA_MODEL} for note generation...")
    print(f"   Transcript: {word_count:,} words across {len(transcript_chunks)} chunks")

    # ── Short video: single-pass ───────────────────────────────────
    if word_count <= MAX_WORDS_PER_CHUNK:
        print("   Mode: single-pass (fits in context window)\n")
        print("📝 Generating notes:\n" + "─" * 60)
        notes = call_ollama(
            SUMMARY_PROMPT.format(transcript=full_transcript),
            ctx=32768
        )
        print("\n" + "─" * 60)
        return notes

    # ── Long video: chunked summarisation ─────────────────────────
    words = full_transcript.split()
    total_parts = -(-word_count // MAX_WORDS_PER_CHUNK)  # ceiling division
    print(f"   Mode: chunked ({total_parts} parts — long video detected)\n")

    section_points = []
    for i in range(total_parts):
        start = i * MAX_WORDS_PER_CHUNK
        end   = start + MAX_WORDS_PER_CHUNK
        section_text = " ".join(words[start:end])

        print(f"\n📝 Extracting key points from part {i+1}/{total_parts}:\n" + "─" * 60)
        points = call_ollama(
            CHUNK_SUMMARY_PROMPT.format(
                part=i + 1,
                total=total_parts,
                transcript=section_text
            ),
            ctx=16384
        )
        print("\n" + "─" * 60)
        section_points.append(f"### Section {i+1}/{total_parts}\n{points}")

    # ── Final merge pass ───────────────────────────────────────────
    print(f"\n🔀 Merging all sections into final notes...\n" + "─" * 60)
    combined = "\n\n".join(section_points)
    final_notes = call_ollama(
        MERGE_PROMPT.format(combined_points=combined),
        ctx=32768
    )
    print("\n" + "─" * 60)
    return final_notes


# ── File Output ────────────────────────────────────────────────────────────────
def save_output(video_title: str, transcript: str, notes: str):
    """Save transcript and notes to timestamped Markdown files."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in video_title)
    safe_title = safe_title[:50].strip()

    base_name = f"{timestamp}_{safe_title}" if safe_title else timestamp

    # Save notes
    notes_path = OUTPUT_DIR / f"{base_name}_notes.md"
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(f"# Notes: {video_title}\n")
        f.write(f"*Captured: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}*\n\n")
        f.write(notes)

    # Save raw transcript
    transcript_path = OUTPUT_DIR / f"{base_name}_transcript.md"
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(f"# Transcript: {video_title}\n")
        f.write(f"*Captured: {datetime.datetime.now().strftime('%B %d, %Y %H:%M')}*\n\n")
        f.write(transcript)

    return notes_path, transcript_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  🎓 Training Video Note-Taker")
    print(f"  Model: {OLLAMA_MODEL} via Ollama")
    print(f"  Transcription: faster-whisper ({WHISPER_MODEL_SIZE})")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # ── Check Ollama is running ────────────────────────────────────
    print("\n🔍 Checking Ollama connection...", end="", flush=True)
    try:
        ollama.list()
        print(" ✓")
    except Exception:
        print(" ✗")
        print("\n❌ Cannot connect to Ollama. Please start it with:  ollama serve")
        sys.exit(1)

    # ── Load Whisper model ─────────────────────────────────────────
    print(f"📦 Loading Whisper ({WHISPER_MODEL_SIZE}) on {WHISPER_DEVICE}...", end="", flush=True)
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
        print(" ✓")
    except Exception as e:
        print(f" ✗\n   Error: {e}")
        print("   Retrying on CPU...")
        model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("   CPU fallback ✓")

    # ── Find audio device ──────────────────────────────────────────
    device_id = find_loopback_device()

    if device_id is None:
        list_input_devices()
        print("⚠️  No loopback device auto-detected.")
        print("   Enable 'Stereo Mix' in Windows Sound settings, or enter device ID:")
        try:
            device_id = int(input("   Device ID: ").strip())
        except ValueError:
            print("❌ Invalid device ID. Exiting.")
            sys.exit(1)
    else:
        print(f"🔊 Audio device: [{device_id}] {sd.query_devices(device_id)['name']}")

    # ── Get video title ────────────────────────────────────────────
    print()
    video_title = input("📌 Enter video/course title (or press Enter to skip): ").strip()
    if not video_title:
        video_title = "Training Session"

    # ── Start capture ──────────────────────────────────────────────
    print(f"\n▶️  Starting capture... Play your training video now.")
    print("   Press  ENTER  to stop recording and generate notes.\n")

    transcription_thread = threading.Thread(
        target=transcription_worker, args=(model,), daemon=True
    )

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        device=device_id,
        callback=audio_callback,
        blocksize=SAMPLE_RATE,  # 1-second blocks
    ):
        transcription_thread.start()
        input()  # Wait for Enter key

    print("\n⏹️  Stopping capture...")
    stop_event.set()
    transcription_thread.join(timeout=60)

    if not transcript_chunks:
        print("\n⚠️  No speech was transcribed. Check your audio device and try again.")
        sys.exit(0)

    full_transcript = "\n\n".join(transcript_chunks)
    print(f"\n✅ Transcription complete. Total chunks: {len(transcript_chunks)}")

    # ── Generate notes ─────────────────────────────────────────────
    notes = generate_notes(full_transcript, video_title)

    # ── Save files ─────────────────────────────────────────────────
    notes_path, transcript_path = save_output(video_title, full_transcript, notes)

    print(f"\n✅ Done! Files saved to: {OUTPUT_DIR}")
    print(f"   📄 Notes:      {notes_path.name}")
    print(f"   📜 Transcript: {transcript_path.name}")
    print(f"\n   Open folder: explorer \"{OUTPUT_DIR}\"")


if __name__ == "__main__":
    main()