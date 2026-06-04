# 🎓 Training Video Note-Taker

Watches training videos playing in your browser, transcribes audio in real-time using Whisper (GPU-accelerated), and generates structured notes via a local LLM (Ollama). Also captures slide screenshots on demand, embeds them in your notes, and exports Anki flashcards. Everything runs locally — no data leaves your machine.

---

## Requirements

| Component | Required | Notes |
|---|---|---|
| Python | **3.11 exactly** | PyTorch CUDA builds require 3.11; 3.12/3.13 are not supported |
| GPU | Optional | CPU works; RTX 4050 6 GB recommended |
| RAM | 8 GB min | 16 GB+ recommended for large Whisper + Ollama |
| Ollama | Latest | https://ollama.com/download |
| Disk | ~12 GB | PyTorch + Whisper + gemma3:12b model |

---

## One-Time Setup

### 1. Install Prerequisites

- **Python 3.11**: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe  
  ✅ Check "Add Python to PATH" and use "Customize installation"

- **Ollama**: https://ollama.com/download  
  Install and it starts automatically.

### 2. Enable System Audio Capture

The app auto-detects loopback devices. Enable one of these:

**Option A — Stereo Mix (built-in, free):**
1. Press `Win+R` → type `mmsys.cpl` → Enter
2. Go to the **Recording** tab
3. Right-click empty area → **Show Disabled Devices**
4. Right-click **Stereo Mix** → **Enable**

**Option B — VB-Cable (more reliable, free):**
1. Download from https://vb-audio.com/Cable/
2. Install and reboot
3. Set your browser's audio output to **CABLE Input**  
   (`ms-settings:apps-volume` → change browser output device)
4. To still hear audio: `mmsys.cpl` → Recording → CABLE Output Properties → Listen tab → enable **Listen**

### 3. Run Setup & Launch

Double-click **`run.bat`**

On first run it will:
- Create a Python 3.11 virtual environment
- Install PyTorch with CUDA 12.1 (for RTX series GPUs)
- Install faster-whisper, sounddevice, pystray, mss, pytesseract, and all other dependencies
- Pull `gemma3:12b` from Ollama (~8 GB download)

First run takes 10–20 minutes. Subsequent launches start in seconds.

---

## Usage

The app runs as a **system tray application** — look for its icon in the bottom-right taskbar.

### Starting a Session

1. Double-click `run.bat` — a tray icon appears
2. **Right-click the tray icon → ▶ New Session**
3. Fill in the session dialog:
   - **Course title** — auto-filled from the active browser tab if detected
   - **Domain** — picks the right note-taking focus (see Domains below)
   - **Audio device** — select your loopback device
4. Click **Start** — recording begins immediately
5. Switch to your browser and play the video

### During a Session

| Action | How |
|---|---|
| Pause / Resume | **F9** or tray → ⏸ Pause / Resume |
| **Capture a slide screenshot** | **F10** — takes a screenshot immediately, saves it, and OCRs it |
| Stop and generate notes | Tray → ⏹ Stop & Generate Notes |

> **F10 tip:** Press F10 whenever a diagram, architecture slide, or code snippet appears on screen. The screenshot is saved and embedded in your final notes automatically.

### After Recording

1. Tray → **⏹ Stop & Generate Notes** — waits for transcription to drain, then generates notes (1–3 min depending on session length)
2. A popup confirms the notes are ready with the file path
3. Tray → **💬 Post-Session Q&A** — opens a chat window to ask questions about the session
4. Tray → **📂 Open Notes Folder** — opens the output directory in Explorer
5. Tray → **✕ Quit** — exits the app; the `run.bat` terminal shows `[DONE]` and waits for a key

---

## Output Files

Each session creates a folder: `~/TrainingNotes/<date>/<course>/`

| File / Folder | Contents |
|---|---|
| `<course>_<time>_notes.md` | Structured notes with embedded slide images |
| `<course>_<time>_transcript.md` | Full timestamped transcript |
| `slides/slide_HHMMSS.png` | Screenshots captured with F10 during the session |
| `~/TrainingNotes/anki_<course>_<date>.txt` | Anki-importable flashcard deck |

### Notes structure

- **Summary** — 8–12 sentence overview of the full session
- **Key Concepts** — every concept with definition and why it matters
- **Important Details** — commands, flags, config values, step-by-step procedures
- **Extended Concepts** — terms mentioned but not explained, filled in from LLM knowledge
- **Action Items / Things to Remember** — practical takeaways with reasoning
- **Questions to Explore** — gaps worth researching further
- **Captured Slides** *(if F10 was used)* — screenshot gallery of any slides not already embedded inline

Slide screenshots are embedded inline at the point in the notes where they were captured, falling back to a gallery section for any the LLM didn't place inline.

### Anki Flashcards

The app automatically generates flashcards from the notes and saves them as a tab-separated `.txt` file.

**To import into Anki:**
1. Open Anki → **File → Import**
2. Select the `anki_*.txt` file from `~/TrainingNotes/`
3. Anki reads the `#deck:`, `#separator:Tab` headers automatically
4. Click **Import** — cards land in the `TrainingNotes` deck

---

## Domains

When starting a session, pick the domain that matches the video content. This tells the LLM what vocabulary and level of detail to apply:

| Domain | Focus |
|---|---|
| **General** | Balanced coverage of all topics |
| **Cybersecurity** | CVEs, MITRE ATT&CK, tools (Burp, Nmap, Metasploit), frameworks |
| **DevOps / Cloud** | CI/CD, Docker, Kubernetes, Terraform, AWS/Azure/GCP |
| **Programming** | Language features, design patterns, APIs, performance |
| **Data Science / ML** | Model architectures, training, PyTorch, metrics, preprocessing |
| **Networking** | Protocols (TCP/IP, DNS, BGP), OSI model, subnetting, VLANs |
| **Business** | Agile, OKRs, strategy frameworks, leadership, finance |

---

## Configuration

All settings are in **`config.py`**:

```python
# Audio / Transcription
CHUNK_SECONDS      = 30             # Transcribe every N seconds
WHISPER_MODEL_SIZE = "medium"       # tiny | base | small | medium | large-v3
WHISPER_DEVICE     = "cuda"         # cuda | cpu

# LLM
OLLAMA_MODEL       = "gemma3:12b"   # Change to llama3.1:8b for lighter hardware

# Slide capture
SLIDE_CAPTURE_KEY  = "F10"          # Hotkey for manual screenshot
OCR_ENABLED        = False          # True = enable automatic continuous capture

# Output
OUTPUT_DIR         = Path.home() / "TrainingNotes"

# Anki
ANKI_MAX_CARDS     = 20
ANKI_DECK_NAME     = "TrainingNotes"
```

**Whisper model tradeoffs:**

| Model | Speed (RTX 4050) | Accuracy |
|---|---|---|
| `tiny` | ~5× real-time | Basic |
| `small` | ~3× real-time | Good |
| `medium` | ~1.5× real-time | Very good ← default |
| `large-v3` | ~0.8× real-time | Best |

---

## Project Structure

```
localscribe/
├── main.py                  # Entry point — boots all subsystems, tkinter loop
├── session.py               # SessionController — full recording lifecycle
├── app_state.py             # Shared thread-safe state (chunks, slides, queues)
├── config.py                # All settings in one place
├── hotkeys.py               # F9 pause/resume, F10 slide capture
├── training_noter.py        # Original single-file prototype (kept for reference)
│
├── audio/
│   ├── capture.py           # WASAPI loopback audio stream
│   └── transcriber.py       # faster-whisper transcription worker
│
├── ocr/
│   ├── screen_capture.py    # Screenshot capture + perceptual hash change detection
│   └── ocr_engine.py        # Tesseract / Ollama vision OCR
│
├── llm/
│   ├── ollama_client.py     # Ollama HTTP client wrapper
│   ├── note_generator.py    # Single-pass and chunked-merge note generation
│   ├── anki_generator.py    # Q&A pair generation for Anki
│   ├── qa_chat.py           # Post-session Q&A chat session
│   └── prompts.py           # All LLM prompt templates
│
├── export/
│   ├── obsidian_exporter.py # Writes notes + transcript as Obsidian Markdown
│   ├── anki_exporter.py     # Writes Anki-importable tab-separated CSV
│   └── course_aggregator.py # Maintains a cross-session course index
│
├── browser/
│   └── window_detector.py   # Polls active window title to detect course/domain
│
├── ui/
│   ├── tray.py              # pystray system tray icon and menu
│   ├── session_dialog.py    # Session start dialog (title, domain, device)
│   ├── chat_window.py       # Post-session Q&A chat UI
│   └── icon_factory.py      # Generates tray icon images per state
│
├── run.bat                  # One-click setup and launcher
└── requirements.txt         # Python dependencies
```

---

## Troubleshooting

**"No loopback device found"**  
→ Enable Stereo Mix (see setup above) or install VB-Cable

**"Cannot connect to Ollama"**  
→ Run `ollama serve` in a separate terminal, then re-launch

**Transcription is empty or gibberish**  
→ Check audio device is capturing system audio, not the microphone  
→ Try switching `WHISPER_DEVICE = "cpu"` in `config.py` to test

**GPU out of memory**  
→ Change `WHISPER_MODEL_SIZE = "small"` in `config.py`  
→ Or run `set OLLAMA_MAX_LOADED_MODELS=1` before launching Ollama

**F10 screenshot is blank or wrong window**  
→ The screenshot captures the primary monitor — make sure your video is on the primary display  
→ Screenshots are saved in `~/TrainingNotes/<date>/<course>/slides/` for manual inspection

**Notes don't include slide images**  
→ Confirm `.png` files exist in the `slides/` folder  
→ Open the notes `.md` file in a Markdown viewer (Obsidian, VS Code) — plain text editors won't render images

---

## Privacy

All processing is 100% local:
- Whisper runs on your GPU via faster-whisper
- LLM runs via Ollama on your machine
- Screenshots never leave your disk
- No audio, text, or images are sent to any external server
