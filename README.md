# 🎓 Training Video Note-Taker

Watches training videos playing in your browser, transcribes audio in real-time using Whisper (GPU-accelerated), and generates structured notes + summary via Ollama (local LLM). Everything runs locally — no data leaves your machine.

---

## Requirements

| Component | Minimum | Your Setup |
|---|---|---|
| Python | 3.10+ | — |
| GPU | Optional (CPU works) | RTX 4050 6GB ✓ |
| RAM | 8 GB | 24 GB ✓ |
| Ollama | Latest | — |
| Disk space | ~12 GB (model + torch) | — |

---

## One-Time Setup

### 1. Install Prerequisites

- **Python 3.10+**: https://www.python.org/downloads/  
  ✅ Check "Add Python to PATH" during install

- **Ollama**: https://ollama.com/download  
  Just install and it's ready.

### 2. Enable System Audio Capture (WASAPI Loopback)

The script auto-detects loopback devices. To enable one:

**Option A — Stereo Mix (built-in, free):**
1. Right-click the speaker icon in the taskbar → **Sounds**
2. Go to the **Recording** tab
3. Right-click in empty space → **Show Disabled Devices**
4. Right-click **Stereo Mix** → **Enable**

**Option B — VB-Cable (more reliable):**
1. Download from https://vb-audio.com/Cable/ (free)
2. Install and restart
3. Set your browser's audio output to "CABLE Input"
4. The script will auto-detect "CABLE Output"

### 3. Run Setup & Launch

Double-click `run.bat`

On first run it will:
- Create a Python virtual environment
- Install PyTorch with CUDA support (for your RTX 4050)
- Install faster-whisper, sounddevice, ollama
- Pull `gemma3:12b` from Ollama (~8 GB download)

This takes 10-20 minutes on first run. Subsequent launches start in seconds.

---

## Usage

1. Double-click `run.bat`
2. Enter a title for the video when prompted
3. Switch to your browser and start playing the training video
4. Press **Enter** in the terminal when the video ends (or when you want to stop)
5. Wait ~1-2 minutes for note generation
6. Notes are saved to `C:\Users\<you>\TrainingNotes\`

---

## Output Files

For each session, two files are created in `~/TrainingNotes/`:

| File | Contents |
|---|---|
| `YYYY-MM-DD_HH-MM_<title>_notes.md` | Summary + structured notes |
| `YYYY-MM-DD_HH-MM_<title>_transcript.md` | Full raw transcript |

**Notes structure:**
- **Summary** — 3-5 sentence overview
- **Key Concepts** — core ideas as bullets
- **Important Details** — technical specifics, definitions, examples
- **Action Items / Things to Remember** — practical takeaways
- **Questions to Explore** — gaps worth researching

---

## Configuration

Edit the top of `training_noter.py` to change settings:

```python
CHUNK_SECONDS      = 30             # How often to transcribe (seconds)
WHISPER_MODEL_SIZE = "medium"       # tiny / base / small / medium / large-v3
OLLAMA_MODEL       = "gemma3:12b"   # Change to llama3.1:8b for faster/lighter
```

**Whisper model tradeoffs:**

| Model | Speed (RTX 4050) | Accuracy |
|---|---|---|
| `tiny` | Very fast | Basic |
| `small` | Fast | Good |
| `medium` | Moderate | Very good ← default |
| `large-v3` | Slow | Best |

---

## Troubleshooting

**"No loopback device found"**  
→ Enable Stereo Mix (see setup above) or install VB-Cable

**"Cannot connect to Ollama"**  
→ Run `ollama serve` in a separate terminal

**Transcription is empty / gibberish**  
→ Check your audio device is capturing system audio, not microphone  
→ In run.bat, change `WHISPER_DEVICE` to `"cpu"` temporarily to test

**GPU out of memory**  
→ Change `WHISPER_MODEL_SIZE` to `"small"` in the script  
→ Or run Ollama with `OLLAMA_MAX_LOADED_MODELS=1`

---

## Privacy

All processing is 100% local:
- Whisper runs on your GPU via faster-whisper
- LLM runs via Ollama on your machine
- No audio or text is sent to any external server
