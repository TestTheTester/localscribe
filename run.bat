@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo   Training Video Note-Taker — Setup ^& Launcher
echo ============================================================
echo.

REM ── Check Python 3.11 specifically ───────────────────────────
REM PyTorch CUDA builds require Python 3.11. 3.12/3.13 are NOT supported.
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 not found.
    echo.
    echo  PyTorch CUDA requires Python 3.11 specifically.
    echo  Python 3.12 and 3.13 do NOT have CUDA-compatible PyTorch builds.
    echo.
    echo  Download Python 3.11.9 from:
    echo  https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
    echo.
    echo  During install: check "Add Python to PATH" and use "Customize installation"
    pause & exit /b 1
)
for /f "tokens=*" %%i in ('py -3.11 --version 2^>^&1') do echo [OK] %%i

REM ── Check Ollama ───────────────────────────────────────────────
ollama --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama not found. Download from https://ollama.com/
    pause & exit /b 1
)
echo [OK] Ollama found

REM ── Create venv using Python 3.11 explicitly ──────────────────
set VENV_DIR=%~dp0venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo.
    echo [SETUP] Creating virtual environment with Python 3.11...
    py -3.11 -m venv "%VENV_DIR%"
    if errorlevel 1 ( echo [ERROR] Failed to create venv & pause & exit /b 1 )

    call "%VENV_DIR%\Scripts\activate.bat"

    echo [SETUP] Upgrading pip...
    python -m pip install --upgrade pip --quiet

    echo [SETUP] Installing PyTorch with CUDA 12.1 for RTX 4050...
    echo         ^(~2.5 GB download — this will take several minutes^)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo.
        echo [WARN] PyTorch CUDA download failed — likely a network/firewall issue.
        echo.
        echo  Option A: Try on a different network or mobile hotspot, then re-run.
        echo.
        echo  Option B: Download wheels manually in your browser:
        echo    torch:
        echo    https://download.pytorch.org/whl/cu121/torch-2.3.0+cu121-cp311-cp311-win_amd64.whl
        echo    torchaudio:
        echo    https://download.pytorch.org/whl/cu121/torchaudio-2.3.0+cu121-cp311-cp311-win_amd64.whl
        echo  Then run:
        echo    pip install C:\Users\%USERNAME%\Downloads\torch-2.3.0+cu121-cp311-cp311-win_amd64.whl
        echo    pip install C:\Users\%USERNAME%\Downloads\torchaudio-2.3.0+cu121-cp311-cp311-win_amd64.whl
        echo.
        pause & exit /b 1
    )

    echo [SETUP] Installing remaining dependencies...
    pip install -r "%~dp0requirements.txt" --quiet
    if errorlevel 1 ( echo [ERROR] Failed to install dependencies & pause & exit /b 1 )

    echo.
    echo [OK] All dependencies installed successfully.
    echo [CHECK] Verifying CUDA...
    python -c "import torch; print('[OK] Torch', torch.__version__); print('[OK] CUDA:', torch.cuda.is_available())"
) else (
    call "%VENV_DIR%\Scripts\activate.bat"
    echo [SETUP] Syncing any new dependencies...
    pip install -r "%~dp0requirements.txt" --quiet
)

REM ── Pull Ollama model if not present ──────────────────────────
echo.
echo [CHECK] Checking for gemma3:12b model...
ollama list | findstr "gemma3:12b" >nul 2>&1
if errorlevel 1 (
    echo [SETUP] Pulling gemma3:12b — ~8GB download, please wait...
    echo         You can open a separate terminal and run your video meanwhile.
    ollama pull gemma3:12b
    if errorlevel 1 (
        echo [WARN] gemma3:12b pull failed. Trying llama3.1:8b as fallback...
        ollama pull llama3.1:8b
    )
) else (
    echo [OK] gemma3:12b already available
)

REM ── Start Ollama serve in background if not running ───────────
tasklist | findstr "ollama.exe" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Starting Ollama server in background...
    start /min "" ollama serve
    timeout /t 3 /nobreak >nul
)

REM ── Audio device reminder ─────────────────────────────────────
echo.
echo ------------------------------------------------------------
echo  AUDIO SETUP
echo  The script auto-detects system audio. If it fails to find
echo  a device, enable one of these:
echo.
echo  Option A — Stereo Mix (built-in):
echo    Press Win+R, type: mmsys.cpl
echo    Go to Recording tab ^> right-click empty area
echo    ^> Show Disabled Devices ^> Enable "Stereo Mix"
echo.
echo  Option B — VB-Cable (more reliable, free):
echo    Download: https://vb-audio.com/Cable/
echo    Install as Administrator, then reboot.
echo    Set browser audio output via: ms-settings:apps-volume
echo    Change browser Output to "CABLE Input"
echo    To hear audio while recording: open mmsys.cpl ^> Recording
echo    ^> CABLE Output Properties ^> Listen tab ^> enable Listen
echo ------------------------------------------------------------
echo.

REM ── Launch the note-taker ─────────────────────────────────────
echo [LAUNCH] Starting Training Note-Taker (multi-file edition)...
echo.
python "%~dp0main.py"

echo.
echo [DONE] Session complete. Press any key to exit.
pause >nul