# SilentNote Setup Guide

SilentNote is an offline AI meeting minutes generator. It records or imports audio, transcribes it with Whisper, summarizes meetings, extracts action items, detects emotion, and exports PDF, DOCX, and JSON reports. Internet is only required for the initial dependency and model downloads.

Updated: 2026-05-16

## Requirements

| Requirement | Details |
| --- | --- |
| OS | Windows 10 or Windows 11 |
| Python | Python 3.10.x |
| RAM | 4 GB minimum, 8 GB recommended |
| Storage | 5 GB or more free for packages, models, and runtime data |
| Microphone | Required for live recording |
| Webcam | Optional, used for meeting snapshots |

## Step 1: Install Python 3.10

If Python 3.10 is not installed:

1. Open https://www.python.org/downloads/release/python-31011/
2. Download the Windows installer, 64-bit.
3. Run the installer and check "Add Python to PATH".

Verify the install:

```powershell
py -3.10 --version
```

Expected output:

```text
Python 3.10.x
```

## Step 2: Open PowerShell in the project folder

1. Open the `files` folder in File Explorer.
2. Click the address bar.
3. Type `powershell` and press Enter.

## Step 3: Create and activate a virtual environment

```powershell
py -3.10 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
```

You should see `(venv)` at the start of the terminal prompt. Re-run `venv\Scripts\activate` whenever you open a new terminal.

## Step 4: Install dependencies

Install everything from the updated requirements file:

```powershell
pip install -r requirements.txt
```

This includes the current CPU PyTorch audio stack, Whisper, spaCy with `en_core_web_sm`, PyQt5, DeepFilterNet, export libraries, and optional diarization or voice-command packages.

Notes:

- `requirements.txt` uses the PyTorch CPU wheel index for `torch==2.11.0+cpu` and `torchaudio==2.11.0+cpu`.
- PyPI currently lists `torch==2.12.0`, but the latest matching `torchaudio` CPU wheel is `2.11.0`, so SilentNote pins the matched audio pair.
- `DeepFilterNet==0.5.6` requires `packaging<24`, so the requirements pin `packaging==23.2`.

## Step 5: Optional model pre-downloads

SilentNote can download models automatically on first use, but pre-downloading them makes the first launch smoother.

### Whisper model

```powershell
python -c "import whisper; whisper.load_model('small', device='cpu')"
```

The model is cached under:

```text
C:\Users\<you>\.cache\whisper\
```

### Emotion detection model

The app uses:

```text
j-hartmann/emotion-english-distilroberta-base
```

If this model is not already cached, emotion detection falls back to offline keyword detection.

## Step 6: Optional pyannote speaker diarization

Without this step, SilentNote still runs and uses a local energy-based speaker split fallback.

1. Accept the model terms:
   - https://hf.co/pyannote/speaker-diarization-3.1
   - https://hf.co/pyannote/segmentation-3.0
2. Create a Hugging Face token at https://hf.co/settings/tokens
3. Set it in PowerShell before launching the app:

```powershell
$env:HF_TOKEN="hf_your_token_here"
python main.py
```

To save the token as a persistent Windows user variable:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "hf_your_token_here", "User")
```

## Step 7: Run the application

```powershell
python main.py
```

On first launch, Whisper may download the small model and other cached models may initialize. Wait for the status bar to say that models are ready before using the app.

## Daily launch

```powershell
venv\Scripts\activate
python main.py
```

You can also double-click `run.bat` from the project folder.

## Moving to another PC

1. Copy the entire `files` folder to the new PC.
2. Do not copy the `venv` folder.
3. Optionally copy model caches to avoid re-downloading:
   - `C:\Users\<OldUser>\.cache\whisper\`
   - `C:\Users\<OldUser>\.cache\huggingface\`
4. On the new PC, recreate the virtual environment and run `pip install -r requirements.txt`.
5. Run `python main.py`.

## Runtime files

```text
data/
  recordings/      WAV files saved from recording and upload sessions
  snapshots/       Webcam images captured during meetings
  exports/         PDF, DOCX, and JSON exports
  silentnote.db    Local SQLite database
  silentnote.key   Encryption key
```

Do not delete `data/silentnote.key`. If it is removed, encrypted database records become unreadable.

## Supported languages

| Language | Code | Notes |
| --- | --- | --- |
| English | `en` | Best accuracy |
| Urdu | `ur` | Select Urdu manually for best results |
| Punjabi | `pa` | Basic support |
| Auto-detect | `auto` | Useful for unknown audio, less reliable for Urdu |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `py -3.10` not found | Install Python 3.10 and check "Add Python to PATH". |
| PyTorch download is slow | Re-run `pip install -r requirements.txt`; pip resumes cached downloads when possible. |
| `PyAudio` install fails | Use Python 3.10 64-bit and then re-run `pip install -r requirements.txt`. |
| `spaCy model not found` | Re-run `pip install -r requirements.txt`; the model wheel is included in the requirements file. |
| App crashes on audio upload | Close other apps and retry; large audio files need extra RAM. |
| Urdu transcription looks wrong | Select `Urdu (ur)` instead of auto-detect. |
| `(venv)` is missing | Run `venv\Scripts\activate`. |
| `ModuleNotFoundError` on launch | Make sure the venv is active and dependencies installed successfully. |
| Whisper downloads on every launch | Do not delete `C:\Users\<you>\.cache\whisper\`. |
| Emotion detection uses fallback | Download/cache the Hugging Face emotion model before using offline mode. |
| `silentnote.key` was deleted | The old database is unreadable; restore the key from backup or start with a fresh database. |

## Project structure

```text
files/
  main.py              Entry point
  run.bat              Double-click launcher
  requirements.txt     Python dependencies
  SETUP.md             Setup guide
  database/            SQLite and encrypted storage
  modules/             Audio, transcription, NLP, diarization, export, and voice modules
  gui/                 PyQt5 user interface
  tests/               Test suite
```

## Running tests

```powershell
venv\Scripts\activate
python test_system.py
```

The older expected result for this project is `39/39 tests passing`.
