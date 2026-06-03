"""
modules/audio_handler.py
──────────────────────────────────────────────────────────────────────────────
Handles all audio input for SilentNote:
  • Real-time microphone recording via sounddevice
  • Saving recorded audio to WAV files
  • Loading pre-recorded audio files (WAV, MP3, M4A, OGG)
  • Emitting live audio level readings for a VU-meter widget in the GUI
  • DeepFilterNet noise cancellation applied after recording and file upload

Design notes
────────────
  • Uses sounddevice instead of PyAudio because sounddevice gives cleaner
    NumPy arrays and does not require manually managing a Pa_Stream object.
  • Recording runs in a background thread so the Qt main thread stays
    responsive. The caller receives progress via a callback.
  • All saved audio is stored under data/recordings/ in WAV format.
    If the user uploads an MP3/M4A/OGG file, pydub converts it to WAV first
    so Whisper always gets a clean WAV input.
  • DeepFilterNet is initialised once as a singleton to avoid reloading
    the model on every call.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import time
import threading
import wave
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment


# ── Constants ─────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16000
CHANNELS       = 1
DTYPE          = "int16"
CHUNK_FRAMES   = 1024
RECORDINGS_DIR = Path(__file__).resolve().parent.parent / "data" / "recordings"


# ── DeepFilterNet singleton ───────────────────────────────────────────────────

_df_model  = None
_df_state  = None
_df_lock   = threading.Lock()


def _get_df():
    """Lazy-load DeepFilterNet model once and reuse."""
    global _df_model, _df_state
    if _df_model is None:
        with _df_lock:
            if _df_model is None:
                from df.enhance import init_df
                _df_model, _df_state, _ = init_df()
    return _df_model, _df_state


def _denoise_wav(wav_path: str) -> str:
    """
    Run DeepFilterNet on a WAV file.
    Overwrites the file in-place with the cleaned audio.
    Returns the same path.
    """
    try:
        from df.enhance import enhance
        import torch

        model, df_state = _get_df()

        # Read audio
        audio, sr = sf.read(wav_path, always_2d=True)
        audio = audio.T  # (channels, samples)

        # DeepFilterNet expects float32 tensor (1, samples) at its own sample rate
        audio_tensor = torch.from_numpy(audio.astype(np.float32))

        enhanced = enhance(model, df_state, audio_tensor)

        # Write back
        sf.write(wav_path, enhanced.numpy().T, sr)

    except Exception as e:
        # Never crash the app over noise cancellation
        print(f"[DeepFilterNet] Skipped: {e}")

    return wav_path


# ── AudioHandler ──────────────────────────────────────────────────────────────

class AudioHandler:
    """
    Manages one recording session at a time.

    Typical lifecycle
    ─────────────────
      handler = AudioHandler()
      handler.start_recording(on_level=my_vu_callback)
      ...
      wav_path = handler.stop_recording()
      # wav_path is now ready to hand to Transcriber
    """

    def __init__(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

        self._frames:     list[np.ndarray] = []
        self._recording:  bool             = False
        self._stream:     Optional[sd.InputStream] = None
        self._start_time: Optional[float]  = None
        self._lock        = threading.Lock()

        self._level_cb: Optional[Callable[[float], None]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def list_devices(self) -> list[dict]:
        devices = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                devices.append({"index": i, "name": dev["name"]})
        return devices

    def start_recording(self,
                        device_index: Optional[int] = None,
                        on_level: Optional[Callable[[float], None]] = None
                        ) -> None:
        if self._recording:
            return

        self._frames     = []
        self._level_cb   = on_level
        self._recording  = True
        self._start_time = time.time()

        self._stream = sd.InputStream(
            samplerate = SAMPLE_RATE,
            channels   = CHANNELS,
            dtype      = DTYPE,
            blocksize  = CHUNK_FRAMES,
            device     = device_index,
            callback   = self._audio_callback,
        )
        self._stream.start()

    def stop_recording(self) -> Optional[str]:
        """
        Stop the current recording, save to WAV, apply DeepFilterNet.
        Returns absolute path to the cleaned WAV file.
        """
        if not self._recording:
            return None

        self._recording = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            frames = list(self._frames)
            self._frames = []

        if not frames:
            return None

        audio_data = np.concatenate(frames, axis=0)
        wav_path   = self._save_wav(audio_data)

        # Apply noise cancellation
        wav_path = _denoise_wav(wav_path)

        return wav_path

    def pause_recording(self) -> None:
        if self._stream and self._recording:
            self._stream.stop()

    def resume_recording(self) -> None:
        if self._stream and self._recording:
            self._stream.start()

    def is_recording(self) -> bool:
        return self._recording

    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    # ── File upload / conversion ──────────────────────────────────────────────

    @staticmethod
    def load_audio_file(source_path: str) -> str:
        """
        Accept a user-supplied audio file, convert to 16 kHz mono WAV,
        then apply DeepFilterNet noise cancellation.
        Returns path to the cleaned WAV file.
        """
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        source = Path(source_path)
        ext    = source.suffix.lower()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name  = f"upload_{timestamp}.wav"
        out_path  = RECORDINGS_DIR / out_name

        if ext == ".wav":
            audio = AudioSegment.from_wav(str(source))
        elif ext == ".mp3":
            audio = AudioSegment.from_mp3(str(source))
        elif ext in (".m4a", ".aac"):
            audio = AudioSegment.from_file(str(source), format="m4a")
        elif ext == ".ogg":
            audio = AudioSegment.from_ogg(str(source))
        elif ext == ".flac":
            audio = AudioSegment.from_file(str(source), format="flac")
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

        # Normalise to 16 kHz mono
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        audio.export(str(out_path), format="wav")

        # Apply noise cancellation
        _denoise_wav(str(out_path))

        return str(out_path)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _audio_callback(self,
                        indata:    np.ndarray,
                        frames:    int,
                        time_info,
                        status:    sd.CallbackFlags) -> None:
        if not self._recording:
            return

        chunk = indata.copy()
        with self._lock:
            self._frames.append(chunk)

        if self._level_cb is not None:
            rms   = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            level = min(rms / 32768.0 * 10, 1.0)
            self._level_cb(level)

    def _save_wav(self, audio_data: np.ndarray) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = RECORDINGS_DIR / f"meeting_{timestamp}.wav"

        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

        return str(filename)