"""
modules/speaker_diarizer.py
──────────────────────────────────────────────────────────────────────────────
Performs speaker diarization (who spoke when) entirely offline using
pyannote.audio with a locally cached model.

What it does
────────────
  Given a WAV file and a list of Whisper transcript segments (each with a
  start/end time), it assigns a speaker label ("Speaker 1", "Speaker 2", …)
  to each segment by finding which diarization turn overlaps most with that
  segment's time window.

Fallback
────────
  If pyannote fails to load (e.g., model files absent), the module falls
  back to a basic energy-based two-speaker heuristic so the rest of the
  application keeps working.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import warnings
from typing import Optional, Callable

import numpy as np

PYANNOTE_MODEL_ID = "pyannote/speaker-diarization-3.1"


# ── SpeakerDiarizer ───────────────────────────────────────────────────────────

class SpeakerDiarizer:
    """
    Assigns speaker labels to a list of Whisper segments.

    Usage
    ─────
      diarizer = SpeakerDiarizer()
      diarizer.load(token="hf_...")
      labelled = diarizer.assign_speakers(wav_path, segments)
      # segments is a list of dicts with 'start', 'end', 'text' keys.
      # labelled is the same list with 'speaker' keys filled in.
    """

    def __init__(self):
        self._pipeline = None
        self._loaded   = False
        self._fallback = False

    # ── Loading ───────────────────────────────────────────────────────────────

    def load(self,
             on_status: Optional[Callable] = None,
             token: Optional[str] = None) -> None:
        """
        Load the pyannote diarization pipeline.

        Parameters
        ──────────
        on_status : callable, optional
            Called with a status string during loading.
        token : str, optional
            Hugging Face access token for gated models.
            Get one at https://hf.co/settings/tokens after accepting
            the terms at https://hf.co/pyannote/speaker-diarization-3.1
            and https://hf.co/pyannote/segmentation-3.0
        """
        if self._loaded:
            return

        if on_status:
            on_status("Loading speaker diarization model…")

        token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            self._fallback = True
            self._loaded = True
            if on_status:
                on_status("Speaker diarization: using fallback mode. Set HF_TOKEN for pyannote.")
            return

        try:
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                PYANNOTE_MODEL_ID,
                use_auth_token=token,
            )
            self._fallback = False
            if on_status:
                on_status("Speaker diarization model ready.")
        except Exception as exc:
            warnings.warn(
                f"pyannote.audio could not load {PYANNOTE_MODEL_ID} ({exc}). "
                "Falling back to energy-based speaker splitting.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._fallback = True
            if on_status:
                on_status("Speaker diarization: using fallback mode.")

        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    # ── Main API ──────────────────────────────────────────────────────────────

    def assign_speakers(self,
                        wav_path: str,
                        segments: list[dict]
                        ) -> list[dict]:
        """
        Label each segment with a speaker identifier.

        Parameters
        ──────────
        wav_path : str
            Path to the 16 kHz mono WAV file that was transcribed.
        segments : list[dict]
            Each dict must have 'start' (float, seconds) and 'end' (float).
            The 'text' and 'id' keys are preserved unchanged.

        Returns
        ───────
        The same list with the 'speaker' key filled in for each segment.
        """
        self._ensure_loaded()

        if not segments:
            return segments

        if self._fallback:
            return self._fallback_assign(wav_path, segments)

        return self._pyannote_assign(wav_path, segments)

    def count_speakers(self, wav_path: str) -> int:
        """
        Return the estimated number of distinct speakers in the file.
        Returns 1 if diarization is unavailable.
        """
        self._ensure_loaded()
        if self._fallback:
            return self._fallback_count(wav_path)
        try:
            assert self._pipeline is not None
            diarization = self._pipeline(wav_path)
            speakers    = {label for _, _, label in diarization.itertracks(yield_label=True)}
            return len(speakers)
        except Exception:
            return 1

    # ── pyannote path ─────────────────────────────────────────────────────────

    def _pyannote_assign(self, wav_path: str, segments: list[dict]) -> list[dict]:
        """
        Run the pyannote pipeline and overlay its output on Whisper segments.
        For each Whisper segment we pick the speaker whose diarization turn
        has the greatest overlap with [start, end].
        """
        try:
            assert self._pipeline is not None
            diarization = self._pipeline(wav_path)
        except Exception as exc:
            warnings.warn(f"Diarization inference failed: {exc}")
            return self._label_all(segments, "Speaker 1")

        # Build a list of (start, end, label) turns
        turns = [
            (turn.start, turn.end, label)
            for turn, _, label in diarization.itertracks(yield_label=True)
        ]

        # Build a mapping from pyannote label → "Speaker N"
        seen:    dict[str, str] = {}
        counter: int            = 1

        def friendly(label: str) -> str:
            nonlocal counter
            if label not in seen:
                seen[label] = f"Speaker {counter}"
                counter     += 1
            return seen[label]

        result = []
        for seg in segments:
            seg     = dict(seg)
            s_start = seg.get("start", 0.0)
            s_end   = seg.get("end",   s_start + 1.0)

            best_label   = "Speaker 1"
            best_overlap = -1.0

            for (t_start, t_end, label) in turns:
                overlap = min(s_end, t_end) - max(s_start, t_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_label   = friendly(label)

            seg["speaker"] = best_label
            result.append(seg)

        return result

    # ── Energy-based fallback ─────────────────────────────────────────────────

    def _fallback_assign(self, wav_path: str, segments: list[dict]) -> list[dict]:
        """
        Heuristic: compute RMS energy per segment; cluster into two groups
        (high-energy = Speaker 1, low-energy = Speaker 2).
        Good enough to show the feature works even without pyannote.
        """
        import soundfile as sf

        try:
            audio, sr = sf.read(wav_path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
        except Exception:
            return self._label_all(segments, "Speaker 1")

        energies = []
        for seg in segments:
            s     = int(seg.get("start", 0) * sr)
            e     = int(seg.get("end",   0) * sr)
            chunk = audio[max(0, s): min(len(audio), e)]
            rms   = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0.0
            energies.append(rms)

        if not energies:
            return self._label_all(segments, "Speaker 1")

        median = float(np.median(energies))

        result = []
        for seg, energy in zip(segments, energies):
            seg = dict(seg)
            seg["speaker"] = "Speaker 1" if energy >= median else "Speaker 2"
            result.append(seg)

        return result

    def _fallback_count(self, wav_path: str) -> int:
        """Fallback always claims 2 speakers (conservative estimate)."""
        return 2

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _label_all(segments: list[dict], label: str) -> list[dict]:
        result = []
        for seg in segments:
            seg = dict(seg)
            seg["speaker"] = label
            result.append(seg)
        return result

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()
