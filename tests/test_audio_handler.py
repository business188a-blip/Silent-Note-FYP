"""
tests/test_audio_handler.py
────────────────────────────────────────────────────────────────────────────
Unit tests for modules/audio_handler.py

Tests that do NOT require a physical microphone:
  • load_audio_file() with a programmatically generated WAV
  • Normalisation to 16 kHz mono
  • Rejection of unsupported file formats
  • elapsed_seconds() when not recording
  • is_recording() state transitions

Tests that require hardware (skipped in CI):
  • list_devices() returns a list
────────────────────────────────────────────────────────────────────────────
"""

import os
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.audio_handler import AudioHandler


def _write_wav(path: str, sample_rate: int = 44100,
               channels: int = 2, duration_sec: float = 0.5) -> None:
    """Write a minimal silent WAV file to the given path."""
    n_frames = int(sample_rate * duration_sec)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * n_frames)


class TestAudioHandlerLoadFile(unittest.TestCase):

    def setUp(self):
        self.handler = AudioHandler()
        self.tmp_files = []

    def tearDown(self):
        for f in self.tmp_files:
            if os.path.exists(f):
                os.unlink(f)

    def _make_wav(self, sample_rate=44100, channels=2) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        _write_wav(path, sample_rate=sample_rate, channels=channels)
        self.tmp_files.append(path)
        return path

    def test_load_wav_returns_string_path(self):
        src    = self._make_wav()
        result = AudioHandler.load_audio_file(src)
        self.assertIsInstance(result, str)
        self.tmp_files.append(result)

    def test_load_wav_output_exists(self):
        src    = self._make_wav()
        result = AudioHandler.load_audio_file(src)
        self.assertTrue(os.path.exists(result))
        self.tmp_files.append(result)

    def test_load_wav_output_is_16khz_mono(self):
        src    = self._make_wav(sample_rate=44100, channels=2)
        result = AudioHandler.load_audio_file(src)
        self.tmp_files.append(result)
        with wave.open(result, "rb") as wf:
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnchannels(), 1)

    def test_load_wav_already_16khz_mono(self):
        """A file already at 16 kHz mono should still be processed without error."""
        src    = self._make_wav(sample_rate=16000, channels=1)
        result = AudioHandler.load_audio_file(src)
        self.tmp_files.append(result)
        with wave.open(result, "rb") as wf:
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getnchannels(), 1)

    def test_unsupported_format_raises(self):
        fd, path = tempfile.mkstemp(suffix=".xyz")
        os.close(fd)
        self.tmp_files.append(path)
        with self.assertRaises(ValueError):
            AudioHandler.load_audio_file(path)

    def test_output_saved_to_recordings_dir(self):
        src    = self._make_wav()
        result = AudioHandler.load_audio_file(src)
        self.tmp_files.append(result)
        self.assertIn("recordings", result)


class TestAudioHandlerState(unittest.TestCase):

    def setUp(self):
        self.handler = AudioHandler()

    def test_not_recording_initially(self):
        self.assertFalse(self.handler.is_recording())

    def test_elapsed_zero_when_not_started(self):
        self.assertAlmostEqual(self.handler.elapsed_seconds(), 0.0, places=1)

    def test_list_devices_returns_list(self):
        devices = self.handler.list_devices()
        self.assertIsInstance(devices, list)
        # Each device dict must have 'index' and 'name'
        for d in devices:
            self.assertIn("index", d)
            self.assertIn("name",  d)

    def test_stop_when_not_recording_returns_none(self):
        result = self.handler.stop_recording()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
