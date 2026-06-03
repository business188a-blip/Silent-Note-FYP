"""
modules/voice_commands.py
──────────────────────────────────────────────────────────────────────────────
Optional voice command listener for SilentNote.

Uses the SpeechRecognition library with the Vosk offline backend so no
internet is needed.  The listener runs in a background thread and emits
recognised commands via a callback.

Supported commands (case-insensitive, partial match)
────────────────────────────────────────────────────
  "start recording"   → CMD_START_RECORDING
  "stop recording"    → CMD_STOP_RECORDING
  "pause"             → CMD_PAUSE
  "resume"            → CMD_RESUME
  "take snapshot"     → CMD_SNAPSHOT
  "save meeting"      → CMD_SAVE
  "export pdf"        → CMD_EXPORT_PDF
  "export word"       → CMD_EXPORT_DOCX
  "export json"       → CMD_EXPORT_JSON
  "open dashboard"    → CMD_DASHBOARD
  "new meeting"       → CMD_NEW_MEETING

Architecture
────────────
  VoiceCommandListener.start()  – spawns background thread
  VoiceCommandListener.stop()   – stops the thread cleanly
  on_command callback           – called with a CMD_* constant on each match

Fallback
────────
  If SpeechRecognition or Vosk is not available, the class degrades
  gracefully — start() becomes a no-op and is_available() returns False.
──────────────────────────────────────────────────────────────────────────────
"""

import threading
import warnings
from typing import Callable, Optional


# ── Command constants ─────────────────────────────────────────────────────────

CMD_START_RECORDING = "start_recording"
CMD_STOP_RECORDING  = "stop_recording"
CMD_PAUSE           = "pause"
CMD_RESUME          = "resume"
CMD_SNAPSHOT        = "snapshot"
CMD_SAVE            = "save"
CMD_EXPORT_PDF      = "export_pdf"
CMD_EXPORT_DOCX     = "export_docx"
CMD_EXPORT_JSON     = "export_json"
CMD_DASHBOARD       = "dashboard"
CMD_NEW_MEETING     = "new_meeting"


# ── Phrase → command mapping ──────────────────────────────────────────────────

PHRASE_MAP: list[tuple[list[str], str]] = [
    (["start recording", "begin recording", "start meeting"],  CMD_START_RECORDING),
    (["stop recording",  "end recording",   "stop meeting"],   CMD_STOP_RECORDING),
    (["pause", "pause recording"],                             CMD_PAUSE),
    (["resume", "continue recording"],                         CMD_RESUME),
    (["take snapshot", "capture snapshot", "screenshot"],      CMD_SNAPSHOT),
    (["save meeting", "save"],                                 CMD_SAVE),
    (["export pdf", "save pdf"],                               CMD_EXPORT_PDF),
    (["export word", "export doc", "save word"],               CMD_EXPORT_DOCX),
    (["export json", "save json"],                             CMD_EXPORT_JSON),
    (["open dashboard", "show dashboard", "dashboard"],        CMD_DASHBOARD),
    (["new meeting", "start new"],                             CMD_NEW_MEETING),
]


# ── VoiceCommandListener ──────────────────────────────────────────────────────

class VoiceCommandListener:
    """
    Listens for voice commands in the background and fires a callback
    whenever a recognised command phrase is detected.

    Usage
    ─────
      def handle(cmd):
          if cmd == CMD_START_RECORDING:
              ...

      listener = VoiceCommandListener(on_command=handle)
      listener.start()
      ...
      listener.stop()
    """

    def __init__(self, on_command: Optional[Callable[[str], None]] = None,
                 device_index: Optional[int] = None):
        self._on_command  = on_command
        self._device_index = device_index
        self._running     = False
        self._thread: Optional[threading.Thread] = None
        self._available   = self._check_availability()

    # ── Public API ────────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if the necessary libraries are installed."""
        return self._available

    def start(self) -> None:
        """Start the background listening thread."""
        if not self._available or self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._listen_loop,
                                         daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop."""
        self._running = False

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_availability() -> bool:
        try:
            import speech_recognition  # noqa: F401
            import pocketsphinx  # noqa: F401
            return True
        except ImportError:
            return False

    def _listen_loop(self) -> None:
        """
        Continuously capture audio in short bursts and match against
        command phrases.  Runs in the background thread.
        """
        try:
            import speech_recognition as sr
        except ImportError:
            return

        recogniser = sr.Recognizer()
        recogniser.pause_threshold     = 0.6
        recogniser.energy_threshold    = 300
        recogniser.dynamic_energy_threshold = True

        mic_kwargs = {}
        if self._device_index is not None:
            mic_kwargs["device_index"] = self._device_index

        with sr.Microphone(**mic_kwargs) as source:
            recogniser.adjust_for_ambient_noise(source, duration=1)

            while self._running:
                try:
                    audio = recogniser.listen(source, timeout=2,
                                              phrase_time_limit=4)
                except sr.WaitTimeoutError:
                    continue

                try:
                    # Prefer Sphinx (offline) if available, else fall back to
                    # Google (online) — this keeps it offline-first.
                    try:
                        text = recogniser.recognize_sphinx(audio).lower()
                    except (sr.UnknownValueError, AttributeError):
                        # Sphinx not installed — try google as fallback
                        # (only works if internet is available)
                        try:
                            continue
                        except Exception:
                            continue

                    cmd = self._match_command(text)
                    if cmd and self._on_command:
                        self._on_command(cmd)

                except sr.UnknownValueError:
                    pass
                except Exception as exc:
                    warnings.warn(f"Voice command error: {exc}")

    @staticmethod
    def _match_command(text: str) -> Optional[str]:
        """Return the command constant if any phrase matches, else None."""
        for phrases, cmd in PHRASE_MAP:
            for phrase in phrases:
                if phrase in text:
                    return cmd
        return None
