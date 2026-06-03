"""
modules/snapshot_handler.py
──────────────────────────────────────────────────────────────────────────────
Manages screen snapshot capture and image storage for SilentNote.

Captures a full-screen screenshot using pyautogui, saves it to disk as a
PNG, and returns the file path for storage in the database.

Design notes
────────────
  • pyautogui.screenshot() captures the full screen without needing a webcam.
  • Images are stored under data/snapshots/ with timestamped filenames.
  • A thumbnail is also generated for fast display in the dashboard.
  • Output folder is created automatically if it does not exist.
──────────────────────────────────────────────────────────────────────────────
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pyautogui
from PIL import Image


# ── Exceptions ────────────────────────────────────────────────────────────────

class CameraNotFoundError(Exception):
    """Raised when a capture device or screen cannot be accessed."""
    pass


# ── Paths ─────────────────────────────────────────────────────────────────────

SNAPSHOTS_DIR  = Path(__file__).resolve().parent.parent / "data" / "snapshots"
THUMBNAIL_SIZE = (240, 135)    # 16:9 thumbnail for dashboard cards


# ── SnapshotHandler ───────────────────────────────────────────────────────────

class SnapshotHandler:
    """
    Captures and manages meeting screen snapshots.

    Usage
    ─────
      handler = SnapshotHandler()
      result  = handler.capture(meeting_id=3)
      print(result["image_path"])     # full-res PNG path
      print(result["thumb_path"])     # thumbnail PNG path
    """

    def __init__(self):
        SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def capture(self, meeting_id: int, note: str = "") -> dict:
        """
        Capture a full-screen screenshot and save it.

        Parameters
        ──────────
        meeting_id : int
            Used to name the file, linking it to the meeting.
        note : str
            Optional text note attached to this snapshot.

        Returns
        ───────
        dict with keys:
          image_path  – str, absolute path to full-resolution PNG
          thumb_path  – str, absolute path to thumbnail PNG
          timestamp   – str, ISO-8601 datetime of capture
          note        – str, the note passed in
        """
        try:
            screenshot = pyautogui.screenshot()
        except Exception as e:
            raise CameraNotFoundError(f"Failed to capture screen: {e}") from e

        timestamp = datetime.now()
        ts_str    = timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"meeting{meeting_id}_{ts_str}"

        image_path = self._save_image(screenshot, base_name)
        thumb_path = self._save_thumbnail(screenshot, base_name)

        return {
            "image_path": image_path,
            "thumb_path": thumb_path,
            "timestamp":  timestamp.isoformat(),
            "note":       note,
        }

    @staticmethod
    def delete_snapshot(image_path: str) -> None:
        """Delete the full-res image and its thumbnail from disk."""
        p = Path(image_path)
        if p.exists():
            p.unlink()

        thumb = p.parent / (p.stem + "_thumb" + p.suffix)
        if thumb.exists():
            thumb.unlink()

    @staticmethod
    def load_for_display(image_path: str,
                         max_width:  int = 800,
                         max_height: int = 600) -> Optional[Image.Image]:
        """
        Load an image from disk and scale it to fit within max_width x max_height.
        Returns a PIL Image, or None if the file does not exist.
        """
        p = Path(image_path)
        if not p.exists():
            return None

        img = Image.open(str(p))
        img.thumbnail((max_width, max_height), Image.LANCZOS)
        return img

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _save_image(screenshot: Image.Image, base_name: str) -> str:
        """Save full-resolution screenshot as PNG."""
        path = SNAPSHOTS_DIR / f"{base_name}.png"
        screenshot.save(str(path), "PNG")
        return str(path)

    @staticmethod
    def _save_thumbnail(screenshot: Image.Image, base_name: str) -> str:
        """Generate and save a small thumbnail."""
        thumb = screenshot.copy()
        thumb.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        thumb_path = SNAPSHOTS_DIR / f"{base_name}_thumb.png"
        thumb.save(str(thumb_path), "PNG")
        return str(thumb_path)