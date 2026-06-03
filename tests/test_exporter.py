"""
tests/test_exporter.py
────────────────────────────────────────────────────────────────────────────
Unit tests for modules/exporter.py

Tests verify that:
  • Each export function creates a file at the given path
  • The file is non-empty
  • JSON output is valid and contains expected keys
  • Edge cases (empty fields, missing optional data) do not crash the exporter
────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.exporter import Exporter


# ── Sample meeting data ───────────────────────────────────────────────────────

FULL_MEETING = {
    "title":         "Q4 Budget Review",
    "started_at":    "2026-01-15T09:00:00",
    "ended_at":      "2026-01-15T10:15:00",
    "duration_sec":  4500,
    "language":      "en",
    "speaker_count": 3,
    "emotion_label": "joy",
    "emotion_score": 0.72,
    "summary": (
        "The team reviewed the Q4 budget and agreed on a ten percent increase "
        "for the marketing department. Key performance metrics were discussed "
        "and the finance team will prepare a detailed report."
    ),
    "decisions": [
        "Approved ten percent marketing budget increase.",
        "Weekly check-ins will be held every Tuesday.",
    ],
    "transcript": (
        "Speaker 1: Good morning everyone. Let us begin with the budget review.\n"
        "Speaker 2: I have prepared the numbers. Marketing needs a ten percent increase.\n"
        "Speaker 1: That is approved. Alice, please prepare the report by Friday.\n"
        "Speaker 3: I will schedule the check-ins starting next week."
    ),
    "action_items": [
        {"text": "Alice will prepare the budget report by Friday.",
         "assignee": "Alice", "done": False},
        {"text": "Schedule weekly check-ins starting next Tuesday.",
         "assignee": "Bob",   "done": True},
    ],
    "snapshots": [
        {"image_path": "/data/snapshots/snap1.jpg",
         "captured_at": "2026-01-15T09:30:00",
         "note": "Whiteboard with figures"},
    ],
}

MINIMAL_MEETING = {
    "title": "Quick Sync",
}

EMPTY_FIELDS_MEETING = {
    "title":         "Empty Fields Test",
    "summary":       "",
    "transcript":    "",
    "action_items":  [],
    "decisions":     [],
    "snapshots":     [],
    "duration_sec":  0,
    "emotion_label": None,
    "emotion_score": None,
}


# ── Helper ────────────────────────────────────────────────────────────────────

def tmp_path(suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path


# ── PDF tests ─────────────────────────────────────────────────────────────────

class TestExportPDF(unittest.TestCase):

    def setUp(self):
        self.exporter = Exporter()

    def _cleanup(self, path):
        if os.path.exists(path):
            os.unlink(path)

    def test_full_meeting_creates_file(self):
        path = tmp_path(".pdf")
        try:
            result = self.exporter.export_pdf(FULL_MEETING, path)
            self.assertTrue(os.path.exists(result))
            self.assertGreater(os.path.getsize(result), 0)
        finally:
            self._cleanup(path)

    def test_minimal_meeting_does_not_crash(self):
        path = tmp_path(".pdf")
        try:
            result = self.exporter.export_pdf(MINIMAL_MEETING, path)
            self.assertTrue(os.path.exists(result))
        finally:
            self._cleanup(path)

    def test_empty_fields_meeting_does_not_crash(self):
        path = tmp_path(".pdf")
        try:
            result = self.exporter.export_pdf(EMPTY_FIELDS_MEETING, path)
            self.assertTrue(os.path.exists(result))
        finally:
            self._cleanup(path)

    def test_returns_correct_path(self):
        path = tmp_path(".pdf")
        try:
            result = self.exporter.export_pdf(FULL_MEETING, path)
            self.assertEqual(result, path)
        finally:
            self._cleanup(path)

    def test_pdf_header_bytes(self):
        """A valid PDF starts with %PDF."""
        path = tmp_path(".pdf")
        try:
            self.exporter.export_pdf(FULL_MEETING, path)
            with open(path, "rb") as f:
                header = f.read(4)
            self.assertEqual(header, b"%PDF")
        finally:
            self._cleanup(path)


# ── DOCX tests ────────────────────────────────────────────────────────────────

class TestExportDOCX(unittest.TestCase):

    def setUp(self):
        self.exporter = Exporter()

    def _cleanup(self, path):
        if os.path.exists(path):
            os.unlink(path)

    def test_full_meeting_creates_file(self):
        path = tmp_path(".docx")
        try:
            result = self.exporter.export_docx(FULL_MEETING, path)
            self.assertTrue(os.path.exists(result))
            self.assertGreater(os.path.getsize(result), 0)
        finally:
            self._cleanup(path)

    def test_minimal_meeting_does_not_crash(self):
        path = tmp_path(".docx")
        try:
            result = self.exporter.export_docx(MINIMAL_MEETING, path)
            self.assertTrue(os.path.exists(result))
        finally:
            self._cleanup(path)

    def test_empty_fields_does_not_crash(self):
        path = tmp_path(".docx")
        try:
            result = self.exporter.export_docx(EMPTY_FIELDS_MEETING, path)
            self.assertTrue(os.path.exists(result))
        finally:
            self._cleanup(path)

    def test_docx_is_zip(self):
        """DOCX files are ZIP archives — magic bytes PK\\x03\\x04."""
        path = tmp_path(".docx")
        try:
            self.exporter.export_docx(FULL_MEETING, path)
            with open(path, "rb") as f:
                magic = f.read(4)
            self.assertEqual(magic, b"PK\x03\x04")
        finally:
            self._cleanup(path)

    def test_docx_readable_by_python_docx(self):
        from docx import Document
        path = tmp_path(".docx")
        try:
            self.exporter.export_docx(FULL_MEETING, path)
            doc   = Document(path)
            texts = [p.text for p in doc.paragraphs]
            full  = " ".join(texts)
            self.assertIn("Q4 Budget Review", full)
        finally:
            self._cleanup(path)


# ── JSON tests ────────────────────────────────────────────────────────────────

class TestExportJSON(unittest.TestCase):

    def setUp(self):
        self.exporter = Exporter()

    def _cleanup(self, path):
        if os.path.exists(path):
            os.unlink(path)

    def test_full_meeting_creates_file(self):
        path = tmp_path(".json")
        try:
            result = self.exporter.export_json(FULL_MEETING, path)
            self.assertTrue(os.path.exists(result))
            self.assertGreater(os.path.getsize(result), 0)
        finally:
            self._cleanup(path)

    def test_json_is_valid(self):
        path = tmp_path(".json")
        try:
            self.exporter.export_json(FULL_MEETING, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
        finally:
            self._cleanup(path)

    def test_json_contains_required_keys(self):
        path = tmp_path(".json")
        try:
            self.exporter.export_json(FULL_MEETING, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key in ("title", "summary", "transcript",
                        "action_items", "decisions", "emotion"):
                self.assertIn(key, data, f"Missing key in JSON: {key}")
        finally:
            self._cleanup(path)

    def test_json_action_items_structure(self):
        path = tmp_path(".json")
        try:
            self.exporter.export_json(FULL_MEETING, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data["action_items"], list)
            for item in data["action_items"]:
                self.assertIn("text",     item)
                self.assertIn("assignee", item)
                self.assertIn("done",     item)
        finally:
            self._cleanup(path)

    def test_json_utf8_urdu_text(self):
        """Non-ASCII characters must survive the JSON round-trip."""
        meeting = {
            "title":    "Urdu Test",
            "summary":  "یہ ایک آزمائشی خلاصہ ہے۔",
            "transcript": "بیٹھک میں بجٹ کا جائزہ لیا گیا۔",
        }
        path = tmp_path(".json")
        try:
            self.exporter.export_json(meeting, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["summary"], "یہ ایک آزمائشی خلاصہ ہے۔")
        finally:
            self._cleanup(path)

    def test_minimal_meeting_does_not_crash(self):
        path = tmp_path(".json")
        try:
            result = self.exporter.export_json(MINIMAL_MEETING, path)
            self.assertTrue(os.path.exists(result))
        finally:
            self._cleanup(path)

    def test_exported_at_key_present(self):
        path = tmp_path(".json")
        try:
            self.exporter.export_json(FULL_MEETING, path)
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("exported_at", data)
        finally:
            self._cleanup(path)


# ── Helper method tests ───────────────────────────────────────────────────────

class TestExporterHelpers(unittest.TestCase):

    def test_fmt_dur_seconds_only(self):
        self.assertEqual(Exporter._fmt_dur(45), "45s")

    def test_fmt_dur_minutes_and_seconds(self):
        self.assertEqual(Exporter._fmt_dur(125), "2m 5s")

    def test_fmt_dur_hours(self):
        self.assertEqual(Exporter._fmt_dur(3661), "1h 1m")

    def test_fmt_dur_zero(self):
        self.assertEqual(Exporter._fmt_dur(0), "0s")

    def test_fmt_dur_none(self):
        self.assertEqual(Exporter._fmt_dur(None), "0s")

    def test_fmt_dt_valid(self):
        result = Exporter._fmt_dt("2026-01-15T09:00:00")
        self.assertIn("2026", result)
        self.assertIn("January", result)

    def test_fmt_dt_empty(self):
        self.assertEqual(Exporter._fmt_dt(""), "—")

    def test_fmt_dt_none(self):
        self.assertEqual(Exporter._fmt_dt(None), "—")


if __name__ == "__main__":
    unittest.main(verbosity=2)
