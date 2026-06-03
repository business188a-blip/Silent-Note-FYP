"""Tests for review-tab behavior that does not require real NLP models."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from PyQt5.QtWidgets import QApplication
except Exception as exc:  # pragma: no cover - import-time environment guard
    QApplication = None
    PYQT_IMPORT_ERROR = exc
else:
    PYQT_IMPORT_ERROR = None


@unittest.skipIf(QApplication is None, f"PyQt5 is not available: {PYQT_IMPORT_ERROR}")
class TestSummaryTabLanguage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_rerun_uses_saved_meeting_language(self):
        from gui import summary_tab as mod

        fake_db = Mock()
        fake_db.get_meeting.return_value = {
            "title": "Urdu Meeting",
            "transcript": "Team ko report bhejni hai.",
            "summary": "",
            "decisions": [],
            "language": "ur",
        }
        fake_db.get_action_items.return_value = []

        summarizer = Mock()
        summarizer.summarize.return_value = {"summary": "short summary"}

        extractor = Mock()
        extractor.extract.return_value = {"action_items": [], "decisions": []}

        with patch.object(mod, "DBManager", return_value=fake_db):
            tab = mod.SummaryTab()

        tab.load_meeting(10)

        with patch.object(mod, "Summarizer", return_value=summarizer), \
             patch.object(mod, "ActionExtractor", return_value=extractor):
            tab._on_rerun()

        summarizer.summarize.assert_called_once_with(
            "Team ko report bhejni hai.",
            ratio=0.25,
            language="ur",
        )
        extractor.extract.assert_called_once_with(
            "Team ko report bhejni hai.",
            language="ur",
        )

