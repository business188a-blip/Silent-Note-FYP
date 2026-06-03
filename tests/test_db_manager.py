"""
tests/test_db_manager.py
────────────────────────────────────────────────────────────────────────────
Unit tests for database/db_manager.py

Runs against a temporary in-memory-like database (temp file) so the
production database is never touched.  Each test class creates its own
DBManager pointed at a fresh temp file and deletes it on teardown.
────────────────────────────────────────────────────────────────────────────
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import DBManager, encrypt, decrypt


class TestEncryption(unittest.TestCase):
    """encrypt() and decrypt() must be inverses of each other."""

    def test_roundtrip_english(self):
        plain = "This is a test transcript."
        self.assertEqual(decrypt(encrypt(plain)), plain)

    def test_roundtrip_urdu(self):
        plain = "یہ ایک آزمائشی متن ہے۔"
        self.assertEqual(decrypt(encrypt(plain)), plain)

    def test_empty_string(self):
        self.assertEqual(encrypt(""), "")
        self.assertEqual(decrypt(""), "")

    def test_ciphertext_differs_from_plaintext(self):
        plain = "hello"
        self.assertNotEqual(encrypt(plain), plain)


class TestDBManagerMeetings(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DBManager(db_path=Path(self.tmp.name))

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_create_and_get_meeting(self):
        mid = self.db.create_meeting(title="Board Meeting", language="en")
        self.assertIsInstance(mid, int)
        m = self.db.get_meeting(mid)
        self.assertIsNotNone(m)
        self.assertEqual(m["title"], "Board Meeting")

    def test_update_transcript_is_encrypted_at_rest(self):
        mid = self.db.create_meeting()
        self.db.update_meeting(mid, transcript="Secret discussion about budgets.")
        m = self.db.get_meeting(mid)
        self.assertEqual(m["transcript"], "Secret discussion about budgets.")

    def test_update_summary(self):
        mid = self.db.create_meeting()
        self.db.update_meeting(mid, summary="Short summary.")
        m = self.db.get_meeting(mid)
        self.assertEqual(m["summary"], "Short summary.")

    def test_get_all_meetings_order(self):
        self.db.create_meeting(title="First")
        self.db.create_meeting(title="Second")
        meetings = self.db.get_all_meetings()
        # Most recent first
        self.assertEqual(meetings[0]["title"], "Second")
        self.assertEqual(meetings[1]["title"], "First")

    def test_search_meetings(self):
        self.db.create_meeting(title="Alpha Team Sync")
        self.db.create_meeting(title="Beta Review")
        results = self.db.get_all_meetings(search="Alpha")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Alpha Team Sync")

    def test_delete_meeting(self):
        mid = self.db.create_meeting(title="Temp")
        self.db.delete_meeting(mid)
        self.assertIsNone(self.db.get_meeting(mid))

    def test_get_nonexistent_meeting(self):
        self.assertIsNone(self.db.get_meeting(99999))

    def test_update_emotion(self):
        mid = self.db.create_meeting()
        self.db.update_meeting(mid, emotion_label="joy", emotion_score=0.85)
        m = self.db.get_meeting(mid)
        self.assertEqual(m["emotion_label"], "joy")
        self.assertAlmostEqual(m["emotion_score"], 0.85, places=2)

    def test_decisions_and_language_roundtrip(self):
        mid = self.db.create_meeting(title="Urdu Review", language="ur")
        decisions = ["Approve the revised launch date", "Keep weekly check-ins"]
        self.db.update_meeting(mid, decisions=decisions)

        meeting = self.db.get_meeting(mid)

        self.assertEqual(meeting["language"], "ur")
        self.assertEqual(meeting["decisions"], decisions)


class TestDBManagerActionItems(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db  = DBManager(db_path=Path(self.tmp.name))
        self.mid = self.db.create_meeting(title="Action Test")

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_add_and_retrieve_action_items(self):
        items = ["Send report to Alice", "Schedule follow-up by Friday"]
        self.db.add_action_items(self.mid, items)
        retrieved = self.db.get_action_items(self.mid)
        texts = [r["item"] for r in retrieved]
        self.assertIn("Send report to Alice", texts)
        self.assertIn("Schedule follow-up by Friday", texts)

    def test_action_items_encrypted_at_rest(self):
        """Items should decrypt back to original text."""
        self.db.add_action_items(self.mid, ["Confidential action item"])
        retrieved = self.db.get_action_items(self.mid)
        self.assertEqual(retrieved[0]["item"], "Confidential action item")

    def test_toggle_done(self):
        self.db.add_action_items(self.mid, ["Toggle test item"])
        item = self.db.get_action_items(self.mid)[0]
        self.assertEqual(item["done"], 0)
        self.db.toggle_action_item(item["id"])
        item = self.db.get_action_items(self.mid)[0]
        self.assertEqual(item["done"], 1)
        self.db.toggle_action_item(item["id"])
        item = self.db.get_action_items(self.mid)[0]
        self.assertEqual(item["done"], 0)

    def test_delete_action_items(self):
        self.db.add_action_items(self.mid, ["Item A", "Item B"])
        self.db.delete_action_items(self.mid)
        self.assertEqual(len(self.db.get_action_items(self.mid)), 0)

    def test_empty_items_not_inserted(self):
        self.db.add_action_items(self.mid, ["", "  ", "Valid item"])
        items = self.db.get_action_items(self.mid)
        self.assertEqual(len(items), 1)

    def test_add_action_items_accepts_dicts(self):
        self.db.add_action_items(
            self.mid,
            [
                {
                    "text": "Send the revised deck",
                    "assignee": "Aisha",
                    "due_date": "2026-05-20",
                    "done": True,
                },
                {
                    "item": "Schedule the follow-up call",
                    "assignee": "Team",
                },
            ],
        )

        items = self.db.get_action_items(self.mid)

        self.assertEqual(items[0]["item"], "Send the revised deck")
        self.assertEqual(items[0]["assignee"], "Aisha")
        self.assertEqual(items[0]["due_date"], "2026-05-20")
        self.assertEqual(items[0]["done"], 1)
        self.assertEqual(items[1]["item"], "Schedule the follow-up call")
        self.assertEqual(items[1]["assignee"], "Team")


class TestDBManagerSnapshots(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db  = DBManager(db_path=Path(self.tmp.name))
        self.mid = self.db.create_meeting()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_add_and_get_snapshot(self):
        sid = self.db.add_snapshot(self.mid, "/data/snapshots/test.jpg", note="Whiteboard")
        snaps = self.db.get_snapshots(self.mid)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0]["image_path"], "/data/snapshots/test.jpg")
        self.assertEqual(snaps[0]["note"], "Whiteboard")

    def test_delete_snapshot(self):
        sid = self.db.add_snapshot(self.mid, "/data/snapshots/del.jpg")
        self.db.delete_snapshot(sid)
        self.assertEqual(len(self.db.get_snapshots(self.mid)), 0)

    def test_cascade_delete(self):
        self.db.add_snapshot(self.mid, "/data/snapshots/cascade.jpg")
        self.db.delete_meeting(self.mid)
        self.assertEqual(len(self.db.get_snapshots(self.mid)), 0)


class TestDBManagerStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DBManager(db_path=Path(self.tmp.name))

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_stats_empty_db(self):
        stats = self.db.get_stats()
        self.assertEqual(stats["total_meetings"], 0)
        self.assertEqual(stats["total_snapshots"], 0)

    def test_stats_counts(self):
        mid1 = self.db.create_meeting(title="M1")
        mid2 = self.db.create_meeting(title="M2")
        self.db.update_meeting(mid1, duration_sec=120, emotion_label="joy")
        self.db.update_meeting(mid2, duration_sec=180, emotion_label="joy")
        self.db.add_action_items(mid1, ["Task A", "Task B"])
        self.db.add_snapshot(mid1, "/path/snap.jpg")

        stats = self.db.get_stats()
        self.assertEqual(stats["total_meetings"], 2)
        self.assertEqual(stats["total_action_items"], 2)
        self.assertEqual(stats["total_snapshots"], 1)
        self.assertAlmostEqual(stats["total_duration_sec"], 300)
        self.assertAlmostEqual(stats["avg_duration_sec"], 150)
        self.assertIn("joy", stats["emotion_distribution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
