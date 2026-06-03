import tempfile
import unittest
from pathlib import Path

from modules.urdu_learning import clean_and_learn, learn_from_correction
from modules.urdu_memory import apply_corrections, load_corrections, save_correction


class TestUrduMemory(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "urdu_memory.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_correction_applies_after_threshold(self):
        save_correction("wrong", "right", db_path=self.db_path)
        self.assertEqual(
            apply_corrections("wrong text", db_path=self.db_path),
            "wrong text",
        )

        save_correction("wrong", "right", db_path=self.db_path)
        self.assertEqual(
            apply_corrections("wrong text", db_path=self.db_path),
            "right text",
        )

    def test_longer_phrases_apply_first(self):
        save_correction("bad", "ok", db_path=self.db_path)
        save_correction("bad", "ok", db_path=self.db_path)
        save_correction("bad phrase", "good phrase", db_path=self.db_path)
        save_correction("bad phrase", "good phrase", db_path=self.db_path)

        self.assertEqual(
            apply_corrections("bad phrase", db_path=self.db_path),
            "good phrase",
        )


class TestUrduLearning(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "urdu_memory.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_learns_replacements_from_user_correction(self):
        saved = learn_from_correction(
            "ap meeting start karo",
            "aap meeting start karo",
            db_path=self.db_path,
        )

        self.assertEqual(saved, 1)
        self.assertEqual(load_corrections(db_path=self.db_path)[0][:2], ("ap", "aap"))

    def test_clean_and_learn_applies_existing_memory(self):
        save_correction("ap", "aap", db_path=self.db_path)
        save_correction("ap", "aap", db_path=self.db_path)

        self.assertEqual(
            clean_and_learn("ap meeting", db_path=self.db_path),
            "aap meeting",
        )

    def test_learning_ignores_identical_text(self):
        saved = learn_from_correction(
            "same text",
            "same text",
            db_path=self.db_path,
        )

        self.assertEqual(saved, 0)
        self.assertEqual(load_corrections(db_path=self.db_path), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
