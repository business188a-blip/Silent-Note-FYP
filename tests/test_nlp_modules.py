"""
tests/test_nlp_modules.py
────────────────────────────────────────────────────────────────────────────
Unit tests for:
  • modules/summarizer.py        (Summarizer)
  • modules/action_extractor.py  (ActionExtractor)

These tests do NOT require audio input, Whisper, or internet access.
They operate entirely on text strings.
────────────────────────────────────────────────────────────────────────────
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.summarizer       import Summarizer
from modules.action_extractor import ActionExtractor


# ── Sample transcripts ────────────────────────────────────────────────────────

SHORT_TRANSCRIPT = (
    "We discussed the quarterly budget review. "
    "The finance team should prepare a detailed report by next Friday. "
    "Alice will send the updated projections to the board. "
    "We agreed to increase the marketing budget by ten percent. "
    "Bob must review the vendor contracts before the end of the month. "
    "The team decided to hold weekly check-ins going forward. "
    "Everyone needs to submit their department goals by Thursday."
)

LONG_TRANSCRIPT = (
    "Good morning everyone. Today's agenda covers three main topics. "
    "First, the product roadmap for the next quarter. "
    "Second, the hiring plan for the engineering team. "
    "Third, the status of the new office setup in Lahore. "
    "Regarding the product roadmap, we have decided to prioritise the mobile app. "
    "The design team will create mockups by the end of this week. "
    "Ahmad should coordinate with the backend team to align the API timelines. "
    "For hiring, we need to post three new job listings immediately. "
    "HR must shortlist candidates within two weeks. "
    "Usman will schedule the technical interviews for shortlisted candidates. "
    "The office in Lahore is almost ready. "
    "Arslan needs to arrange the furniture delivery by next Monday. "
    "The IT team should set up the network infrastructure before the move-in date. "
    "We confirmed that the move-in date is the fifteenth of next month. "
    "Please make sure all departments submit their equipment lists to procurement. "
    "The management decided to approve a budget of five hundred thousand rupees for the office. "
    "Any remaining items should be escalated to the project manager. "
    "We will meet again next Tuesday to review progress on all three topics."
)

URDU_LIKE_TRANSCRIPT = (
    "The meeting was held in Urdu. "
    "Participants discussed supply chain issues. "
    "The manager should prepare a report in Urdu for the regional office. "
    "All team members need to attend the next session."
)


# ── Summarizer tests ──────────────────────────────────────────────────────────

class TestSummarizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.s = Summarizer()
        cls.s.load()

    def test_returns_dict_with_required_keys(self):
        result = self.s.summarize(SHORT_TRANSCRIPT)
        for key in ("summary", "key_points", "word_count", "sent_count"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_summary_is_non_empty(self):
        result = self.s.summarize(SHORT_TRANSCRIPT)
        self.assertTrue(len(result["summary"]) > 0)

    def test_summary_shorter_than_original(self):
        result = self.s.summarize(LONG_TRANSCRIPT)
        self.assertLess(len(result["summary"]), len(LONG_TRANSCRIPT))

    def test_key_points_are_list(self):
        result = self.s.summarize(SHORT_TRANSCRIPT)
        self.assertIsInstance(result["key_points"], list)

    def test_key_points_max_15_words(self):
        result = self.s.summarize(LONG_TRANSCRIPT)
        for point in result["key_points"]:
            # Allow for the ellipsis character at the end
            cleaned = point.rstrip("…")
            words   = cleaned.split()
            self.assertLessEqual(len(words), 15,
                                 f"Key point too long: '{point}'")

    def test_word_count_is_positive(self):
        result = self.s.summarize(SHORT_TRANSCRIPT)
        self.assertGreater(result["word_count"], 0)

    def test_empty_transcript(self):
        result = self.s.summarize("")
        self.assertEqual(result["summary"], "")
        self.assertEqual(result["key_points"], [])
        self.assertEqual(result["word_count"], 0)

    def test_whitespace_only_transcript(self):
        result = self.s.summarize("   \n  \t  ")
        self.assertEqual(result["summary"], "")

    def test_ratio_affects_length(self):
        r_short = self.s.summarize(LONG_TRANSCRIPT, ratio=0.1)
        r_long  = self.s.summarize(LONG_TRANSCRIPT, ratio=0.8)
        self.assertLessEqual(
            len(r_short["summary"]), len(r_long["summary"])
        )

    def test_summary_contains_original_words(self):
        result = self.s.summarize(SHORT_TRANSCRIPT)
        summary_words = set(result["summary"].lower().split())
        original_words = set(SHORT_TRANSCRIPT.lower().split())
        overlap = summary_words & original_words
        self.assertGreater(len(overlap), 5,
                           "Summary shares too few words with the original")

    def test_single_sentence_transcript(self):
        result = self.s.summarize("The team agreed to meet on Monday.")
        self.assertIsInstance(result["summary"], str)

    def test_speaker_labels_stripped(self):
        text   = "Speaker 1: We should review the budget. Speaker 2: I agree with that."
        result = self.s.summarize(text)
        self.assertNotIn("Speaker 1:", result["summary"])
        self.assertNotIn("Speaker 2:", result["summary"])


# ── ActionExtractor tests ─────────────────────────────────────────────────────

class TestActionExtractor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.e = ActionExtractor()
        cls.e.load()

    def test_returns_dict_with_required_keys(self):
        result = self.e.extract(SHORT_TRANSCRIPT)
        self.assertIn("action_items", result)
        self.assertIn("decisions",    result)

    def test_detects_obligation_verbs(self):
        text   = "Alice should send the report by Friday."
        result = self.e.extract(text)
        self.assertTrue(
            len(result["action_items"]) > 0,
            "Expected at least one action item for 'should send'"
        )

    def test_detects_task_verbs(self):
        text   = "Bob will review the vendor contracts."
        result = self.e.extract(text)
        self.assertTrue(len(result["action_items"]) > 0)

    def test_detects_decisions(self):
        text   = "We decided to increase the marketing budget."
        result = self.e.extract(text)
        self.assertTrue(len(result["decisions"]) > 0)

    def test_agreed_classified_as_decision(self):
        text   = "The team agreed to hold weekly check-ins."
        result = self.e.extract(text)
        self.assertTrue(len(result["decisions"]) > 0)

    def test_action_items_have_text(self):
        result = self.e.extract(SHORT_TRANSCRIPT)
        for item in result["action_items"]:
            self.assertIn("text",     item)
            self.assertIn("assignee", item)
            self.assertIn("confidence", item)
            self.assertIsInstance(item["text"], str)
            self.assertTrue(len(item["text"]) > 0)

    def test_confidence_in_range(self):
        result = self.e.extract(SHORT_TRANSCRIPT)
        for item in result["action_items"]:
            self.assertGreaterEqual(item["confidence"], 0.0)
            self.assertLessEqual(item["confidence"],    1.0)

    def test_empty_text(self):
        result = self.e.extract("")
        self.assertEqual(result["action_items"], [])
        self.assertEqual(result["decisions"],    [])

    def test_no_false_positives_on_plain_statement(self):
        text   = "The sky is blue. Water is wet. Grass is green."
        result = self.e.extract(text)
        # Plain factual statements should not be flagged as action items
        self.assertEqual(len(result["action_items"]), 0)

    def test_long_transcript_action_items(self):
        result = self.e.extract(LONG_TRANSCRIPT)
        self.assertGreater(len(result["action_items"]), 2,
                           "Expected multiple action items in the long transcript")

    def test_long_transcript_decisions(self):
        result = self.e.extract(LONG_TRANSCRIPT)
        self.assertGreater(len(result["decisions"]), 0,
                           "Expected at least one decision in the long transcript")

    def test_deduplication(self):
        # If the same content appears twice it should not produce duplicate items
        text   = SHORT_TRANSCRIPT + " " + SHORT_TRANSCRIPT
        result = self.e.extract(text)
        texts  = [item["text"] for item in result["action_items"]]
        self.assertEqual(len(texts), len(set(texts)),
                         "Duplicate action items were not removed")

    def test_assignee_extraction(self):
        text   = "Alice should prepare the presentation."
        result = self.e.extract(text)
        if result["action_items"]:
            # The assignee field should be a string (may be empty if NER misses it)
            self.assertIsInstance(result["action_items"][0]["assignee"], str)


# ── Integration: summarizer → extractor pipeline ─────────────────────────────

class TestNLPPipeline(unittest.TestCase):
    """Ensure the two modules work correctly when chained."""

    @classmethod
    def setUpClass(cls):
        cls.s = Summarizer()
        cls.e = ActionExtractor()
        cls.s.load()
        cls.e.load()

    def test_extract_from_summary(self):
        """Action items extracted from a summary should still be meaningful."""
        sum_result = self.s.summarize(LONG_TRANSCRIPT)
        ext_result = self.e.extract(sum_result["summary"])
        # At minimum, both keys should exist
        self.assertIn("action_items", ext_result)
        self.assertIn("decisions",    ext_result)

    def test_pipeline_does_not_crash_on_minimal_input(self):
        sum_result = self.s.summarize("We must submit the form.")
        ext_result = self.e.extract(sum_result["summary"])
        self.assertIsInstance(ext_result["action_items"], list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
