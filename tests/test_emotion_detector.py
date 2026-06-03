"""
tests/test_emotion_detector.py
────────────────────────────────────────────────────────────────────────────
Unit tests for modules/emotion_detector.py

Two sets of tests:
  1. Keyword-based fallback (always available, no model download needed)
  2. Full model path (skipped if transformers / model not installed)
────────────────────────────────────────────────────────────────────────────
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.emotion_detector import EmotionDetector


HAPPY_TEXT = (
    "This was an excellent meeting. Everyone was excited about the new product. "
    "We celebrated a fantastic quarter and the results were wonderful."
)

ANGRY_TEXT = (
    "I am very frustrated with the delays. The team is annoyed and the manager "
    "is furious. There was a serious conflict during the discussion."
)

NEUTRAL_TEXT = (
    "We reviewed the document. The report was presented. The meeting ended at noon."
)

EMPTY_TEXT = ""

SEGMENTS = [
    {"id": 0, "start": 0.0,  "end": 5.0,  "text": "I am so happy with the results."},
    {"id": 1, "start": 5.0,  "end": 10.0, "text": "The team is frustrated with the delays."},
    {"id": 2, "start": 10.0, "end": 15.0, "text": "We reviewed the document carefully."},
]


class TestEmotionDetectorFallback(unittest.TestCase):
    """
    Force the keyword fallback path by setting _fallback=True directly.
    These tests run without any model download.
    """

    def setUp(self):
        self.det          = EmotionDetector()
        self.det._loaded  = True
        self.det._fallback = True

    def test_returns_required_keys(self):
        result = self.det.detect(HAPPY_TEXT)
        for key in ("dominant_emotion", "confidence", "distribution", "meeting_mood"):
            self.assertIn(key, result)

    def test_happy_text_positive_emotion(self):
        result = self.det.detect(HAPPY_TEXT)
        self.assertIn(result["dominant_emotion"],
                      ["joy", "surprise"],
                      "Happy text should produce a positive emotion")

    def test_angry_text_negative_emotion(self):
        result = self.det.detect(ANGRY_TEXT)
        self.assertIn(result["dominant_emotion"],
                      ["anger", "disgust", "fear"])

    def test_confidence_in_range(self):
        for text in (HAPPY_TEXT, ANGRY_TEXT, NEUTRAL_TEXT):
            result = self.det.detect(text)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"],    1.0)

    def test_distribution_all_emotions_present(self):
        result = self.det.detect(HAPPY_TEXT)
        dist   = result["distribution"]
        for emo in EmotionDetector.EMOTIONS:
            self.assertIn(emo, dist)

    def test_distribution_values_in_range(self):
        result = self.det.detect(HAPPY_TEXT)
        for score in result["distribution"].values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score,    1.0)

    def test_empty_text_returns_neutral(self):
        result = self.det.detect(EMPTY_TEXT)
        self.assertEqual(result["dominant_emotion"], "neutral")
        self.assertEqual(result["confidence"],       1.0)

    def test_meeting_mood_is_string(self):
        result = self.det.detect(HAPPY_TEXT)
        self.assertIsInstance(result["meeting_mood"], str)
        self.assertGreater(len(result["meeting_mood"]), 0)

    def test_detect_per_segment_adds_emotion_key(self):
        result = self.det.detect_per_segment(SEGMENTS)
        self.assertEqual(len(result), len(SEGMENTS))
        for seg in result:
            self.assertIn("emotion", seg)
            self.assertIsInstance(seg["emotion"], str)

    def test_detect_per_segment_preserves_original_keys(self):
        result = self.det.detect_per_segment(SEGMENTS)
        for orig, proc in zip(SEGMENTS, result):
            self.assertEqual(proc["id"],    orig["id"])
            self.assertEqual(proc["text"],  orig["text"])
            self.assertEqual(proc["start"], orig["start"])

    def test_detect_per_segment_empty_list(self):
        result = self.det.detect_per_segment([])
        self.assertEqual(result, [])

    def test_neutral_result_structure(self):
        result = EmotionDetector._neutral_result()
        self.assertEqual(result["dominant_emotion"], "neutral")
        self.assertEqual(result["confidence"],       1.0)
        self.assertIn("distribution",   result)
        self.assertIn("meeting_mood",   result)


class TestEmotionDetectorMoodPhrases(unittest.TestCase):

    def test_all_emotions_have_mood_phrases(self):
        for emotion in EmotionDetector.EMOTIONS:
            phrase = EmotionDetector._mood_phrase(emotion, 0.5)
            self.assertIsInstance(phrase, str)
            self.assertGreater(len(phrase), 0)

    def test_high_confidence_adds_very(self):
        phrase_high = EmotionDetector._mood_phrase("joy", 0.9)
        phrase_low  = EmotionDetector._mood_phrase("joy", 0.3)
        self.assertIn("very", phrase_high)
        self.assertNotIn("very", phrase_low)

    def test_unknown_emotion_returns_fallback(self):
        phrase = EmotionDetector._mood_phrase("confusion", 0.5)
        self.assertIsInstance(phrase, str)


if __name__ == "__main__":
    unittest.main(verbosity=2)
