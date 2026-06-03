"""Tests for model loader idempotency without loading real models."""

import sys
import unittest
from unittest.mock import Mock, patch


class TestModelLoadGuards(unittest.TestCase):

    def tearDown(self):
        for module_name, class_name in (
            ("modules.summarizer", "Summarizer"),
            ("modules.action_extractor", "ActionExtractor"),
            ("modules.emotion_detector", "EmotionDetector"),
            ("modules.transcriber", "Transcriber"),
        ):
            module = sys.modules.get(module_name)
            if module is not None:
                getattr(module, class_name)._instance = None

    def test_summarizer_loads_spacy_once(self):
        try:
            from modules import summarizer as mod
        except Exception as exc:
            self.skipTest(f"Summarizer dependencies are not importable: {exc}")

        mod.Summarizer._instance = None
        summarizer = mod.Summarizer()

        with patch.object(mod.spacy, "load", return_value=object()) as load:
            summarizer.load()
            summarizer.load()

        load.assert_called_once_with(
            mod.SPACY_MODEL,
            disable=["ner", "lemmatizer"],
        )

    def test_action_extractor_loads_spacy_once(self):
        try:
            from modules import action_extractor as mod
        except Exception as exc:
            self.skipTest(f"Action extractor dependencies are not importable: {exc}")

        mod.ActionExtractor._instance = None
        extractor = mod.ActionExtractor()

        with patch.object(mod.spacy, "load", return_value=object()) as load:
            extractor.load()
            extractor.load()

        load.assert_called_once_with("en_core_web_sm")

    def test_emotion_detector_loads_pipeline_once(self):
        try:
            from modules import emotion_detector as mod
        except Exception as exc:
            self.skipTest(f"Emotion detector dependencies are not importable: {exc}")

        fake_transformers = Mock()
        fake_transformers.pipeline.return_value = object()
        mod.EmotionDetector._instance = None
        detector = mod.EmotionDetector()

        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            detector.load()
            detector.load()

        fake_transformers.pipeline.assert_called_once()

    def test_transcriber_loads_whisper_once(self):
        try:
            from modules import transcriber as mod
        except Exception as exc:
            self.skipTest(f"Whisper is not importable: {exc}")

        mod.Transcriber._instance = None
        transcriber = mod.Transcriber()

        with patch.object(mod.whisper, "load_model", return_value=object()) as load:
            transcriber.load_model()
            transcriber.load_model()

        load.assert_called_once_with(
            mod.MODEL_NAME,
            device=mod.DEVICE,
            in_memory=True,
        )
