"""
emotion_detector.py
Detects emotional tone of meeting transcripts.
Uses distilroberta model, falls back to keywords if model unavailable.
"""

import os
import threading
import warnings
from typing import Optional

# force offline mode immediately — never attempt internet
os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"


class EmotionDetector:

    MODEL_NAME  = "j-hartmann/emotion-english-distilroberta-base"
    CHUNK_WORDS = 300
    EMOTIONS    = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._classifier = None
            cls._instance._loaded = False
            cls._instance._fallback = False
            cls._instance._load_lock = threading.Lock()
        return cls._instance

    def __init__(self):
        pass


    def load(self, on_status=None):
        if self._loaded:
            return

        with self._load_lock:
            if self._loaded:
                return

            if on_status:
                on_status("Loading emotion detection model...")

            try:
                from transformers import pipeline as hf_pipeline
                self._classifier = hf_pipeline(
                    "text-classification",
                    model            = self.MODEL_NAME,
                    top_k            = None,
                    device           = -1,
                    local_files_only = True,
                )
                self._fallback = False
                if on_status:
                    on_status("Emotion detection model ready.")
            except Exception:
                self._fallback = True
                if on_status:
                    on_status("Emotion detection: using keyword fallback.")

            self._loaded = True

    def is_loaded(self):
        return self._loaded

    def detect(self, text):
        self._ensure_loaded()

        if not text or not text.strip():
            return self._neutral_result()

        if self._fallback:
            return self._keyword_detect(text)

        return self._model_detect(text)

    def detect_per_segment(self, segments):
        self._ensure_loaded()
        result = []
        for seg in segments:
            seg  = dict(seg)
            text = seg.get("text", "")
            if text.strip():
                detected       = self.detect(text)
                seg["emotion"] = detected["dominant_emotion"]
            else:
                seg["emotion"] = "neutral"
            result.append(seg)
        return result

    def _model_detect(self, text):
        words  = text.split()
        chunks = [
            " ".join(words[i: i + self.CHUNK_WORDS])
            for i in range(0, len(words), self.CHUNK_WORDS)
        ]
        total = {e: 0.0 for e in self.EMOTIONS}
        valid = 0

        for chunk in chunks:
            if not chunk.strip():
                continue
            try:
                raw = self._classifier(chunk[:512])[0]
                for item in raw:
                    label = item["label"].lower()
                    if label in total:
                        total[label] += item["score"]
                valid += 1
            except Exception:
                continue

        if valid == 0:
            return self._neutral_result()

        distribution = {k: round(v / valid, 4) for k, v in total.items()}
        dominant     = max(distribution, key=distribution.get)
        confidence   = distribution[dominant]

        return {
            "dominant_emotion": dominant,
            "confidence":       confidence,
            "distribution":     distribution,
            "meeting_mood":     self._mood_phrase(dominant, confidence),
        }

    # simple keyword counting, not perfect but works offline
    _KEYWORD_MAP = {
        "anger":    ["angry", "furious", "annoyed", "frustrated", "upset",
                     "disagree", "conflict", "argue", "hostile"],
        "joy":      ["happy", "great", "excellent", "wonderful", "fantastic",
                     "excited", "pleased", "good", "positive", "celebrate"],
        "sadness":  ["sad", "sorry", "unfortunately", "regret", "disappoint",
                     "miss", "loss", "failed", "difficult", "struggle"],
        "fear":     ["worried", "concern", "risk", "danger", "afraid",
                     "uncertain", "anxious", "threat", "problem", "issue"],
        "surprise": ["unexpected", "surprise", "suddenly", "shocked",
                     "unbelievable", "amazing", "wow", "astonish"],
        "disgust":  ["disgusting", "awful", "terrible", "horrible",
                     "unacceptable", "wrong", "bad", "reject"],
    }

    def _keyword_detect(self, text):
        text_lower = text.lower()
        counts     = {emotion: 0 for emotion in self._KEYWORD_MAP}

        for emotion, keywords in self._KEYWORD_MAP.items():
            for kw in keywords:
                counts[emotion] += text_lower.count(kw)

        total        = sum(counts.values()) or 1
        distribution = {e: round(c / total, 4) for e, c in counts.items()}
        distribution["neutral"] = round(max(0.0, 1.0 - sum(distribution.values())), 4)

        dominant   = max(distribution, key=distribution.get)
        confidence = distribution[dominant]

        return {
            "dominant_emotion": dominant,
            "confidence":       confidence,
            "distribution":     distribution,
            "meeting_mood":     self._mood_phrase(dominant, confidence),
        }

    @staticmethod
    def _neutral_result():
        dist = {e: 0.0 for e in EmotionDetector.EMOTIONS}
        dist["neutral"] = 1.0
        return {
            "dominant_emotion": "neutral",
            "confidence":       1.0,
            "distribution":     dist,
            "meeting_mood":     "Calm and professional",
        }

    @staticmethod
    def _mood_phrase(emotion, confidence):
        intensity = "very " if confidence > 0.7 else ""
        phrases = {
            "joy":      f"The meeting had a {intensity}positive atmosphere.",
            "anger":    f"The meeting had {intensity}tense moments.",
            "sadness":  f"The meeting carried a {intensity}sombre tone.",
            "fear":     f"The meeting involved {intensity}concerned discussion.",
            "surprise": f"The meeting had {intensity}unexpected elements.",
            "disgust":  f"The meeting included {intensity}critical feedback.",
            "neutral":  "The meeting was calm and professional.",
        }
        return phrases.get(emotion, "The meeting tone was mixed.")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()
