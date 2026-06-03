"""
modules/summarizer.py
──────────────────────────────────────────────────────────────────────────────
Generates a concise summary from a meeting transcript using extractive
Natural Language Processing — no internet or cloud API required.

Supports English and Urdu transcripts.

For English: TF-IDF sentence scoring via spaCy + NLTK.
For Urdu: frequency-based sentence scoring using basic tokenization,
          since spaCy does not have an Urdu model.
──────────────────────────────────────────────────────────────────────────────
"""

import math
import re
import threading
from collections import Counter
from typing import Optional

import spacy
from nltk.corpus import stopwords


# ── One-time downloads ────────────────────────────────────────────────────────



# ── Constants ─────────────────────────────────────────────────────────────────

SPACY_MODEL   = "en_core_web_sm"
DEFAULT_RATIO = 0.35
MIN_SENTENCES = 3
MAX_SENTENCES = 15
try:
    ENGLISH_STOPS = set(stopwords.words("english"))
except LookupError:
    ENGLISH_STOPS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with", "we", "you", "our", "their",
    }

FILLER_WORDS = {
    "um", "uh", "er", "ah", "hmm", "okay", "ok", "yeah", "yep",
    "right", "like", "basically", "actually", "literally", "so",
    "you know", "i mean", "kind of", "sort of", "just", "well",
}

KEYWORD_BOOST_WORDS = {
    "budget", "timeline", "deadline", "decision", "agree", "agreed",
    "confirm", "confirmed", "approve", "approved", "reject", "rejected",
    "propose", "proposed", "plan", "strategy", "goal", "objective",
    "action", "task", "responsible", "assign", "owner", "priority",
    "issue", "problem", "risk", "solution", "resolve", "resolved",
    "cost", "price", "revenue", "profit", "loss", "target", "milestone",
    "next", "step", "follow", "meeting", "schedule", "call", "zoom",
    "client", "customer", "lead", "job", "quote", "estimate", "contract",
    "service", "product", "offer", "deal", "close", "pitch", "qualify",
    "flooring", "install", "project", "work", "team", "fit",
}

# Urdu filler words to exclude from scoring
URDU_FILLER_WORDS = {
    "ہے", "ہیں", "کا", "کی", "کے", "میں", "سے", "پر", "کو",
    "اور", "یہ", "وہ", "جو", "تو", "بھی", "نہیں", "ہو", "تھا",
    "تھی", "تھے", "ایک", "اس", "ان", "آپ", "ہم", "مجھے", "اب",
}

# Urdu keywords that boost sentence importance
URDU_KEYWORD_BOOST = {
    "فیصلہ", "میٹنگ", "کام", "رپورٹ", "بجٹ", "ڈیڈ لائن", "ڈیڈلائن",
    "منصوبہ", "ہدف", "مسئلہ", "حل", "اگلی", "ضروری", "اہم",
    "ویب سائٹ", "پروجیکٹ", "کلائنٹ", "ٹیم", "شیڈول", "تاریخ",
    "مکمل", "تیار", "بھیجنا", "جائزہ", "تصدیق", "اپ ڈیٹ",
}

FILLER_SENTENCE_PATTERNS = [
    r"^(okay|ok|yeah|yep|right|so|well|alright|sure|great|sounds good)[.,!?]?\s*$",
    r"^(i see|i understand|got it|makes sense)[.,!?]?\s*$",
]
_FILLER_SENT_RE = re.compile(
    "|".join(FILLER_SENTENCE_PATTERNS), re.IGNORECASE
)


# ── Summarizer ────────────────────────────────────────────────────────────────

class Summarizer:

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nlp = None
            cls._instance._load_lock = threading.Lock()
        return cls._instance

    def __init__(self):
        pass


    def load(self) -> None:
        if self._nlp is None:
            with self._load_lock:
                if self._nlp is None:
                    self._nlp = spacy.load(
                        SPACY_MODEL,
                        disable=["ner", "lemmatizer"],
                    )

    def summarize(self, transcript: str,
                  ratio: float = DEFAULT_RATIO,
                  language: str = "en") -> dict:
        """
        Summarise a transcript string.

        Parameters
        ──────────
        transcript : str   — raw transcript from Whisper
        ratio      : float — fraction of sentences to retain
        language   : str   — 'en' for English, 'ur' for Urdu

        Returns
        ───────
        dict with keys: summary, key_points, word_count, sent_count
        """
        if language == "ur":
            return self._summarize_urdu(transcript, ratio)

        return self._summarize_english(transcript, ratio)

    # ── English pipeline ──────────────────────────────────────────────────────

    def _summarize_english(self, transcript: str, ratio: float) -> dict:
        self._ensure_loaded()

        cleaned = self._clean_transcript(transcript)
        if not cleaned.strip():
            return {"summary": "", "key_points": [], "word_count": 0, "sent_count": 0}

        sentences = self._split_sentences(cleaned)
        if len(sentences) == 0:
            return {
                "summary":    cleaned,
                "key_points": [],
                "word_count": len(cleaned.split()),
                "sent_count": 1,
            }

        if len(sentences) <= MIN_SENTENCES:
            return {
                "summary":    " ".join(sentences),
                "key_points": self._extract_key_points(sentences),
                "word_count": len(cleaned.split()),
                "sent_count": len(sentences),
            }

        scores   = self._score_sentences(sentences)
        n        = self._target_count(len(sentences), ratio)
        ranked   = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:n])

        summary_sents = [sentences[i] for i in selected]
        return {
            "summary":    " ".join(summary_sents),
            "key_points": self._extract_key_points(summary_sents),
            "word_count": len(cleaned.split()),
            "sent_count": len(sentences),
        }

    # ── Urdu pipeline ─────────────────────────────────────────────────────────

    def _summarize_urdu(self, transcript: str, ratio: float) -> dict:
        """
        Extractive summarization for Urdu using word frequency scoring.
        No spaCy model needed — uses basic regex tokenization.
        """
        cleaned = self._clean_transcript_urdu(transcript)
        if not cleaned.strip():
            return {"summary": "", "key_points": [], "word_count": 0, "sent_count": 0}

        # Split on Urdu sentence endings
        raw_sents = re.split(r"[۔؟!\n]+", cleaned)
        sentences = [s.strip() for s in raw_sents if len(s.strip()) > 8]

        if not sentences:
            return {
                "summary":    cleaned,
                "key_points": [],
                "word_count": len(cleaned.split()),
                "sent_count": 1,
            }

        if len(sentences) <= MIN_SENTENCES:
            return {
                "summary":    "۔ ".join(sentences),
                "key_points": self._extract_key_points_urdu(sentences),
                "word_count": len(cleaned.split()),
                "sent_count": len(sentences),
            }

        scores   = self._score_sentences_urdu(sentences)
        n        = self._target_count(len(sentences), ratio)
        ranked   = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:n])

        summary_sents = [sentences[i] for i in selected]
        return {
            "summary":    "۔ ".join(summary_sents),
            "key_points": self._extract_key_points_urdu(summary_sents),
            "word_count": len(cleaned.split()),
            "sent_count": len(sentences),
        }

    def _clean_transcript_urdu(self, text: str) -> str:
        """Remove speaker labels and normalize whitespace for Urdu text."""
        text = re.sub(r"(\[?Speaker\s*\d+\]?\s*:?)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _score_sentences_urdu(self, sentences: list[str]) -> list[float]:
        """Score Urdu sentences by word frequency + keyword boost + position."""
        stop = URDU_FILLER_WORDS
        n    = len(sentences)

        token_lists = []
        for sent in sentences:
            # Tokenize by whitespace for Urdu
            tokens = [t for t in sent.split() if t not in stop and len(t) > 1]
            token_lists.append(tokens)

        freq: Counter = Counter()
        for tokens in token_lists:
            freq.update(tokens)

        if not freq:
            return [1.0] * n

        max_freq  = freq.most_common(1)[0][1]
        norm_freq = {word: count / max_freq for word, count in freq.items()}

        scores = []
        for idx, tokens in enumerate(token_lists):
            if not tokens:
                scores.append(0.0)
                continue

            raw   = sum(norm_freq.get(t, 0.0) for t in tokens)
            score = raw / math.sqrt(len(tokens))

            # Keyword boost
            sent_text    = sentences[idx]
            keyword_hits = sum(1 for kw in URDU_KEYWORD_BOOST if kw in sent_text)
            score       += min(keyword_hits * 0.15, 0.60)

            # Position bias
            position_ratio = idx / max(n - 1, 1)
            if position_ratio <= 0.20 or position_ratio >= 0.80:
                score *= 1.25

            scores.append(score)

        return scores

    def _extract_key_points_urdu(self, sentences: list[str]) -> list[str]:
        """Shorten Urdu sentences into brief bullets (max ~12 words)."""
        bullets = []
        for sent in sentences:
            words = sent.split()
            if len(words) <= 12:
                bullets.append(sent.rstrip("۔"))
            else:
                bullets.append(" ".join(words[:12]) + "…")
        return bullets

    # ── English internal pipeline ─────────────────────────────────────────────

    def _clean_transcript(self, text: str) -> str:
        text = re.sub(r"(\[?Speaker\s*\d+\]?\s*:?)", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _split_sentences(self, text: str) -> list[str]:
        self._ensure_loaded()
        doc   = self._nlp(text)
        sents = []
        for s in doc.sents:
            sent = s.text.strip()
            if len(sent) < 10:
                continue
            if _FILLER_SENT_RE.match(sent):
                continue
            sents.append(sent)
        return sents

    def _score_sentences(self, sentences: list[str]) -> list[float]:
        stop = ENGLISH_STOPS | FILLER_WORDS
        n    = len(sentences)

        token_lists = []
        for sent in sentences:
            tokens = re.findall(r"\b[a-z]{2,}\b", sent.lower())
            tokens = [t for t in tokens if t not in stop]
            token_lists.append(tokens)

        freq: Counter = Counter()
        for tokens in token_lists:
            freq.update(tokens)

        if not freq:
            return [1.0] * n

        max_freq  = freq.most_common(1)[0][1]
        norm_freq = {word: count / max_freq for word, count in freq.items()}

        scores = []
        for idx, tokens in enumerate(token_lists):
            if not tokens:
                scores.append(0.0)
                continue

            raw   = sum(norm_freq.get(t, 0.0) for t in tokens)
            score = raw / math.sqrt(len(tokens))

            sent_lower   = sentences[idx].lower()
            keyword_hits = sum(1 for kw in KEYWORD_BOOST_WORDS if kw in sent_lower)
            score       += min(keyword_hits * 0.15, 0.60)

            position_ratio = idx / max(n - 1, 1)
            if position_ratio <= 0.20 or position_ratio >= 0.80:
                score *= 1.25

            scores.append(score)

        return scores

    def _target_count(self, total: int, ratio: float) -> int:
        n = max(MIN_SENTENCES, round(total * ratio))
        return min(n, MAX_SENTENCES, total)

    def _extract_key_points(self, sentences: list[str]) -> list[str]:
        bullets = []
        for sent in sentences:
            words = sent.split()
            if len(words) <= 15:
                bullets.append(sent.rstrip("."))
            else:
                bullets.append(" ".join(words[:15]) + "…")
        return bullets

    def _ensure_loaded(self) -> None:
        if self._nlp is None:
            self.load()
