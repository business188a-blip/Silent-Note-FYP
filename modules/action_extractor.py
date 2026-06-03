"""
modules/action_extractor.py
──────────────────────────────────────────────────────────────────────────────
Extracts action items and key decisions from meeting transcripts using
spaCy's linguistic annotations — no internet required.

Supports English and Urdu transcripts. For Urdu, regex-based pattern
matching is used since spaCy does not have an Urdu model.

Scoring logic (0.0 - 1.0, threshold = 0.25 to include):
  +0.50  strong follow-up cue  (book a call, walk you through, reach out)
  +0.40  obligation phrase     (will, should, must, need to, let me)
  +0.30  question-form action  (X will tell/show/update us?)
  +0.20  task verb in obligation/imperative context
  +0.30  imperative sentence   (starts with base-form verb)
  -0.40  process/descriptive   (first we, then we, we find, we identify)
──────────────────────────────────────────────────────────────────────────────
"""

import re
import threading
from typing import Optional

import spacy


# ── English Patterns ──────────────────────────────────────────────────────────

OBLIGATION_PHRASES = [
    r"\bshould\b", r"\bmust\b", r"\bneed to\b", r"\bneeds to\b",
    r"\bhave to\b", r"\bhas to\b", r"\bwill\b", r"\bshall\b",
    r"\bgoing to\b", r"\bgonna\b", r"\bwant to\b", r"\bwants to\b",
    r"\bplanning to\b", r"\bplan to\b", r"\bwould like to\b",
    r"\blet me\b", r"\blet us\b", r"\blet's\b",
    r"\bwill tell\b", r"\bwill show\b", r"\bwill update\b",
    r"\bwill present\b", r"\bwill share\b", r"\bwill report\b",
    r"\bwill inform\b", r"\bwill explain\b", r"\bwill give\b",
]

TASK_VERBS = {
    "prepare", "send", "email", "submit", "review", "update", "check",
    "schedule", "create", "write", "draft", "arrange", "organise",
    "organize", "contact", "call", "inform", "notify", "present", "confirm",
    "coordinate", "ensure", "fix", "resolve", "complete", "finish",
    "deliver", "share", "upload", "download", "implement", "deploy",
    "test", "verify", "analyse", "analyze", "research", "investigate",
    "report", "document", "record",
    "book", "set", "hop", "jump", "connect", "reach",
    "walk", "show", "demonstrate", "demo", "discuss", "explore",
    "ask", "explain", "clarify", "provide", "give",
    "start", "begin", "open", "close", "handle", "meet", "talk",
    "follow", "circle",
    "tell", "present", "brief", "summarise", "summarize", "outline",
    "highlight", "describe", "cover", "address", "raise", "mention",
}

DECISION_PHRASES = [
    r"\bdecided\b", r"\bagreed\b", r"\bconfirmed\b", r"\bapproved\b",
    r"\bresolved\b", r"\bfinalised\b", r"\bfinalized\b",
    r"the decision is", r"we have decided", r"it was agreed",
    r"it has been decided", r"consensus was", r"we agreed that",
]

FOLLOWUP_PHRASES = [
    r"\blet me know\b", r"\bget back to\b", r"\btouch base\b",
    r"\bcircle back\b", r"\bfollow up\b", r"\bfollow-up\b",
    r"\bbook a\b", r"\bschedule a\b", r"\bset up a\b",
    r"\bhop on a\b", r"\bjump on a\b", r"\bget on a\b",
    r"\bopen a.*zoom\b", r"\bopen a.*call\b", r"\bopen a.*meeting\b",
    r"\bbook the\b", r"\bschedule the\b",
    r"\bsend me\b", r"\bsend you\b", r"\bsend us\b",
    r"\bgive me a call\b", r"\bgive you a call\b",
    r"\bwalk you through\b", r"\bwalk me through\b",
    r"\breach out\b", r"\bbook a calendar\b",
]

PROCESS_INDICATORS = [
    r"\bfirst we\b", r"\bthen we\b", r"\bnext we\b",
    r"\bwe (identify|find|locate|scan|search|look for)\b",
    r"\bwe (qualify|pre-qualify|filter|sort|rank)\b",
    r"\bwe (pass|transfer|hand|refer) them\b",
    r"\bthe (system|process|pipeline|workflow)\b",
    r"\bwe are actively\b", r"\bwe might\b", r"\bwe could\b",
]

_OBLIGATION_RE = re.compile("|".join(OBLIGATION_PHRASES), re.IGNORECASE)
_DECISION_RE   = re.compile("|".join(DECISION_PHRASES),   re.IGNORECASE)
_FOLLOWUP_RE   = re.compile("|".join(FOLLOWUP_PHRASES),   re.IGNORECASE)
_PROCESS_RE    = re.compile("|".join(PROCESS_INDICATORS), re.IGNORECASE)

MIN_CONFIDENCE = 0.25


# ── Urdu Patterns ─────────────────────────────────────────────────────────────

URDU_OBLIGATION_PHRASES = [
    "گا", "گی", "گے",
    "کرے گا", "کرے گی", "کریں گے",
    "بھیجے گا", "بھیجے گی", "بھیجیں گے",
    "دے گا", "دے گی", "دیں گے",
    "لے گا", "لے گی", "لیں گے",
    "بتائے گا", "بتائے گی", "بتائیں گے",
    "پیش کرے گا", "پیش کرے گی", "پیش کریں گے",
    "چاہیے", "ضروری ہے", "کرنا ہے", "کرنا ہوگا",
    "کرنی ہے", "کرنی ہوگی", "کرنے ہیں", "کرنے ہوں گے",
    "لازمی", "لازم ہے",
    "چلیں", "آئیں", "ہم کریں گے", "ہمیں کرنا ہے",
]

URDU_DECISION_PHRASES = [
    "فیصلہ ہوا", "طے پایا", "منظور ہوا", "منظور کیا",
    "اتفاق ہوا", "اتفاق کیا", "طے ہو گیا", "فیصلہ کیا گیا",
    "یہ طے ہوا", "یہ فیصلہ ہوا",
]

URDU_TASK_PHRASES = [
    "بھیجنا", "بھیجیں", "جمع کرنا", "جمع کریں",
    "جائزہ لینا", "جائزہ لیں", "چیک کرنا", "چیک کریں",
    "تیار کرنا", "تیار کریں", "بنانا", "بنائیں",
    "اپ ڈیٹ کرنا", "اپ ڈیٹ کریں", "بتانا", "بتائیں",
    "آگاہ کرنا", "آگاہ کریں", "مطلع کرنا",
    "مکمل کرنا", "مکمل کریں", "ختم کرنا", "ختم کریں",
    "شیڈول کرنا", "شیڈول کریں", "ترتیب دینا", "ترتیب دیں",
    "پیش کرنا", "پیش کریں", "دکھانا", "دکھائیں",
    "تصدیق کرنا", "تصدیق کریں", "کنفرم کرنا", "کنفرم کریں",
    "رابطہ کرنا", "رابطہ کریں", "فون کرنا", "فون کریں",
    "رپورٹ دینا", "رپورٹ دیں", "رپورٹ کرنا",
]

# Common Pakistani names for Urdu assignee detection
URDU_NAMES = [
    "عثمان", "ارسلان", "احمد", "محمد", "علی", "حسن", "عمر",
    "زید", "طلحہ", "بلال", "سارہ", "فاطمہ", "عائشہ", "مریم",
    "زینب", "نور", "حنا", "رابعہ", "صدیق", "انعام", "کامران",
    "عمران", "ناصر", "طارق", "ظفر", "شاہد", "وقار", "فیصل",
]

_URDU_OBLIGATION_RE = re.compile(
    "|".join(re.escape(p) for p in URDU_OBLIGATION_PHRASES)
)
_URDU_DECISION_RE = re.compile(
    "|".join(re.escape(p) for p in URDU_DECISION_PHRASES)
)
_URDU_TASK_RE = re.compile(
    "|".join(re.escape(p) for p in URDU_TASK_PHRASES)
)


# ── ActionExtractor ───────────────────────────────────────────────────────────

class ActionExtractor:

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
                    self._nlp = spacy.load("en_core_web_sm")

    def extract(self, text: str, language: str = "en") -> dict:
        """
        Extract action items and decisions from transcript.

        Parameters
        ──────────
        text     : str  — transcript text
        language : str  — 'en' for English, 'ur' for Urdu, others fall back to English
        """
        if not text.strip():
            return {"action_items": [], "decisions": []}

        if language == "ur":
            return self._extract_urdu(text)

        return self._extract_english(text)

    # ── English extraction ────────────────────────────────────────────────────

    def _extract_english(self, text: str) -> dict:
        self._ensure_loaded()
        assert self._nlp is not None

        doc       = self._nlp(text)
        sentences = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 5]

        action_items = []
        decisions    = []

        for sent in sentences:
            sent_doc = self._nlp(sent)

            if _DECISION_RE.search(sent):
                decisions.append(sent)
                continue

            confidence = self._score_confidence(sent, sent_doc)

            if confidence >= MIN_CONFIDENCE:
                assignee = self._find_assignee(sent_doc)
                action_items.append({
                    "text":       sent,
                    "assignee":   assignee,
                    "confidence": confidence,
                })

        action_items = self._deduplicate(action_items)
        return {"action_items": action_items, "decisions": decisions}

    # ── Urdu extraction ───────────────────────────────────────────────────────

    def _extract_urdu(self, text: str) -> dict:
        """
        Regex-based extraction for Urdu transcripts.
        spaCy does not support Urdu, so we use pattern matching only.
        Sentences are split on Urdu punctuation and newlines.
        """
        raw_sents = re.split(r"[۔؟!\n]+", text)
        sentences = [s.strip() for s in raw_sents if len(s.strip()) > 4]

        action_items = []
        decisions    = []

        for sent in sentences:
            if _URDU_DECISION_RE.search(sent):
                decisions.append(sent)
                continue

            score = 0.0

            if _URDU_OBLIGATION_RE.search(sent):
                score += 0.40

            if _URDU_TASK_RE.search(sent):
                score += 0.30

            # Question form with obligation
            if sent.strip().endswith("؟") and _URDU_OBLIGATION_RE.search(sent):
                score += 0.30

            if score >= MIN_CONFIDENCE:
                assignee = self._find_urdu_assignee(sent)
                action_items.append({
                    "text":       sent,
                    "assignee":   assignee,
                    "confidence": round(min(score, 1.0), 2),
                })

        action_items = self._deduplicate(action_items)
        return {"action_items": action_items, "decisions": decisions}

    def _find_urdu_assignee(self, sent: str) -> str:
        for name in URDU_NAMES:
            if name in sent:
                return name
        # Also catch English names embedded in Urdu (mixed meetings)
        english_name = re.search(r"\b[A-Z][a-z]{2,}\b", sent)
        if english_name:
            return english_name.group()
        return ""

    # ── English scoring ───────────────────────────────────────────────────────

    def _score_confidence(self, sent: str, doc: spacy.tokens.Doc) -> float:  # type: ignore
        score = 0.0

        if _FOLLOWUP_RE.search(sent):
            score += 0.50

        if _OBLIGATION_RE.search(sent):
            score += 0.40

        if sent.strip().endswith("?") and _OBLIGATION_RE.search(sent):
            score += 0.30

        has_obligation = _OBLIGATION_RE.search(sent) is not None
        is_imperative  = self._is_imperative(doc)

        if has_obligation or is_imperative:
            for token in doc:
                if token.pos_ == "VERB" and token.lemma_.lower() in TASK_VERBS:
                    score += 0.20
                    break

        if is_imperative:
            score += 0.30

        if _PROCESS_RE.search(sent):
            score -= 0.40

        return round(min(max(score, 0.0), 1.0), 2)

    def _is_imperative(self, doc: spacy.tokens.Doc) -> bool:  # type: ignore
        FILLERS = {
            "so", "okay", "ok", "well", "now", "just", "right",
            "yeah", "yes", "no", "hey", "hi", "alright", "great",
            "next", "first", "then", "also", "and", "but",
        }
        for token in doc:
            if token.is_punct or token.is_space:
                continue
            if token.text.lower() in FILLERS:
                continue
            return token.tag_ == "VB" and token.lemma_.lower() in TASK_VERBS
        return False

    def _find_assignee(self, doc: spacy.tokens.Doc) -> str:  # type: ignore
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        for token in doc:
            if token.dep_ in ("nsubj", "nsubjpass") and token.pos_ == "PROPN":
                return token.text
        for token in doc:
            if token.dep_ == "nsubj" and token.pos_ == "PRON":
                if token.text.lower() in ("i", "we"):
                    return "Team"
                if token.text.lower() == "you":
                    return "Client"
        return ""

    # ── Post-processing ───────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(items: list[dict]) -> list[dict]:
        texts = [it["text"] for it in items]
        seen: set[str] = set()
        keep = []
        for item in items:
            normalized = re.sub(r"\s+", " ", item["text"].strip().lower())
            if normalized in seen:
                continue
            dominated = any(
                item["text"] != other and item["text"] in other
                for other in texts
            )
            if not dominated:
                seen.add(normalized)
                keep.append(item)
        return keep

    def _ensure_loaded(self) -> None:
        if self._nlp is None:
            self.load()
