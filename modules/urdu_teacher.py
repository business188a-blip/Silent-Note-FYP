"""
Offline Urdu teacher helpers for SilentNote.

This module does lightweight, fully local cleanup before learned corrections
are applied. If urduhack is installed, its normalizer is used; otherwise the
fallback keeps the app running offline with standard Unicode normalization.
"""

from __future__ import annotations

import re
import unicodedata


_CHAR_REPLACEMENTS = {
    "ي": "ی",
    "ى": "ی",
    "ك": "ک",
    "ۀ": "ہ",
    "ة": "ہ",
    "ؤ": "و",
}


def normalize_urdu(text: str) -> str:
    """Normalize Urdu text without requiring an internet-backed model."""
    if not text:
        return text

    try:
        import urduhack

        normalized = urduhack.normalization.normalize(text)
    except Exception:
        normalized = unicodedata.normalize("NFC", text)

    for wrong, correct in _CHAR_REPLACEMENTS.items():
        normalized = normalized.replace(wrong, correct)

    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def teacher_fix(text: str, ai_model=None) -> str:
    """
    Produce a clean Urdu version of text.

    SilentNote is configured for fully offline use, so ai_model should normally
    be None. The hook stays here for future hybrid mode experiments.
    """
    text = normalize_urdu(text)

    if ai_model:
        prompt = f"""
Fix Urdu spelling only.
Do NOT change meaning.

Text:
{text}
"""
        text = ai_model(prompt)

    return text
