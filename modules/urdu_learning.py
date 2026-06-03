"""
Offline Urdu learning pipeline for SilentNote.

The teacher can only make local rule/normalization changes in fully offline
mode. This module learns from those changes automatically and also supports
explicit user edits from the transcript box.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path

from modules.urdu_memory import apply_corrections, save_correction
from modules.urdu_teacher import teacher_fix


MIN_TOKEN_LEN = 1


def _tokens(text: str) -> list[str]:
    return [token for token in (text or "").split() if token.strip()]


def _valid_pair(wrong: str, correct: str) -> bool:
    if wrong == correct:
        return False
    if len(wrong) < MIN_TOKEN_LEN or len(correct) < MIN_TOKEN_LEN:
        return False
    if any(ch.isdigit() for ch in wrong + correct):
        return False
    return True


def learn_from_correction(
    original: str,
    corrected: str,
    db_path: str | Path | None = None,
) -> int:
    """
    Save word/phrase replacements discovered between two versions of text.

    Returns the number of correction pairs stored.
    """
    original_tokens = _tokens(original)
    corrected_tokens = _tokens(corrected)
    if not original_tokens or not corrected_tokens:
        return 0

    matcher = SequenceMatcher(a=original_tokens, b=corrected_tokens)
    saved = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        wrong = " ".join(original_tokens[i1:i2]).strip()
        correct = " ".join(corrected_tokens[j1:j2]).strip()
        if _valid_pair(wrong, correct):
            save_correction(wrong, correct, db_path=db_path)
            saved += 1

    return saved


def clean_and_learn(
    text: str,
    db_path: str | Path | None = None,
    min_count: int = 2,
) -> str:
    """Run the offline teacher, learn teacher changes, then apply memory."""
    if not text:
        return text

    taught = teacher_fix(text)
    learn_from_correction(text, taught, db_path=db_path)
    return apply_corrections(taught, min_count=min_count, db_path=db_path)
