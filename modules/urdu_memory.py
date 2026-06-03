"""
Offline Urdu learning memory for SilentNote.

Corrections are stored locally in SQLite and only become automatic after they
have been seen at least twice, which keeps one-off edits from overfitting the
transcription pipeline.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "urdu_memory.db"


def _resolve_path(db_path: str | Path | None = None) -> Path:
    return Path(db_path) if db_path is not None else DB_PATH


def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    path = _resolve_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def init_db(db_path: str | Path | None = None) -> None:
    with _connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS corrections (
                wrong TEXT PRIMARY KEY,
                correct TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 1
            )
        """)


def save_correction(
    wrong: str,
    correct: str,
    db_path: str | Path | None = None,
) -> None:
    wrong = (wrong or "").strip()
    correct = (correct or "").strip()
    if not wrong or not correct or wrong == correct:
        return

    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO corrections (wrong, correct, count)
            VALUES (?, ?, 1)
            ON CONFLICT(wrong) DO UPDATE SET
                correct = excluded.correct,
                count = count + 1
            """,
            (wrong, correct),
        )


def load_corrections(
    db_path: str | Path | None = None,
) -> list[tuple[str, str, int]]:
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT wrong, correct, count FROM corrections ORDER BY count DESC"
        ).fetchall()
    return [(wrong, correct, count) for wrong, correct, count in rows]


def apply_corrections(
    text: str,
    min_count: int = 2,
    db_path: str | Path | None = None,
) -> str:
    if not text:
        return text

    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT wrong, correct
            FROM corrections
            WHERE count >= ?
            ORDER BY LENGTH(wrong) DESC
            """,
            (min_count,),
        ).fetchall()

    for wrong, correct in rows:
        text = text.replace(wrong, correct)
    return text
