"""
database/db_manager.py
──────────────────────────────────────────────────────────────────────────────
Handles all SQLite operations for SilentNote.

Sensitive text fields (transcript, summary, action_items) are encrypted using
the Fernet symmetric encryption scheme from the 'cryptography' library before
being written to disk. The encryption key is derived from a machine-specific
salt and stored in a local key file alongside the database. This satisfies the
proposal's requirement of encrypted local storage without needing sqlcipher.

Tables
──────
  meetings        – one row per meeting session
  snapshots       – one row per captured image, foreign-keyed to meetings
  action_items    – individual extracted tasks, foreign-keyed to meetings

All datetime values are stored as ISO-8601 strings (UTC).
──────────────────────────────────────────────────────────────────────────────
"""

import os
import sqlite3
import json
from datetime import datetime, timezone
from pathlib import Path
from cryptography.fernet import Fernet


# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
DB_PATH    = DATA_DIR / "silentnote.db"
KEY_PATH   = DATA_DIR / "silentnote.key"


# ── Encryption helpers ────────────────────────────────────────────────────────

def _load_or_create_key() -> bytes:
    """
    Load the Fernet key from KEY_PATH, or generate and save a new one.
    Called once at module level so every DBManager instance shares the key.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes()
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    return key


_FERNET = Fernet(_load_or_create_key())


def encrypt(plaintext: str) -> str:
    """Encrypt a UTF-8 string and return the ciphertext as a UTF-8 string."""
    if not plaintext:
        return ""
    return _FERNET.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt(ciphertext: str) -> str:
    """Decrypt a ciphertext string previously produced by encrypt()."""
    if not ciphertext:
        return ""
    return _FERNET.decrypt(ciphertext.encode("utf-8")).decode("utf-8")


# ── DBManager ─────────────────────────────────────────────────────────────────

class DBManager:
    """
    Lightweight wrapper around sqlite3.  Every public method opens a fresh
    connection so the class is safe to use from any thread (Qt worker threads
    included) without needing a connection-per-thread pool.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row          # access columns by name
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_schema(self) -> None:
        """Create tables if they do not already exist."""
        ddl = """
        CREATE TABLE IF NOT EXISTS meetings (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            title           TEXT    NOT NULL DEFAULT 'Untitled Meeting',
            started_at      TEXT    NOT NULL,
            ended_at        TEXT,
            duration_sec    INTEGER DEFAULT 0,
            language        TEXT    DEFAULT 'en',
            transcript_enc  TEXT,
            summary_enc     TEXT,
            decisions_enc   TEXT,
            emotion_label   TEXT,
            emotion_score   REAL,
            speaker_count   INTEGER DEFAULT 1,
            audio_path      TEXT,
            created_at      TEXT    NOT NULL
        );

        CREATE TABLE IF NOT EXISTS snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
            image_path  TEXT    NOT NULL,
            captured_at TEXT    NOT NULL,
            note        TEXT    DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS action_items (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id  INTEGER NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
            item_enc    TEXT    NOT NULL,
            assignee    TEXT    DEFAULT '',
            due_date    TEXT    DEFAULT '',
            done        INTEGER DEFAULT 0
        );
        """
        with self._connect() as conn:
            conn.executescript(ddl)
            self._ensure_column(conn, "meetings", "decisions_enc", "TEXT")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str,
                       column: str, column_type: str) -> None:
        columns = {
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ── Meeting CRUD ──────────────────────────────────────────────────────────

    def create_meeting(self, title: str = "Untitled Meeting",
                       language: str = "en") -> int:
        """
        Insert a new meeting row with started_at = now.
        Returns the new row's id.
        """
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO meetings (title, started_at, language, created_at)
            VALUES (?, ?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.execute(sql, (title, now, language, now))
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def update_meeting(self, meeting_id: int, **kwargs) -> None:
        """
        Update any subset of meeting fields.

        Sensitive fields that will be encrypted automatically:
          transcript, summary

        Other accepted kwargs:
          title, ended_at, duration_sec, language, emotion_label,
          emotion_score, speaker_count, audio_path
        """
        col_map = {
            "transcript":    "transcript_enc",
            "summary":       "summary_enc",
            "decisions":     "decisions_enc",
            "title":         "title",
            "ended_at":      "ended_at",
            "duration_sec":  "duration_sec",
            "language":      "language",
            "emotion_label": "emotion_label",
            "emotion_score": "emotion_score",
            "speaker_count": "speaker_count",
            "audio_path":    "audio_path",
        }
        encrypted_fields = {"transcript", "summary", "decisions"}

        assignments = []
        values      = []

        for key, value in kwargs.items():
            if key not in col_map:
                continue
            col = col_map[key]
            if key in encrypted_fields:
                if key == "decisions" and not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False)
                value = encrypt(str(value))
            assignments.append(f"{col} = ?")
            values.append(value)

        if not assignments:
            return

        values.append(meeting_id)
        sql = f"UPDATE meetings SET {', '.join(assignments)} WHERE id = ?"
        with self._connect() as conn:
            conn.execute(sql, values)

    def get_meeting(self, meeting_id: int) -> dict | None:
        """Return a single meeting as a plain dict with decrypted fields."""
        sql = "SELECT * FROM meetings WHERE id = ?"
        with self._connect() as conn:
            row = conn.execute(sql, (meeting_id,)).fetchone()
        if row is None:
            return None
        return self._decrypt_meeting_row(dict(row))

    def get_all_meetings(self, search: str = "",
                         limit: int = 100, offset: int = 0) -> list[dict]:
        """
        Return meetings ordered by newest first.
        If search is given, filter by title (case-insensitive).
        """
        sql = """
            SELECT * FROM meetings
            WHERE title LIKE ?
            ORDER BY started_at DESC, id DESC
            LIMIT ? OFFSET ?
        """
        pattern = f"%{search}%"
        with self._connect() as conn:
            rows = conn.execute(sql, (pattern, limit, offset)).fetchall()
        return [self._decrypt_meeting_row(dict(r)) for r in rows]

    def delete_meeting(self, meeting_id: int) -> None:
        """Delete a meeting and all its snapshots/action items (cascade)."""
        with self._connect() as conn:
            conn.execute("DELETE FROM meetings WHERE id = ?", (meeting_id,))

    def _decrypt_meeting_row(self, row: dict) -> dict:
        """Decrypt encrypted columns in a meeting dict in-place."""
        row["transcript"] = decrypt(row.pop("transcript_enc", "") or "")
        row["summary"]    = decrypt(row.pop("summary_enc",    "") or "")
        decisions_raw = decrypt(row.pop("decisions_enc", "") or "")
        if decisions_raw:
            try:
                row["decisions"] = json.loads(decisions_raw)
            except json.JSONDecodeError:
                row["decisions"] = [
                    line.strip() for line in decisions_raw.splitlines()
                    if line.strip()
                ]
        else:
            row["decisions"] = []
        return row

    # ── Snapshot CRUD ─────────────────────────────────────────────────────────

    def add_snapshot(self, meeting_id: int, image_path: str,
                     note: str = "") -> int:
        """Insert a snapshot row. Returns the new row's id."""
        now = datetime.now(timezone.utc).isoformat()
        sql = """
            INSERT INTO snapshots (meeting_id, image_path, captured_at, note)
            VALUES (?, ?, ?, ?)
        """
        with self._connect() as conn:
            cur = conn.execute(sql, (meeting_id, image_path, now, note))
            assert cur.lastrowid is not None
            return int(cur.lastrowid)

    def get_snapshots(self, meeting_id: int) -> list[dict]:
        """Return all snapshots for a meeting ordered by captured_at."""
        sql = """
            SELECT * FROM snapshots
            WHERE meeting_id = ?
            ORDER BY captured_at ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (meeting_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_snapshot(self, snapshot_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM snapshots WHERE id = ?", (snapshot_id,))

    # ── Action Items CRUD ─────────────────────────────────────────────────────

    def add_action_items(self, meeting_id: int,
                         items: list[str | dict]) -> None:
        """
        Bulk-insert action items for a meeting.
        Accepts plain strings or dicts with text/item, assignee, due_date, done.
        Item text is encrypted before storage.
        """
        rows = []
        for item in items:
            if isinstance(item, dict):
                text = str(item.get("text") or item.get("item") or "").strip()
                assignee = str(item.get("assignee") or "")
                due_date = str(item.get("due_date") or "")
                done = 1 if item.get("done") else 0
            else:
                text = str(item).strip()
                assignee = ""
                due_date = ""
                done = 0
            if text:
                rows.append((meeting_id, encrypt(text), assignee, due_date, done))

        sql = """
            INSERT INTO action_items
                (meeting_id, item_enc, assignee, due_date, done)
            VALUES (?, ?, ?, ?, ?)
        """
        with self._connect() as conn:
            conn.executemany(sql, rows)

    def get_action_items(self, meeting_id: int) -> list[dict]:
        """Return decrypted action items for a meeting."""
        sql = """
            SELECT * FROM action_items
            WHERE meeting_id = ?
            ORDER BY id ASC
        """
        with self._connect() as conn:
            rows = conn.execute(sql, (meeting_id,)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["item"] = decrypt(d.pop("item_enc", "") or "")
            d["text"] = d["item"]
            result.append(d)
        return result

    def toggle_action_item(self, item_id: int) -> None:
        """Flip the done flag of an action item."""
        sql = "UPDATE action_items SET done = 1 - done WHERE id = ?"
        with self._connect() as conn:
            conn.execute(sql, (item_id,))

    def delete_action_items(self, meeting_id: int) -> None:
        """Delete all action items for a meeting."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM action_items WHERE meeting_id = ?", (meeting_id,)
            )

    # ── Dashboard / Analytics ─────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return aggregate statistics used by the dashboard.

        Keys returned:
          total_meetings, total_duration_sec, avg_duration_sec,
          total_snapshots, total_action_items, emotion_distribution (dict),
          meetings_per_day (list of {date, count})
        """
        with self._connect() as conn:
            total_meetings = conn.execute(
                "SELECT COUNT(*) FROM meetings"
            ).fetchone()[0]

            total_duration = conn.execute(
                "SELECT COALESCE(SUM(duration_sec), 0) FROM meetings"
            ).fetchone()[0]

            avg_duration = conn.execute(
                "SELECT COALESCE(AVG(duration_sec), 0) FROM meetings"
            ).fetchone()[0]

            total_snapshots = conn.execute(
                "SELECT COUNT(*) FROM snapshots"
            ).fetchone()[0]

            total_actions = conn.execute(
                "SELECT COUNT(*) FROM action_items"
            ).fetchone()[0]

            # Emotion distribution
            emotion_rows = conn.execute("""
                SELECT emotion_label, COUNT(*) as cnt
                FROM meetings
                WHERE emotion_label IS NOT NULL
                GROUP BY emotion_label
                ORDER BY cnt DESC
            """).fetchall()
            emotion_dist = {r["emotion_label"]: r["cnt"] for r in emotion_rows}

            # Meetings per day (last 30 days)
            mpd_rows = conn.execute("""
                SELECT DATE(started_at) as day, COUNT(*) as cnt
                FROM meetings
                GROUP BY day
                ORDER BY day DESC
                LIMIT 30
            """).fetchall()
            meetings_per_day = [{"date": r["day"], "count": r["cnt"]}
                                 for r in mpd_rows]

        return {
            "total_meetings":    total_meetings,
            "total_duration_sec": total_duration,
            "avg_duration_sec":  avg_duration,
            "total_snapshots":   total_snapshots,
            "total_action_items": total_actions,
            "emotion_distribution": emotion_dist,
            "meetings_per_day":  meetings_per_day,
        }
