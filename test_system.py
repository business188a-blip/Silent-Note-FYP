import os
import sys
import wave
import struct
import math
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Project root ko path mein add karo
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Colors for terminal output ─────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

PASS = f"{GREEN}✓ PASS{RESET}"
FAIL = f"{RED}✗ FAIL{RESET}"
INFO = f"{BLUE}ℹ{RESET}"

results = []


def section(title):
    print(f"\n{BOLD}{BLUE}{'─'*55}{RESET}")
    print(f"{BOLD}{BLUE}  {title}{RESET}")
    print(f"{BOLD}{BLUE}{'─'*55}{RESET}")


def test(name, fn):
    """Run a single test and record result."""
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, ""))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"         {RED}{e}{RESET}")
        results.append((name, False, str(e)))


def make_test_wav(duration_sec=10, sample_rate=16000, text="test"):
    """
    Synthetic WAV file banao — silence ke saath sine wave mixed.
    Yeh Whisper ke liye valid input hai.
    """
    n_frames   = int(sample_rate * duration_sec)
    fd, path   = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        # 440 Hz sine wave (A note) — Whisper isko silence se better handle karta hai
        frames = []
        for i in range(n_frames):
            val = int(3000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            frames.append(struct.pack('<h', val))
        wf.writeframes(b"".join(frames))

    return path


# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*55}")
print(f"  SilentNote — System Test")
print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
print(f"{'═'*55}{RESET}")

# ── TEST 1: Database ───────────────────────────────────────────────────────────
section("1. DATABASE  (SQLite + Encryption)")

import tempfile as _tmp
from database.db_manager import DBManager, encrypt, decrypt

_tmp_db = _tmp.NamedTemporaryFile(suffix=".db", delete=False)
_tmp_db.close()
db = DBManager(db_path=Path(_tmp_db.name))

def t_encrypt_roundtrip():
    plain = "Alice should send the report by Friday."
    assert decrypt(encrypt(plain)) == plain

def t_encrypt_urdu():
    plain = "یہ ایک آزمائشی متن ہے۔"
    assert decrypt(encrypt(plain)) == plain

def t_create_meeting():
    global _mid
    _mid = db.create_meeting(title="Test Meeting", language="en")
    assert isinstance(_mid, int) and _mid > 0

def t_update_transcript():
    db.update_meeting(_mid, transcript="Alice will prepare the report.")
    m = db.get_meeting(_mid)
    assert m["transcript"] == "Alice will prepare the report."

def t_add_action_items():
    db.add_action_items(_mid, ["Send report", "Schedule meeting"])
    items = db.get_action_items(_mid)
    assert len(items) == 2

def t_add_snapshot():
    db.add_snapshot(_mid, "/fake/path/snap.jpg", note="Whiteboard")
    snaps = db.get_snapshots(_mid)
    assert len(snaps) == 1

def t_stats():
    stats = db.get_stats()
    assert stats["total_meetings"] >= 1

def t_delete_meeting():
    db.delete_meeting(_mid)
    assert db.get_meeting(_mid) is None

test("Encryption roundtrip (English)",    t_encrypt_roundtrip)
test("Encryption roundtrip (Urdu)",       t_encrypt_urdu)
test("Create meeting",                    t_create_meeting)
test("Update + decrypt transcript",       t_update_transcript)
test("Add & retrieve action items",       t_add_action_items)
test("Add & retrieve snapshot",           t_add_snapshot)
test("Dashboard stats",                   t_stats)
test("Delete meeting (cascade)",          t_delete_meeting)

os.unlink(_tmp_db.name)

# ── TEST 2: Audio Handler ──────────────────────────────────────────────────────
section("2. AUDIO HANDLER")

from modules.audio_handler import AudioHandler

_wav_path = make_test_wav(duration_sec=5)

def t_wav_created():
    assert os.path.exists(_wav_path)
    assert os.path.getsize(_wav_path) > 0

def t_load_wav_normalise():
    out = AudioHandler.load_audio_file(_wav_path)
    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
    os.unlink(out)

def t_list_devices():
    devices = AudioHandler().list_devices()
    assert isinstance(devices, list)

def t_stop_when_not_recording():
    assert AudioHandler().stop_recording() is None

test("Synthetic WAV created",             t_wav_created)
test("WAV normalised to 16kHz mono",      t_load_wav_normalise)
test("list_devices() returns list",       t_list_devices)
test("stop() without record = None",      t_stop_when_not_recording)

# ── TEST 3: Transcriber ────────────────────────────────────────────────────────
section("3. TRANSCRIBER  (Whisper Medium — real inference)")

print(f"  {YELLOW}Loading Whisper medium model into memory…{RESET}")

from modules.transcriber import Transcriber

_transcriber = Transcriber()

def t_model_loads():
    _transcriber.load_model()
    assert _transcriber.is_loaded()

def t_transcribe_returns_dict():
    # Use a 5-second sine wave — Whisper will return empty/noise text but
    # the structure must be correct
    result = _transcriber.transcribe_file(_wav_path)
    assert "text"     in result
    assert "segments" in result
    assert "language" in result

def t_transcribe_segments_structure():
    result = _transcriber.transcribe_file(_wav_path)
    for seg in result["segments"]:
        assert "start"   in seg
        assert "end"     in seg
        assert "text"    in seg
        assert "speaker" in seg

def t_language_detection():
    lang = _transcriber.detect_language(_wav_path)
    assert isinstance(lang, str) and len(lang) >= 2

test("Whisper model loads",               t_model_loads)
test("transcribe_file() returns dict",    t_transcribe_returns_dict)
test("Segments have correct structure",   t_transcribe_segments_structure)
test("Language detection works",          t_language_detection)

# ── TEST 4: NLP — Summarizer ───────────────────────────────────────────────────
section("4. NLP  (Summarizer + Action Extractor)")

from modules.summarizer       import Summarizer
from modules.action_extractor import ActionExtractor

SAMPLE_TEXT = (
    "Alice should send the quarterly budget report to the board by Friday. "
    "We decided to increase the marketing budget by fifteen percent. "
    "Bob must review all vendor contracts before the end of this month. "
    "The team agreed to hold weekly check-ins every Tuesday at ten AM. "
    "Everyone needs to submit their department goals by Thursday. "
    "Sara will schedule the technical interviews for shortlisted candidates. "
    "We confirmed that the office move-in date is the fifteenth of next month. "
    "The management approved a budget of five hundred thousand rupees for renovation."
)

_summarizer = Summarizer()
_extractor  = ActionExtractor()

def t_summarizer_loads():
    _summarizer.load()

def t_summary_non_empty():
    r = _summarizer.summarize(SAMPLE_TEXT)
    assert r["summary"].strip() != ""

def t_summary_shorter_than_original():
    r = _summarizer.summarize(SAMPLE_TEXT)
    assert len(r["summary"]) < len(SAMPLE_TEXT)

def t_key_points_list():
    r = _summarizer.summarize(SAMPLE_TEXT)
    assert isinstance(r["key_points"], list)
    assert len(r["key_points"]) > 0

def t_empty_text_safe():
    r = _summarizer.summarize("")
    assert r["summary"] == ""

def t_extractor_loads():
    _extractor.load()

def t_action_items_detected():
    r = _extractor.extract(SAMPLE_TEXT)
    assert len(r["action_items"]) > 0

def t_decisions_detected():
    r = _extractor.extract(SAMPLE_TEXT)
    assert len(r["decisions"]) > 0

def t_action_item_structure():
    r = _extractor.extract(SAMPLE_TEXT)
    for item in r["action_items"]:
        assert "text"       in item
        assert "assignee"   in item
        assert "confidence" in item

def t_confidence_range():
    r = _extractor.extract(SAMPLE_TEXT)
    for item in r["action_items"]:
        assert 0.0 <= item["confidence"] <= 1.0

test("Summarizer loads",                  t_summarizer_loads)
test("Summary is non-empty",              t_summary_non_empty)
test("Summary shorter than original",     t_summary_shorter_than_original)
test("Key points returned as list",       t_key_points_list)
test("Empty text handled safely",         t_empty_text_safe)
test("ActionExtractor loads",             t_extractor_loads)
test("Action items detected",             t_action_items_detected)
test("Decisions detected",                t_decisions_detected)
test("Action item structure correct",     t_action_item_structure)
test("Confidence scores in [0, 1]",       t_confidence_range)

# ── TEST 5: Emotion Detector ───────────────────────────────────────────────────
section("5. EMOTION DETECTOR")

from modules.emotion_detector import EmotionDetector

_detector = EmotionDetector()
# Force keyword fallback for speed (model inference is slow on CPU)
_detector._loaded  = True
_detector._fallback = True

HAPPY_TEXT = "Excellent results. The team is excited and happy about the fantastic progress."
ANGRY_TEXT = "Very frustrated with delays. The manager is furious and annoyed."

def t_detect_returns_keys():
    r = _detector.detect(HAPPY_TEXT)
    for k in ("dominant_emotion","confidence","distribution","meeting_mood"):
        assert k in r

def t_confidence_range_emotion():
    r = _detector.detect(HAPPY_TEXT)
    assert 0.0 <= r["confidence"] <= 1.0

def t_happy_positive():
    r = _detector.detect(HAPPY_TEXT)
    assert r["dominant_emotion"] in ("joy","surprise")

def t_angry_negative():
    r = _detector.detect(ANGRY_TEXT)
    assert r["dominant_emotion"] in ("anger","disgust","fear")

def t_empty_is_neutral():
    r = _detector.detect("")
    assert r["dominant_emotion"] == "neutral"

def t_mood_phrase_string():
    r = _detector.detect(HAPPY_TEXT)
    assert isinstance(r["meeting_mood"], str) and len(r["meeting_mood"]) > 0

def t_per_segment():
    segs = [
        {"id":0,"start":0.0,"end":5.0,"text":"Great work everyone."},
        {"id":1,"start":5.0,"end":10.0,"text":"I am frustrated with the delays."},
    ]
    result = _detector.detect_per_segment(segs)
    assert len(result) == 2
    for seg in result:
        assert "emotion" in seg

test("detect() returns required keys",    t_detect_returns_keys)
test("Confidence in [0, 1]",             t_confidence_range_emotion)
test("Happy text → positive emotion",     t_happy_positive)
test("Angry text → negative emotion",     t_angry_negative)
test("Empty text → neutral",              t_empty_is_neutral)
test("Meeting mood is a string",          t_mood_phrase_string)
test("Per-segment emotion detection",     t_per_segment)

# ── TEST 6: Exporter ───────────────────────────────────────────────────────────
section("6. EXPORTER  (PDF / DOCX / JSON)")

from modules.exporter import Exporter

EXPORT_DIR = Path("data/test_exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

_exporter = Exporter()

MEETING_DATA = {
    "title":         "Q4 Budget Review",
    "started_at":    "2026-01-15T09:00:00",
    "ended_at":      "2026-01-15T10:15:00",
    "duration_sec":  4500,
    "language":      "en",
    "speaker_count": 2,
    "emotion_label": "joy",
    "emotion_score": 0.72,
    "summary":       "The team approved the marketing budget increase and decided on weekly check-ins.",
    "decisions":     ["Approved ten percent marketing increase.", "Weekly check-ins every Tuesday."],
    "transcript":    "Speaker 1: Good morning. Let us begin.\nSpeaker 2: The budget looks good.",
    "action_items":  [
        {"text":"Alice will prepare the report.","assignee":"Alice","done":False},
        {"text":"Bob schedules check-ins.","assignee":"Bob","done":True},
    ],
    "snapshots":     [],
}

_pdf_path  = str(EXPORT_DIR / "test_export.pdf")
_docx_path = str(EXPORT_DIR / "test_export.docx")
_json_path = str(EXPORT_DIR / "test_export.json")

def t_export_pdf():
    _exporter.export_pdf(MEETING_DATA, _pdf_path)
    assert os.path.exists(_pdf_path)
    assert os.path.getsize(_pdf_path) > 0
    with open(_pdf_path,"rb") as f:
        assert f.read(4) == b"%PDF"

def t_export_docx():
    _exporter.export_docx(MEETING_DATA, _docx_path)
    assert os.path.exists(_docx_path)
    assert os.path.getsize(_docx_path) > 0
    with open(_docx_path,"rb") as f:
        assert f.read(4) == b"PK\x03\x04"

def t_export_json():
    import json
    _exporter.export_json(MEETING_DATA, _json_path)
    assert os.path.exists(_json_path)
    with open(_json_path,"r",encoding="utf-8") as f:
        data = json.load(f)
    for key in ("title","summary","transcript","action_items","decisions"):
        assert key in data

def t_export_urdu_json():
    import json
    meeting = {"title":"Urdu Test","summary":"یہ ایک آزمائشی خلاصہ ہے۔"}
    path    = str(EXPORT_DIR / "urdu_test.json")
    _exporter.export_json(meeting, path)
    with open(path,"r",encoding="utf-8") as f:
        data = json.load(f)
    assert data["summary"] == "یہ ایک آزمائشی خلاصہ ہے۔"

def t_fmt_duration():
    assert Exporter._fmt_dur(45)   == "45s"
    assert Exporter._fmt_dur(125)  == "2m 5s"
    assert Exporter._fmt_dur(3661) == "1h 1m"
    assert Exporter._fmt_dur(0)    == "0s"

test("Export PDF (valid file + header)",  t_export_pdf)
test("Export DOCX (valid ZIP format)",    t_export_docx)
test("Export JSON (valid + keys)",        t_export_json)
test("JSON Urdu text roundtrip",          t_export_urdu_json)
test("Duration formatter",                t_fmt_duration)

# ── TEST 7: Full Pipeline ──────────────────────────────────────────────────────
section("7. FULL PIPELINE  (end-to-end simulation)")

def t_full_pipeline():
    """
    Simulate a complete meeting session:
    DB create → transcribe → summarize → extract → emotion → DB save → export
    """
    import tempfile as _t2
    tmp = _t2.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    _db = DBManager(db_path=Path(tmp.name))

    # Step 1: Create meeting
    mid = _db.create_meeting(title="Pipeline Test", language="en")
    assert mid > 0

    # Step 2: Transcribe (use SAMPLE_TEXT as if it came from Whisper)
    transcript = SAMPLE_TEXT
    _db.update_meeting(mid, transcript=transcript, duration_sec=300, speaker_count=2)

    # Step 3: Summarize
    s   = Summarizer()
    sr  = s.summarize(transcript)
    assert sr["summary"] != ""
    _db.update_meeting(mid, summary=sr["summary"])

    # Step 4: Extract action items
    e   = ActionExtractor()
    er  = e.extract(transcript)
    assert len(er["action_items"]) > 0
    _db.add_action_items(mid, [i["text"] for i in er["action_items"]])

    # Step 5: Emotion
    ed  = EmotionDetector()
    ed._loaded  = True
    ed._fallback = True
    emo = ed.detect(transcript)
    _db.update_meeting(mid, emotion_label=emo["dominant_emotion"],
                       emotion_score=emo["confidence"])

    # Step 6: Retrieve and verify
    meeting = _db.get_meeting(mid)
    assert meeting["transcript"] == transcript
    assert meeting["summary"]    == sr["summary"]
    assert meeting["emotion_label"] is not None

    items = _db.get_action_items(mid)
    assert len(items) > 0

    # Step 7: Export all formats
    meeting["action_items"] = items
    meeting["snapshots"]    = []
    meeting["decisions"]    = er["decisions"]

    pdf_p  = str(EXPORT_DIR / "pipeline_test.pdf")
    docx_p = str(EXPORT_DIR / "pipeline_test.docx")
    json_p = str(EXPORT_DIR / "pipeline_test.json")

    _exporter.export_pdf(meeting,  pdf_p)
    _exporter.export_docx(meeting, docx_p)
    _exporter.export_json(meeting, json_p)

    assert os.path.exists(pdf_p)
    assert os.path.exists(docx_p)
    assert os.path.exists(json_p)

    # Step 8: Delete
    _db.delete_meeting(mid)
    assert _db.get_meeting(mid) is None

    os.unlink(tmp.name)

test("Complete end-to-end pipeline",      t_full_pipeline)

# ── Cleanup ────────────────────────────────────────────────────────────────────
os.unlink(_wav_path)

# ── Summary ────────────────────────────────────────────────────────────────────
total   = len(results)
passed  = sum(1 for _, ok, _ in results if ok)
failed  = total - passed

print(f"\n{BOLD}{'═'*55}")
print(f"  RESULTS:  {GREEN}{passed} passed{RESET}  |  {RED}{failed} failed{RESET}  |  {total} total")
print(f"{'═'*55}{RESET}\n")

if failed > 0:
    print(f"{RED}Failed tests:{RESET}")
    for name, ok, err in results:
        if not ok:
            print(f"  ✗  {name}")
            print(f"     {err}")
    print()

print(f"{INFO} Exported test files saved to:  {EXPORT_DIR.resolve()}")
print(f"{INFO} Open PDF/DOCX files to verify formatting.\n")

sys.exit(0 if failed == 0 else 1)