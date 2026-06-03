import os
from pathlib import Path

from PyQt5.QtCore    import Qt, QTimer, QThread, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QLineEdit, QComboBox, QProgressBar, QFileDialog, QMessageBox,
    QGroupBox, QSplitter, QFrame,
)

from modules.audio_handler    import AudioHandler
from modules.transcriber      import Transcriber
from modules.summarizer       import Summarizer
from modules.action_extractor import ActionExtractor
from modules.speaker_diarizer import SpeakerDiarizer
from modules.emotion_detector import EmotionDetector
from modules.snapshot_handler import SnapshotHandler, CameraNotFoundError
from modules.exporter         import Exporter
from modules.urdu_learning    import learn_from_correction
from database.db_manager      import DBManager


# ── Language map ──────────────────────────────────────────────────────────────
LANG_MAP = {
    "Auto-detect":   None,
    "English (en)":  "en",
    "Urdu (ur)":     "ur",
    "Punjabi (pa)":  "pa",
}

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


class TranscribeWorker(QObject):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, wav_path, language):
        super().__init__()
        self.wav_path = wav_path
        self.language = language or None

    def run(self):
        try:
            t = Transcriber()
            result = t.transcribe_file(self.wav_path, language=self.language)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


def _hline():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    return line


class RecordingTab(QWidget):

    meeting_saved = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._audio     = AudioHandler()
        self._db        = DBManager()
        self._exporter  = Exporter()
        self._snapshots = SnapshotHandler()

        self._meeting_id    = None
        self._wav_path      = None
        self._transcript    = ""
        self._transcript_lines = []
        self._summary       = ""
        self._action_items  = []
        self._decisions     = []
        self._detected_lang = "en"

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick_timer)
        self._elapsed_sec = 0
        self._thread = None
        self._worker = None

        self._build_ui()
        self._populate_devices()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(24, 20, 24, 20)

        # ── Meeting Info ───────────────────────────────────────────────────
        lbl = QLabel("MEETING INFO")
        lbl.setObjectName("lbl_section")
        lbl.setStyleSheet(
            "font-size:9px;font-weight:800;color:#b0a898;"
            "letter-spacing:2px;text-transform:uppercase;padding-bottom:6px;"
        )
        root.addWidget(lbl)

        info_row = QHBoxLayout()
        info_row.setSpacing(12)

        tc = QVBoxLayout()
        tc.setSpacing(4)
        tc.addWidget(QLabel("Title"))
        self.edit_title = QLineEdit("Untitled Meeting")
        self.edit_title.setPlaceholderText("Enter meeting title…")
        self.edit_title.setMinimumHeight(36)
        tc.addWidget(self.edit_title)
        info_row.addLayout(tc, stretch=3)

        lc = QVBoxLayout()
        lc.setSpacing(4)
        lc.addWidget(QLabel("Language"))
        self.combo_lang = QComboBox()
        self.combo_lang.addItems(list(LANG_MAP.keys()))
        self.combo_lang.setMinimumHeight(36)
        lc.addWidget(self.combo_lang)
        info_row.addLayout(lc, stretch=1)

        mc = QVBoxLayout()
        mc.setSpacing(4)
        mc.addWidget(QLabel("Microphone"))
        self.combo_device = QComboBox()
        self.combo_device.setMinimumHeight(36)
        mc.addWidget(self.combo_device)
        info_row.addLayout(mc, stretch=2)

        root.addLayout(info_row)
        root.addSpacing(20)
        root.addWidget(_hline())
        root.addSpacing(16)

        # ── Controls ───────────────────────────────────────────────────────
        lbl2 = QLabel("CONTROLS")
        lbl2.setStyleSheet(
            "font-size:9px;font-weight:800;color:#b0a898;"
            "letter-spacing:2px;text-transform:uppercase;padding-bottom:6px;"
        )
        root.addWidget(lbl2)

        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        self.btn_record = QPushButton("⏺   Record")
        self.btn_record.setObjectName("btn_record")
        self.btn_record.setMinimumHeight(42)
        self.btn_record.setMinimumWidth(140)
        self.btn_record.clicked.connect(self._on_record)

        self.btn_stop = QPushButton("⏹   Stop")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setMinimumHeight(42)
        self.btn_stop.setMinimumWidth(140)
        self.btn_stop.clicked.connect(self._on_stop)

        self.btn_upload = QPushButton("📁   Upload Audio")
        self.btn_upload.setObjectName("btn_upload")
        self.btn_upload.setMinimumHeight(42)
        self.btn_upload.setMinimumWidth(140)
        self.btn_upload.clicked.connect(self._on_upload)

        self.btn_snapshot = QPushButton("📷   Snapshot")
        self.btn_snapshot.setObjectName("btn_snapshot")
        self.btn_snapshot.setMinimumHeight(42)
        self.btn_snapshot.setMinimumWidth(130)
        self.btn_snapshot.setEnabled(False)
        self.btn_snapshot.clicked.connect(self._on_snapshot)

        ctrl_row.addWidget(self.btn_record)
        ctrl_row.addWidget(self.btn_stop)
        ctrl_row.addWidget(self.btn_upload)
        ctrl_row.addWidget(self.btn_snapshot)
        ctrl_row.addStretch()

        self.lbl_timer = QLabel("00:00")
        self.lbl_timer.setObjectName("lbl_timer")
        self.lbl_timer.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        ctrl_row.addWidget(self.lbl_timer)
        root.addLayout(ctrl_row)
        root.addSpacing(10)

        vu_row = QHBoxLayout()
        vu_lbl = QLabel("INPUT LEVEL")
        vu_lbl.setStyleSheet(
            "font-size:9px;font-weight:700;color:#b0a898;"
            "letter-spacing:1px;min-width:72px;"
        )
        vu_row.addWidget(vu_lbl)
        self.vu_bar = QProgressBar()
        self.vu_bar.setObjectName("vu_meter")
        self.vu_bar.setRange(0, 100)
        self.vu_bar.setValue(0)
        self.vu_bar.setFixedHeight(5)
        self.vu_bar.setTextVisible(False)
        vu_row.addWidget(self.vu_bar)
        root.addLayout(vu_row)
        root.addSpacing(6)

        self.lbl_status = QLabel("Ready to record.")
        self.lbl_status.setStyleSheet(
            "font-size:11px;color:#b0a898;font-style:italic;padding:2px 0;"
        )
        root.addWidget(self.lbl_status)
        root.addSpacing(16)
        root.addWidget(_hline())
        root.addSpacing(12)

        # ── Output panels ──────────────────────────────────────────────────
        lbl3 = QLabel("OUTPUT")
        lbl3.setStyleSheet(
            "font-size:9px;font-weight:800;color:#b0a898;"
            "letter-spacing:2px;text-transform:uppercase;padding-bottom:6px;"
        )
        root.addWidget(lbl3)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(8)

        lg = QGroupBox("LIVE TRANSCRIPT")
        ll = QVBoxLayout(lg)
        ll.setContentsMargins(8, 8, 8, 8)
        self.text_transcript = QTextEdit()
        self.text_transcript.setReadOnly(True)
        self.text_transcript.setPlaceholderText(
            "Transcribed text appears here after recording stops…"
        )
        self.text_transcript.setStyleSheet(
            "font-family:'Noto Nastaliq Urdu','Noto Naskh Arabic','Segoe UI',"
            "'Arial',sans-serif;font-size:12px;"
        )
        ll.addWidget(self.text_transcript)
        splitter.addWidget(lg)

        rg = QGroupBox("SUMMARY  &  ACTION ITEMS")
        rl = QVBoxLayout(rg)
        rl.setContentsMargins(8, 8, 8, 8)
        self.text_summary = QTextEdit()
        self.text_summary.setReadOnly(True)
        self.text_summary.setPlaceholderText(
            "Summary and action items appear here after Summarize & Save…"
        )
        self.text_summary.setStyleSheet(
            "font-family:'Noto Nastaliq Urdu','Noto Naskh Arabic','Segoe UI',"
            "'Arial',sans-serif;font-size:13px;"
            "line-height:1.7;"
        )
        rl.addWidget(self.text_summary)
        splitter.addWidget(rg)

        splitter.setSizes([520, 520])
        root.addWidget(splitter, stretch=1)
        root.addSpacing(14)
        root.addWidget(_hline())
        root.addSpacing(12)

        # ── Bottom action row ──────────────────────────────────────────────
        action_row = QHBoxLayout()
        action_row.setSpacing(10)

        self.btn_summarize = QPushButton("⚙   Summarize & Save")
        self.btn_summarize.setObjectName("btn_summarize")
        self.btn_summarize.setEnabled(False)
        self.btn_summarize.setMinimumHeight(40)
        self.btn_summarize.clicked.connect(self._on_summarize)

        sep = QLabel("|")
        sep.setStyleSheet("color:#ddd6c4;font-size:18px;padding:0 4px;")

        self.btn_pdf  = QPushButton("Export PDF")
        self.btn_docx = QPushButton("Export DOCX")
        self.btn_json = QPushButton("Export JSON")

        self.btn_pdf.setObjectName("btn_export_pdf")
        self.btn_docx.setObjectName("btn_export_docx")
        self.btn_json.setObjectName("btn_export_json")

        for btn in (self.btn_pdf, self.btn_docx, self.btn_json):
            btn.setEnabled(False)
            btn.setMinimumHeight(40)

        self.btn_pdf.clicked.connect(lambda: self._on_export("pdf"))
        self.btn_docx.clicked.connect(lambda: self._on_export("docx"))
        self.btn_json.clicked.connect(lambda: self._on_export("json"))

        action_row.addWidget(self.btn_summarize)
        action_row.addWidget(sep)
        action_row.addWidget(self.btn_pdf)
        action_row.addWidget(self.btn_docx)
        action_row.addWidget(self.btn_json)
        action_row.addStretch()
        root.addLayout(action_row)

    def _populate_devices(self):
        devices = self._audio.list_devices()
        self.combo_device.clear()
        if devices:
            for d in devices:
                self.combo_device.addItem(d["name"], userData=d["index"])
        else:
            self.combo_device.addItem("Default", userData=None)

    def _get_language(self):
        return LANG_MAP.get(self.combo_lang.currentText())

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_record(self):
        title = self.edit_title.text().strip() or "Untitled Meeting"
        lang  = self._get_language()
        self._meeting_id = self._db.create_meeting(title=title, language=lang or "en")
        device_idx = self.combo_device.currentData()
        self._audio.start_recording(device_index=device_idx, on_level=self._update_vu)
        self._elapsed_sec = 0
        self._timer.start(1000)
        self.btn_record.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_upload.setEnabled(False)
        self.btn_snapshot.setEnabled(True)
        self.text_transcript.setReadOnly(True)
        self.text_transcript.clear()
        self.text_summary.clear()
        self._transcript_lines = []
        self._set_status("Recording…  speak into the microphone.")

    def _on_stop(self):
        self._timer.stop()
        self._wav_path = self._audio.stop_recording()
        self.btn_stop.setEnabled(False)
        self.btn_record.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.btn_snapshot.setEnabled(False)
        self.vu_bar.setValue(0)
        if not self._wav_path:
            self._set_status("No audio captured.")
            return
        self._db.update_meeting(
            self._meeting_id,
            audio_path=self._wav_path,
            duration_sec=self._elapsed_sec,
        )
        self._start_transcription()

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio File", "",
            "Audio Files (*.wav *.mp3 *.m4a *.ogg *.flac)",
        )
        if not path:
            return
        title = self.edit_title.text().strip() or "Uploaded Meeting"
        self._meeting_id = self._db.create_meeting(title=title)
        self._set_status("Converting audio file…")
        try:
            self._wav_path = AudioHandler.load_audio_file(path)
        except Exception as exc:
            QMessageBox.critical(self, "File Error", str(exc))
            return
        self._db.update_meeting(self._meeting_id, audio_path=self._wav_path)
        self._start_transcription()

    def _on_snapshot(self):
        if self._meeting_id is None:
            return
        try:
            result = self._snapshots.capture(self._meeting_id)
            self._db.add_snapshot(
                self._meeting_id,
                result["image_path"],
                note=result.get("note", ""),
            )
            self._set_status(f"Snapshot saved: {Path(result['image_path']).name}")
        except CameraNotFoundError as exc:
            QMessageBox.warning(self, "Camera Error", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Snapshot Error", str(exc))

    def _start_transcription(self):
        self._set_status("Transcribing audio — please wait…")
        self.btn_summarize.setEnabled(False)
        lang = self._get_language()
        self._thread = QThread()
        self._worker = TranscribeWorker(self._wav_path, lang)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_transcription_done)
        self._worker.error.connect(self._on_transcription_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_transcription_thread_finished)
        self._thread.start()

    def is_busy(self):
        try:
            return self._thread is not None and self._thread.isRunning()
        except RuntimeError:
            return False

    def _on_transcription_thread_finished(self):
        self._thread = None
        self._worker = None

    def _on_transcription_done(self, result: dict):
        # ── 1. Extract core fields ─────────────────────────────────────────
        self._transcript    = result.get("text", "").strip()
        segments            = list(result.get("segments", []))   # copy — don't mutate original
        lang                = result.get("language", "en")
        self._detected_lang = lang

        # ── 2. Fallback: if Whisper gave no segments but gave full text,
        #       synthesize a single segment so the display is never blank ──
        if not segments and self._transcript:
            segments = [{
                "id":      0,
                "start":   0.0,
                "end":     0.0,
                "text":    self._transcript,
                "speaker": "Speaker 1",
            }]

        # ── 3. Speaker diarization — guard against empty/failed result ─────
        speaker_count = 1
        try:
            diarizer = SpeakerDiarizer()
            diarizer.load(on_status=self._set_status, token=HF_TOKEN)
            diarized = diarizer.assign_speakers(self._wav_path, segments)
            # Only accept the result if it is non-empty and same length
            if diarized and len(diarized) == len(segments):
                segments = diarized
            speaker_count = diarizer.count_speakers(self._wav_path)
        except Exception as exc:
            print(f"[Diarizer] Skipped: {exc}")
            speaker_count = 1

        # ── 4. Format for display ──────────────────────────────────────────
        formatted              = []
        self._transcript_lines = []

        for seg in segments:
            sp    = seg.get("speaker", "Speaker 1")
            txt   = seg.get("text", "").strip()
            start = seg.get("start", 0)
            m, s  = int(start) // 60, int(start) % 60

            if txt:                                   # skip truly empty segments
                formatted.append(f"[{m:02d}:{s:02d}]  {sp}:  {txt}")
                self._transcript_lines.append(txt)

        # Final display: prefer formatted lines; fall back to raw Whisper text
        display = "\n".join(formatted) if formatted else self._transcript

        # ── 5. If still nothing to show, make that explicit ───────────────
        if not display:
            display = "(No speech detected. Try speaking louder or checking your microphone.)"

        self.text_transcript.setPlainText(display)
        self.text_transcript.setReadOnly(False)

        # ── 6. Persist to DB ───────────────────────────────────────────────
        self._db.update_meeting(
            self._meeting_id,
            transcript=self._transcript,
            language=lang,
            speaker_count=speaker_count,
        )

        self.btn_summarize.setEnabled(bool(self._transcript))
        self._set_status(
            "Transcription complete. You can edit text before Summarize & Save."
            if self._transcript else
            "Transcription returned no text. Check microphone or audio file."
        )

    def _on_transcription_error(self, msg):
        QMessageBox.critical(self, "Transcription Error", msg)
        self._set_status("Transcription failed.")

    def _on_summarize(self):
        if not self._transcript:
            return
        self._set_status("Running NLP pipeline…")
        try:
            edited_transcript = self._learn_from_transcript_edits()
            if edited_transcript:
                self._transcript = edited_transcript

            s  = Summarizer()
            sr = s.summarize(self._transcript, language=self._detected_lang)
            self._summary = sr["summary"]

            e  = ActionExtractor()
            er = e.extract(self._transcript, language=self._detected_lang)
            self._action_items = er["action_items"]
            self._decisions    = er["decisions"]

            ed  = EmotionDetector()
            ed.load()
            emo = ed.detect(self._transcript)

            self._db.update_meeting(
                self._meeting_id,
                summary=self._summary,
                decisions=self._decisions,
                emotion_label=emo["dominant_emotion"],
                emotion_score=emo["confidence"],
            )
            self._db.add_action_items(
                self._meeting_id,
                [i["text"] for i in self._action_items],
            )

            lines = ["━━━  SUMMARY  ━━━\n", self._summary]
            if self._decisions:
                lines += ["\n\n━━━  KEY DECISIONS  ━━━"]
                lines += [f"  •  {d}" for d in self._decisions]
            if self._action_items:
                lines += ["\n\n━━━  ACTION ITEMS  ━━━"]
                for item in self._action_items:
                    a = f"  [{item['assignee']}]" if item.get("assignee") else ""
                    lines.append(f"  •  {item['text']}{a}")
            lines.append(f"\n\n━━━  MEETING MOOD  ━━━\n  {emo['meeting_mood']}")
            self.text_summary.setPlainText("\n".join(lines))

            for btn in (self.btn_pdf, self.btn_docx, self.btn_json):
                btn.setEnabled(True)
            self._set_status("Meeting saved.  Ready to export.")
            self.meeting_saved.emit(self._meeting_id)
        except Exception as exc:
            QMessageBox.critical(self, "Processing Error", str(exc))
            self._set_status("Processing failed.")

    def _on_export(self, fmt):
        if self._meeting_id is None:
            return
        meeting = self._db.get_meeting(self._meeting_id)
        if not meeting:
            return
        meeting["action_items"] = self._db.get_action_items(self._meeting_id)
        meeting["snapshots"]    = self._db.get_snapshots(self._meeting_id)
        meeting["decisions"]    = self._decisions
        ext_map = {
            "pdf":  "PDF Files (*.pdf)",
            "docx": "Word Documents (*.docx)",
            "json": "JSON Files (*.json)",
        }
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Export", "", ext_map[fmt]
        )
        if not out_path:
            return
        try:
            if fmt == "pdf":
                self._exporter.export_pdf(meeting, out_path)
            elif fmt == "docx":
                self._exporter.export_docx(meeting, out_path)
            else:
                self._exporter.export_json(meeting, out_path)
            QMessageBox.information(self, "Export Complete", f"Saved to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def _learn_from_transcript_edits(self):
        current = self.text_transcript.toPlainText()
        edited_lines = self._extract_transcript_lines(current)
        if not edited_lines:
            return self._transcript

        if self._transcript_lines and len(edited_lines) == len(self._transcript_lines):
            for original, edited in zip(self._transcript_lines, edited_lines):
                learn_from_correction(original, edited)
        else:
            learn_from_correction(self._transcript, " ".join(edited_lines))

        return " ".join(edited_lines).strip()

    @staticmethod
    def _extract_transcript_lines(text):
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and "]" in line:
                line = line.split("]", 1)[1].strip()
            if ":" in line:
                line = line.split(":", 1)[1].strip()
            lines.append(line)
        return lines

    def _update_vu(self, level):
        from PyQt5.QtCore import QMetaObject, Q_ARG
        QMetaObject.invokeMethod(
            self.vu_bar, "setValue",
            Qt.QueuedConnection,
            Q_ARG(int, int(level * 100)),
        )

    def _tick_timer(self):
        self._elapsed_sec += 1
        m, s = self._elapsed_sec // 60, self._elapsed_sec % 60
        self.lbl_timer.setText(f"{m:02d}:{s:02d}")

    def _set_status(self, msg):
        self.lbl_status.setText(msg)