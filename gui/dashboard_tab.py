from pathlib import Path
from typing  import Optional

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtGui     import QPixmap, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QListWidget, QListWidgetItem, QLineEdit, QGroupBox, QSplitter,
    QScrollArea, QGridLayout, QMessageBox, QFileDialog, QTabWidget,
    QSizePolicy,
)

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False

USER_ROLE = getattr(Qt, "UserRole", 0x0100)

from database.db_manager import DBManager
from modules.exporter    import Exporter
from modules.snapshot_handler import SnapshotHandler


# ── Stat card widget ──────────────────────────────────────────────────────────

class StatCard(QGroupBox):
    """A small card showing a single metric (label + value)."""

    def __init__(self, title: str, value: str = "—", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 20, 8, 8)
        layout.setSpacing(4)

        self.lbl_value = QLabel(value)
        self.lbl_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_value.setStyleSheet(
            "font-size: 26px; font-weight: bold; color: #1E3A5F;"
        )

        self.lbl_title = QLabel(title)
        self.lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_title.setStyleSheet("font-size: 11px; color: #6B8DAF;")

        layout.addWidget(self.lbl_value)
        layout.addWidget(self.lbl_title)
        self.setFixedHeight(90)

    def set_value(self, v: str) -> None:
        self.lbl_value.setText(v)


# ── DashboardTab ──────────────────────────────────────────────────────────────

class DashboardTab(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db        = DBManager()
        self._exporter  = Exporter()
        self._selected_meeting_id: Optional[int] = None

        self._build_ui()
        self.refresh()

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(24, 24, 24, 24)

        # ── Stat cards row ────────────────────────────────────────────────
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        cards_row.setContentsMargins(0, 0, 0, 0)
        self.card_total     = StatCard("Total Meetings")
        self.card_duration  = StatCard("Total Duration")
        self.card_avg       = StatCard("Avg Duration")
        self.card_snaps     = StatCard("Snapshots")
        self.card_actions   = StatCard("Action Items")
        for card in (self.card_total, self.card_duration,
                     self.card_avg, self.card_snaps, self.card_actions):
            cards_row.addWidget(card)
        root.addLayout(cards_row)

        # ── Main splitter ─────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left panel: meeting list ───────────────────────────────────────
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        search_row = QHBoxLayout()
        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("Search meetings…")
        self.edit_search.textChanged.connect(self._on_search)

        btn_refresh = QPushButton("↺  Refresh")
        btn_refresh.setObjectName("btn_refresh")
        btn_refresh.setFixedWidth(100)
        btn_refresh.setMinimumHeight(36)
        btn_refresh.clicked.connect(self.refresh)

        search_row.addWidget(self.edit_search)
        search_row.addWidget(btn_refresh)
        left_layout.addLayout(search_row)

        self.meeting_list = QListWidget()
        self.meeting_list.currentItemChanged.connect(self._on_meeting_selected)
        left_layout.addWidget(self.meeting_list)

        btn_delete = QPushButton("🗑  Delete Selected")
        btn_delete.setObjectName("btn_delete")
        btn_delete.setFixedWidth(100)
        btn_delete.setMinimumHeight(38)
        btn_delete.clicked.connect(self._on_delete)
        left_layout.addWidget(btn_delete)

        splitter.addWidget(left)

        # ── Right panel: detail view ───────────────────────────────────────
        right = QTabWidget()

        self._build_overview_tab(right)

        self.txt_transcript = QTextEdit()
        self.txt_transcript.setReadOnly(True)
        self.txt_transcript.setPlaceholderText("Select a meeting to view the transcript…")
        right.addTab(self.txt_transcript, "Transcript")

        self._build_snapshots_tab(right)
        self._build_charts_tab(right)

        splitter.addWidget(right)
        splitter.setSizes([300, 700])
        root.addWidget(splitter, stretch=1)

    def _build_overview_tab(self, tabs: QTabWidget) -> None:
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.lbl_meeting_title = QLabel("No meeting selected")
        self.lbl_meeting_title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #1E3A5F;"
        )
        layout.addWidget(self.lbl_meeting_title)

        self.lbl_meta = QLabel("")
        self.lbl_meta.setWordWrap(True)
        self.lbl_meta.setStyleSheet("color: #6B8DAF; font-size: 12px;")
        layout.addWidget(self.lbl_meta)

        lbl_summary = QLabel("Summary:")
        lbl_summary.setStyleSheet("font-weight: 600; color: #1E3A5F;")
        layout.addWidget(lbl_summary)
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setMaximumHeight(140)
        layout.addWidget(self.txt_summary)

        lbl_actions = QLabel("Action Items:")
        lbl_actions.setStyleSheet("font-weight: 600; color: #1E3A5F;")
        layout.addWidget(lbl_actions)
        self.txt_actions = QTextEdit()
        self.txt_actions.setReadOnly(True)
        self.txt_actions.setMaximumHeight(120)
        layout.addWidget(self.txt_actions)

        lbl_emotion = QLabel("Emotion:")
        lbl_emotion.setStyleSheet("font-weight: 600; color: #1E3A5F;")
        layout.addWidget(lbl_emotion)
        self.lbl_emotion = QLabel("—")
        self.lbl_emotion.setStyleSheet("font-size: 13px; color: #5BA4CF;")
        layout.addWidget(self.lbl_emotion)

        layout.addStretch()
        tabs.addTab(w, "Overview")

    def _build_snapshots_tab(self, tabs: QTabWidget) -> None:
        w      = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        self.lbl_snap_info = QLabel("Select a meeting to view its snapshots.")
        self.lbl_snap_info.setStyleSheet("color: #6B8DAF;")
        layout.addWidget(self.lbl_snap_info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.snap_container = QWidget()
        self.snap_grid      = QGridLayout(self.snap_container)
        scroll.setWidget(self.snap_container)
        layout.addWidget(scroll)

        tabs.addTab(w, "Snapshots")

    def _build_charts_tab(self, tabs: QTabWidget) -> None:
        w      = QWidget()
        layout = QVBoxLayout(w)

        if not HAS_PYQTGRAPH:
            layout.addWidget(QLabel(
                "Install pyqtgraph to view charts:\n  pip install pyqtgraph"
            ))
            tabs.addTab(w, "Charts")
            return

        pg.setConfigOption("background", "#FFFFFF")
        pg.setConfigOption("foreground", "#1E3A5F")

        lbl_emo = QLabel("Emotion Distribution (all meetings):")
        lbl_emo.setStyleSheet("font-weight: 600; color: #1E3A5F;")
        layout.addWidget(lbl_emo)
        self.emotion_chart = pg.PlotWidget()
        self.emotion_chart.setFixedHeight(200)
        self.emotion_chart.showGrid(x=False, y=True)
        layout.addWidget(self.emotion_chart)

        lbl_mpd = QLabel("Meetings per Day (last 30 days):")
        lbl_mpd.setStyleSheet("font-weight: 600; color: #1E3A5F;")
        layout.addWidget(lbl_mpd)
        self.mpd_chart = pg.PlotWidget()
        self.mpd_chart.setFixedHeight(200)
        self.mpd_chart.showGrid(x=False, y=True)
        layout.addWidget(self.mpd_chart)

        layout.addStretch()
        tabs.addTab(w, "Charts")

    # ── Refresh ───────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        self._refresh_stats()
        self._refresh_meeting_list()
        if HAS_PYQTGRAPH:
            self._refresh_charts()

    def _refresh_stats(self) -> None:
        stats = self._db.get_stats()
        self.card_total.set_value(str(stats["total_meetings"]))
        total_s = stats["total_duration_sec"]
        self.card_duration.set_value(self._fmt_dur(total_s))
        avg_s = stats["avg_duration_sec"]
        self.card_avg.set_value(self._fmt_dur(avg_s))
        self.card_snaps.set_value(str(stats["total_snapshots"]))
        self.card_actions.set_value(str(stats["total_action_items"]))

    def _refresh_meeting_list(self, search: str = "") -> None:
        meetings = self._db.get_all_meetings(search=search)
        self.meeting_list.clear()
        for m in meetings:
            date = (m.get("started_at") or "")[:10]
            item = QListWidgetItem(f"{m['title']}  [{date}]")
            item.setData(USER_ROLE, m["id"])
            self.meeting_list.addItem(item)

    def _refresh_charts(self) -> None:
        stats = self._db.get_stats()

        self.emotion_chart.clear()
        emo_dist = stats.get("emotion_distribution", {})
        if emo_dist:
            labels = list(emo_dist.keys())
            values = [emo_dist[k] for k in labels]
            x      = list(range(len(labels)))
            bars   = pg.BarGraphItem(x=x, height=values, width=0.6,
                                     brush="#5BA4CF")
            self.emotion_chart.addItem(bars)
            ax = self.emotion_chart.getAxis("bottom")
            ax.setTicks([[(i, l) for i, l in enumerate(labels)]])

        self.mpd_chart.clear()
        mpd = stats.get("meetings_per_day", [])
        if mpd:
            mpd_sorted = sorted(mpd, key=lambda d: d["date"])
            values     = [d["count"] for d in mpd_sorted]
            labels     = [d["date"][-5:] for d in mpd_sorted]
            x          = list(range(len(labels)))
            bars       = pg.BarGraphItem(x=x, height=values, width=0.6,
                                         brush="#2B5280")
            self.mpd_chart.addItem(bars)
            ax = self.mpd_chart.getAxis("bottom")
            ax.setTicks([[(i, l) for i, l in enumerate(labels)]])

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_search(self, text: str) -> None:
        self._refresh_meeting_list(search=text)

    def _on_meeting_selected(self, current: QListWidgetItem,
                              previous: QListWidgetItem) -> None:
        if current is None:
            return
        meeting_id = current.data(USER_ROLE)
        self._load_meeting_detail(meeting_id)

    def _on_delete(self) -> None:
        item = self.meeting_list.currentItem()
        if item is None:
            return
        meeting_id = item.data(USER_ROLE)
        reply = QMessageBox.question(
            self, "Confirm Delete",
            "Permanently delete this meeting and all its data?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            meeting = self._db.get_meeting(meeting_id) or {}
            snapshots = self._db.get_snapshots(meeting_id)
            self._db.delete_meeting(meeting_id)
            self._delete_meeting_files(meeting, snapshots)
            self._selected_meeting_id = None
            self.refresh()

    def _delete_meeting_files(self, meeting: dict, snapshots: list[dict]) -> None:
        audio_path = meeting.get("audio_path")
        if audio_path:
            self._delete_project_file(audio_path, "recordings")

        for snap in snapshots:
            image_path = snap.get("image_path")
            if not image_path:
                continue
            if self._is_inside_data_dir(image_path, "snapshots"):
                SnapshotHandler.delete_snapshot(image_path)

    @staticmethod
    def _is_inside_data_dir(file_path: str, data_subdir: str) -> bool:
        try:
            root = (Path(__file__).resolve().parent.parent / "data" / data_subdir).resolve()
            target = Path(file_path).resolve()
            return target == root or root in target.parents
        except Exception:
            return False

    @classmethod
    def _delete_project_file(cls, file_path: str, data_subdir: str) -> None:
        if not cls._is_inside_data_dir(file_path, data_subdir):
            return
        try:
            target = Path(file_path)
            if target.exists() and target.is_file():
                target.unlink()
        except OSError:
            pass

    def _on_export(self, fmt: str) -> None:
        mid = self._selected_meeting_id
        if mid is None:
            return
        meeting = self._db.get_meeting(mid)
        if not meeting:
            return
        meeting["action_items"] = self._db.get_action_items(mid)
        meeting["snapshots"]    = self._db.get_snapshots(mid)

        ext_map = {"pdf": "PDF Files (*.pdf)",
                   "docx": "Word Documents (*.docx)",
                   "json": "JSON Files (*.json)"}
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
            QMessageBox.information(self, "Done", f"Saved to:\n{out_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    # ── Detail loader ─────────────────────────────────────────────────────────

    def _load_meeting_detail(self, meeting_id: int) -> None:
        self._selected_meeting_id = meeting_id
        meeting = self._db.get_meeting(meeting_id)
        if not meeting:
            return

        self.lbl_meeting_title.setText(meeting.get("title", "—"))

        date     = (meeting.get("started_at") or "")[:10]
        duration = self._fmt_dur(meeting.get("duration_sec", 0))
        lang     = (meeting.get("language") or "—").upper()
        speakers = meeting.get("speaker_count", "—")
        self.lbl_meta.setText(
            f"Date: {date}  |  Duration: {duration}  |  "
            f"Language: {lang}  |  Speakers: {speakers}"
        )

        self.txt_summary.setPlainText(meeting.get("summary") or "")

        actions = self._db.get_action_items(meeting_id)
        action_text = "\n".join(
            f"{'[✓]' if a.get('done') else '[ ]'}  {a['item']}"
            for a in actions
        ) or "No action items."
        self.txt_actions.setPlainText(action_text)

        emotion = meeting.get("emotion_label") or "—"
        score   = meeting.get("emotion_score") or 0
        self.lbl_emotion.setText(
            f"{emotion.capitalize()}  ({round(float(score) * 100)}% confidence)"
        )

        self.txt_transcript.setPlainText(meeting.get("transcript") or "")
        self._load_snapshots(meeting_id)

    def _load_snapshots(self, meeting_id: int) -> None:
        while self.snap_grid.count():
            item = self.snap_grid.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        snaps = self._db.get_snapshots(meeting_id)
        if not snaps:
            self.lbl_snap_info.setText("No snapshots for this meeting.")
            return

        self.lbl_snap_info.setText(f"{len(snaps)} snapshot(s)")

        for idx, snap in enumerate(snaps):
            col = idx % 3
            row = idx // 3

            thumb_path    = Path(snap["image_path"])
            thumb_variant = thumb_path.parent / (thumb_path.stem + "_thumb" + thumb_path.suffix)
            display_path  = str(thumb_variant) if thumb_variant.exists() else str(thumb_path)

            container = QWidget()
            c_layout  = QVBoxLayout(container)
            c_layout.setSpacing(2)

            lbl_img = QLabel()
            px      = QPixmap(display_path)
            if not px.isNull():
                lbl_img.setPixmap(px.scaled(
                    200, 112,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                ))
            else:
                lbl_img.setText("[Image not found]")
                lbl_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

            c_layout.addWidget(lbl_img)

            date_str = (snap.get("captured_at") or "")[:16].replace("T", " ")
            lbl_date = QLabel(date_str)
            lbl_date.setStyleSheet("font-size: 10px; color: #6B8DAF;")
            lbl_date.setAlignment(Qt.AlignmentFlag.AlignCenter)
            c_layout.addWidget(lbl_date)

            self.snap_grid.addWidget(container, row, col)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_dur(seconds) -> str:  # type: ignore[annotation-unchecked]
        try:
            s = int(seconds or 0)
            h = s // 3600
            m = (s % 3600) // 60
            if h:
                return f"{h}h {m}m"
            return f"{m}m {s % 60}s"
        except Exception:
            return "—"
