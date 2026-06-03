"""
gui/summary_tab.py
──────────────────────────────────────────────────────────────────────────────
Dedicated tab for reviewing and editing meeting summaries and action items
after transcription is complete.

Features
────────
  • Editable summary text box (user can refine the auto-generated summary)
  • Action items table with checkboxes (mark items as done)
  • Add / delete action item rows manually
  • Decisions list (editable)
  • Save edits back to the database
  • Re-run summarization on the current transcript with a different ratio
──────────────────────────────────────────────────────────────────────────────
"""

from typing import Optional, Any, cast

from PyQt5.QtCore    import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox, QLineEdit,
    QGroupBox, QSplitter, QMessageBox, QDoubleSpinBox, QAbstractItemView,
)

from database.db_manager      import DBManager
from modules.summarizer        import Summarizer
from modules.action_extractor  import ActionExtractor


class SummaryTab(QWidget):
    """
    Lets the user review and manually edit the auto-generated summary,
    action items, and decisions for the most recently processed meeting.

    Signals
    ───────
    changes_saved(int)  – emitted with meeting_id after DB save
    """

    changes_saved = pyqtSignal(int)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._db         = DBManager()
        self._meeting_id: Optional[int] = None
        self._transcript: str           = ""
        self._language: str             = "en"

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(16, 16, 16, 16)

        # ── Top bar ───────────────────────────────────────────────────────
        top_row = QHBoxLayout()

        self.lbl_meeting = QLabel("No meeting loaded.")
        self.lbl_meeting.setStyleSheet("font-size:15px; font-weight:bold; color:#e94560;")
        top_row.addWidget(self.lbl_meeting)
        top_row.addStretch()

        top_row.addWidget(QLabel("Summary ratio:"))
        self.spin_ratio = QDoubleSpinBox()
        self.spin_ratio.setRange(0.1, 0.9)
        self.spin_ratio.setSingleStep(0.05)
        self.spin_ratio.setValue(0.25)
        self.spin_ratio.setFixedWidth(70)
        self.spin_ratio.setToolTip(
            "Fraction of sentences to keep in the summary (0.1 = brief, 0.9 = detailed)"
        )
        top_row.addWidget(self.spin_ratio)

        self.btn_rerun = QPushButton("Re-summarize")
        self.btn_rerun.setEnabled(False)
        self.btn_rerun.setToolTip("Re-run NLP with the selected ratio")
        self.btn_rerun.clicked.connect(self._on_rerun)
        top_row.addWidget(self.btn_rerun)

        self.btn_save = QPushButton("Save Changes")
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._on_save)
        top_row.addWidget(self.btn_save)

        root.addLayout(top_row)

        # ── Main splitter ─────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)  # FIX: Qt.Vertical -> Qt.Orientation.Vertical

        # ── Summary box ───────────────────────────────────────────────────
        sum_group = QGroupBox("Summary  (editable)")
        sum_layout = QVBoxLayout(sum_group)
        self.txt_summary = QTextEdit()
        self.txt_summary.setPlaceholderText(
            "Auto-generated summary appears here. You can edit it freely."
        )
        sum_layout.addWidget(self.txt_summary)
        splitter.addWidget(sum_group)

        # ── Decisions box ─────────────────────────────────────────────────
        dec_group = QGroupBox("Key Decisions  (editable, one per line)")
        dec_layout = QVBoxLayout(dec_group)
        self.txt_decisions = QTextEdit()
        self.txt_decisions.setPlaceholderText(
            "Decisions extracted from the meeting. Edit or add your own."
        )
        self.txt_decisions.setMaximumHeight(130)
        dec_layout.addWidget(self.txt_decisions)
        splitter.addWidget(dec_group)

        # ── Action items table ────────────────────────────────────────────
        act_group = QGroupBox("Action Items")
        act_layout = QVBoxLayout(act_group)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Done", "Action Item", "Assignee", "Due Date"])

        # FIX: horizontalHeader() can return None — guard with assert
        h_header = self.table.horizontalHeader()
        assert h_header is not None
        h_header.setSectionResizeMode(1, QHeaderView.Stretch)
        h_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h_header.setSectionResizeMode(2, QHeaderView.Interactive)
        h_header.setSectionResizeMode(3, QHeaderView.Interactive)

        self.table.setColumnWidth(2, 140)
        self.table.setColumnWidth(3, 110)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked
        )

        # FIX: verticalHeader() can return None — guard with assert
        v_header = self.table.verticalHeader()
        assert v_header is not None
        v_header.setVisible(False)

        act_layout.addWidget(self.table)

        row_ctrl = QHBoxLayout()
        self.edit_new_item = QLineEdit()
        self.edit_new_item.setPlaceholderText("Type a new action item and press Add…")
        row_ctrl.addWidget(self.edit_new_item, stretch=1)

        btn_add = QPushButton("+ Add")
        btn_add.setFixedWidth(70)
        btn_add.clicked.connect(self._on_add_row)
        row_ctrl.addWidget(btn_add)

        btn_del = QPushButton("Remove")
        btn_del.setFixedWidth(90)
        btn_del.clicked.connect(self._on_delete_row)
        row_ctrl.addWidget(btn_del)

        act_layout.addLayout(row_ctrl)
        splitter.addWidget(act_group)

        splitter.setSizes([200, 120, 300])
        root.addWidget(splitter, stretch=1)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("label_status")
        root.addWidget(self.lbl_status)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_meeting(self, meeting_id: int) -> None:
        self._meeting_id = meeting_id
        meeting = self._db.get_meeting(meeting_id)
        if not meeting:
            return

        self._transcript = meeting.get("transcript") or ""
        self._language   = meeting.get("language") or "en"
        self.lbl_meeting.setText(meeting.get("title", "Untitled Meeting"))

        self.txt_summary.setPlainText(meeting.get("summary") or "")

        decisions = meeting.get("decisions") or []
        self.txt_decisions.setPlainText("\n".join(decisions))

        self._load_action_table(meeting_id)

        self.btn_save.setEnabled(True)
        self.btn_rerun.setEnabled(bool(self._transcript))
        self._set_status(f"Loaded: {meeting.get('title', '')}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _load_action_table(self, meeting_id: int) -> None:
        self.table.setRowCount(0)
        items = self._db.get_action_items(meeting_id)
        for item in items:
            self._append_table_row(
                db_id    = item["id"],
                text     = item.get("item", ""),
                assignee = item.get("assignee", ""),
                due_date = item.get("due_date", ""),
                done     = bool(item.get("done", False)),
            )

    def _append_table_row(self, db_id: int = -1, text: str = "",
                          assignee: str = "", due_date: str = "",
                          done: bool = False) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Column 0: Done checkbox (centred)
        chk_widget = QWidget()
        chk_layout = QHBoxLayout(chk_widget)
        chk_layout.setContentsMargins(0, 0, 0, 0)
        chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # FIX: use AlignmentFlag enum
        chk = QCheckBox()
        chk.setChecked(done)
        chk_layout.addWidget(chk)
        self.table.setCellWidget(row, 0, chk_widget)

        # Column 1: Text
        item_text = QTableWidgetItem(text)
        item_text.setData(Qt.ItemDataRole.UserRole, db_id)  # FIX: use ItemDataRole enum
        self.table.setItem(row, 1, item_text)

        # Column 2: Assignee
        self.table.setItem(row, 2, QTableWidgetItem(assignee))

        # Column 3: Due date
        self.table.setItem(row, 3, QTableWidgetItem(due_date))

    # ── Event handlers ────────────────────────────────────────────────────────

    def _on_add_row(self) -> None:
        text = self.edit_new_item.text().strip()
        if not text:
            return
        self._append_table_row(text=text)
        self.edit_new_item.clear()

    def _on_delete_row(self) -> None:
        rows = sorted(
            set(idx.row() for idx in self.table.selectedIndexes()),
            reverse=True,
        )
        for row in rows:
            self.table.removeRow(row)

    def _on_rerun(self) -> None:
        if not self._transcript:
            return

        self._set_status("Re-running summarization…")
        ratio = self.spin_ratio.value()

        try:
            summarizer = Summarizer()
            result     = summarizer.summarize(
                self._transcript,
                ratio=ratio,
                language=self._language,
            )
            self.txt_summary.setPlainText(result["summary"])

            extractor  = ActionExtractor()
            ext_result = extractor.extract(
                self._transcript,
                language=self._language,
            )

            self.table.setRowCount(0)
            for item in ext_result["action_items"]:
                self._append_table_row(
                    text     = item.get("text", ""),
                    assignee = item.get("assignee", ""),
                )

            decisions = ext_result.get("decisions", [])
            self.txt_decisions.setPlainText("\n".join(decisions))

            self._set_status("Re-summarization complete. Click Save to store changes.")

        except Exception as exc:
            QMessageBox.critical(self, "Error", str(exc))
            self._set_status("Re-summarization failed.")

    def _on_save(self) -> None:
        if self._meeting_id is None:
            return

        summary   = self.txt_summary.toPlainText().strip()
        decisions = self.txt_decisions.toPlainText().strip()

        decision_items = [
            line.strip() for line in decisions.splitlines()
            if line.strip()
        ]
        self._db.update_meeting(
            self._meeting_id,
            summary=summary,
            decisions=decision_items,
        )

        self._db.delete_action_items(self._meeting_id)

        new_items: list[str | dict[Any, Any]] = []
        for row in range(self.table.rowCount()):
            text_item = self.table.item(row, 1)
            text      = text_item.text().strip() if text_item else ""
            if not text:
                continue

            chk_widget = self.table.cellWidget(row, 0)
            done = False
            if chk_widget:
                chk  = chk_widget.findChild(QCheckBox)
                done = chk.isChecked() if chk else False

            assignee_item = self.table.item(row, 2)
            assignee      = assignee_item.text().strip() if assignee_item else ""

            due_date_item = self.table.item(row, 3)
            due_date      = due_date_item.text().strip() if due_date_item else ""

            new_items.append(cast(dict[Any, Any], {
                "text": text,
                "assignee": assignee,
                "due_date": due_date,
                "done": done,
            }))

        if new_items:
            self._db.add_action_items(
                self._meeting_id,
                new_items,
            )

        self._set_status("Changes saved successfully.")
        self.changes_saved.emit(self._meeting_id)

    def _set_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)