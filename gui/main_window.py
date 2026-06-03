"""
main_window.py
Main application window for SilentNote.
Loads all tabs and pre-loads AI models in background.
"""
from __future__ import annotations
from typing import Optional
from pathlib import Path

from PyQt5.QtCore    import QThread, QObject, pyqtSignal, QSettings, QSize, QTimer
from PyQt5.QtGui     import QPalette, QColor, QCloseEvent
from PyQt5.QtWidgets import (
    QMainWindow, QStatusBar, QLabel, QApplication, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QStackedWidget, QFrame,
)

from gui.recording_tab import RecordingTab
from gui.dashboard_tab import DashboardTab
from gui.summary_tab   import SummaryTab


class ModelLoader(QObject):
    status = pyqtSignal(str)
    done   = pyqtSignal()

    def run(self) -> None:
        import traceback
        try:
            self.status.emit("Loading Whisper model")
            from modules.transcriber import Transcriber
            Transcriber().load_model()

            self.status.emit("Loading NLP models")
            from modules.summarizer       import Summarizer
            from modules.action_extractor import ActionExtractor
            Summarizer().load()
            ActionExtractor().load()

            self.status.emit("Loading emotion model")
            from modules.emotion_detector import EmotionDetector
            EmotionDetector().load()

            self.status.emit("Checking voice command support")
            from modules.voice_commands import VoiceCommandListener
            _ = VoiceCommandListener()

        except Exception as exc:
            traceback.print_exc()
            self.status.emit(f"Error: {exc}")

        self.done.emit()


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SilentNote — Offline Minutes Generator")
        self.setMinimumSize(QSize(1100, 720))

        self._loader_thread: Optional[QThread] = None
        self._loader: Optional[ModelLoader]    = None

        # For animated loading dots
        self._loading_base_msg: str = ""
        self._loading_dot_count: int = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._animate_loading_dots)

        self._set_warm_palette()
        self._load_stylesheet()
        self._restore_geometry()
        self._build_ui()
        self._start_model_loader()

    def _set_warm_palette(self) -> None:
        """Arctic Minimal palette — keeps QPalette in sync with QSS theme."""
        palette = QPalette()

        # Window / background
        palette.setColor(QPalette.Window,          QColor("#F4F7FB"))
        palette.setColor(QPalette.WindowText,      QColor("#1E3A5F"))

        # Input / surface areas
        palette.setColor(QPalette.Base,            QColor("#FFFFFF"))
        palette.setColor(QPalette.AlternateBase,   QColor("#E2EAF4"))
        palette.setColor(QPalette.Text,            QColor("#1E3A5F"))

        # Buttons — ButtonText was causing invisible text on styled buttons
        palette.setColor(QPalette.Button,          QColor("#E2EAF4"))
        palette.setColor(QPalette.ButtonText,      QColor("#1E3A5F"))
        palette.setColor(QPalette.BrightText,      QColor("#FFFFFF"))

        # Selection
        palette.setColor(QPalette.Highlight,       QColor("#5BA4CF"))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))

        # Tooltips
        palette.setColor(QPalette.ToolTipBase,     QColor("#1E3A5F"))
        palette.setColor(QPalette.ToolTipText,     QColor("#FFFFFF"))

        # Links
        palette.setColor(QPalette.Link,            QColor("#5BA4CF"))
        palette.setColor(QPalette.LinkVisited,     QColor("#2B5280"))

        # Disabled state
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor("#A8C4E0"))
        palette.setColor(QPalette.Disabled, QPalette.Text,       QColor("#A8C4E0"))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#A8C4E0"))
        palette.setColor(QPalette.Disabled, QPalette.Base,       QColor("#F0F4FA"))

        app = QApplication.instance()
        if isinstance(app, QApplication):
            app.setPalette(palette)

    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("app_shell")
        layout = QHBoxLayout(shell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(22, 24, 18, 18)
        sidebar_layout.setSpacing(10)

        self.stack = QStackedWidget()

        self.recording_tab = RecordingTab()
        self.summary_tab   = SummaryTab()
        self.dashboard_tab = DashboardTab()

        self.stack.addWidget(self.dashboard_tab)
        self.stack.addWidget(self.recording_tab)
        self.stack.addWidget(self.summary_tab)

        self.nav_buttons = []
        for index, text in enumerate(("Dashboard", "Record", "Review")):
            btn = QPushButton(text)
            btn.setObjectName("nav_button")
            btn.setCheckable(True)
            btn.setMinimumHeight(42)
            btn.clicked.connect(lambda checked=False, i=index: self._set_page(i))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        local = QLabel("Local-first AI\nOffline transcripts")
        local.setObjectName("sidebar_badge")
        local.setWordWrap(True)
        sidebar_layout.addWidget(local)

        layout.addWidget(sidebar)
        layout.addWidget(self.stack, stretch=1)
        self.setCentralWidget(shell)
        self._set_page(0)

        def on_meeting_saved(mid: int) -> None:
            try:
                self.summary_tab.load_meeting(mid)
                self._set_page(2)
                self.dashboard_tab.refresh()
            except Exception as e:
                self.lbl_status.setText(f"Error loading meeting: {e}")

        self.recording_tab.meeting_saved.connect(on_meeting_saved)
        self.summary_tab.changes_saved.connect(
            lambda _: self.dashboard_tab.refresh()
        )

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("lbl_status")
        self.status_bar.addWidget(self.lbl_status)

        self.lbl_offline = QLabel("  Local-first")
        self.lbl_offline.setObjectName("lbl_offline")
        self.status_bar.addPermanentWidget(self.lbl_offline)

    def _set_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def _load_stylesheet(self) -> None:
        qss_path = Path(__file__).parent / "gui" / "styles.qss"
        if not qss_path.exists():
            qss_path = Path(__file__).parent / "styles.qss"
        app = QApplication.instance()
        if isinstance(app, QApplication) and qss_path.exists():
            with open(qss_path, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())

    def _start_model_loader(self) -> None:
        self._loader_thread = QThread()
        self._loader        = ModelLoader()

        self._loader.moveToThread(self._loader_thread)
        self._loader_thread.started.connect(self._loader.run)
        self._loader.status.connect(self._on_loader_status)
        self._loader.done.connect(self._on_loader_done)
        self._loader.done.connect(self._loader_thread.quit)
        self._loader.done.connect(self._loader.deleteLater)
        self._loader_thread.finished.connect(self._loader_thread.deleteLater)
        self._loader_thread.finished.connect(self._on_loader_thread_finished)

        self._loader_thread.start()

    def _on_loader_status(self, msg: str) -> None:
        self._loading_base_msg = msg
        self._loading_dot_count = 0
        self.lbl_status.setText(msg)
        if not self._dot_timer.isActive():
            self._dot_timer.start(500)

    def _animate_loading_dots(self) -> None:
        self._loading_dot_count = (self._loading_dot_count + 1) % 4
        dots = "." * self._loading_dot_count
        self.lbl_status.setText(f"{self._loading_base_msg}{dots}")

    def _on_loader_done(self) -> None:
        self._dot_timer.stop()
        self.lbl_status.setText("Ready")
        QTimer.singleShot(3000, lambda: self.lbl_status.setText(""))

    def _on_loader_thread_finished(self) -> None:
        self._loader        = None
        self._loader_thread = None

    def _is_loader_running(self) -> bool:
        try:
            return self._loader_thread is not None and self._loader_thread.isRunning()
        except RuntimeError:
            return False

    def _restore_geometry(self) -> None:
        settings = QSettings("SilentNote", "SilentNote")
        geom = settings.value("geometry")
        if geom:
            self.restoreGeometry(geom)
        else:
            self.resize(1280, 820)
            self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geometry = screen.geometry()
        x = (geometry.width()  - self.width())  // 2
        y = (geometry.height() - self.height()) // 2
        self.move(x, y)

    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:  # type: ignore[override]
        if self._is_loader_running():
            self.lbl_status.setText("Still loading models — please wait.")
            if a0 is not None:
                a0.ignore()
            return

        if self.recording_tab.is_busy():
            self.lbl_status.setText("Transcription running — please wait.")
            if a0 is not None:
                a0.ignore()
            return

        self._dot_timer.stop()
        settings = QSettings("SilentNote", "SilentNote")
        settings.setValue("geometry", self.saveGeometry())

        super().closeEvent(a0)