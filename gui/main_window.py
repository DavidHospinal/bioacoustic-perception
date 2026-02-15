"""
Main application window with 3D viewport and control panel.
All heavy processing (audio decoding, feature extraction, AI classification)
runs on background QThreads to keep the UI responsive.
"""
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QStatusBar, QMessageBox, QVBoxLayout
)
from gui.spectral_legend import SpectralLegend
from PyQt6.QtCore import Qt
from gui.control_panel import ControlPanel
from gui.file_loader import validate_audio_file, get_file_info
from visualization.scene_manager import SceneManager
from visualization.point_cloud_renderer import PointCloudRenderer
from visualization.trail_renderer import TrailRenderer
from visualization.animation_controller import AnimationController
from core.audio_analyzer import AudioAnalyzer
from core.audio_player import AudioPlayer
from core.feature_mapper import FeatureMapper
from core.audio_classifier import AudioClassifier
from core.analysis_worker import AnalysisWorker
import config
from datetime import datetime


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sound-Visual: Audio-Visual Analysis")
        self.setMinimumSize(1280, 720)

        self.analyzer = AudioAnalyzer()
        self.player = AudioPlayer()
        self.mapper = FeatureMapper()
        self.classifier = AudioClassifier()
        self.animation = None
        self.features = None
        self._worker = None
        self._current_file_path = None

        self.scene = SceneManager()
        self.point_cloud = PointCloudRenderer(self.scene.get_widget())
        self.trail = TrailRenderer(self.scene.get_widget())

        self._build_layout()
        self._build_menu()
        self._build_status_bar()
        self._apply_dark_theme()

    def _build_layout(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)

        gl_widget = self.scene.get_widget()
        splitter.addWidget(gl_widget)

        # Leyenda espectral superpuesta sobre el viewport 3D
        self.spectral_legend = SpectralLegend(gl_widget)
        self.spectral_legend.setGeometry(0, 0, 90, 500)
        self.spectral_legend.show()

        self.controls = ControlPanel()
        splitter.addWidget(self.controls)
        splitter.setSizes([960, 300])

        self.setCentralWidget(splitter)

        self.controls.file_selected.connect(self._on_file_loaded)
        self.controls.play_clicked.connect(self._on_play)
        self.controls.pause_clicked.connect(self._on_pause)
        self.controls.stop_clicked.connect(self._on_stop)
        self.controls.trail_length_changed.connect(self._on_trail_length_changed)
        self.controls.point_scale_changed.connect(self._on_point_scale_changed)

    # ------------------------------------------------------------------ #
    #  File Loading (background thread)                                  #
    # ------------------------------------------------------------------ #

    def _on_file_loaded(self, file_path):
        """Handles audio file selection. Starts background analysis."""
        if not validate_audio_file(file_path):
            QMessageBox.warning(
                self, "Error",
                "Unsupported file format.\n"
                "Valid formats: MP3, WAV, FLAC, OGG")
            return

        self._on_stop()
        self._current_file_path = file_path
        self._cancel_worker()

        # Prepare worker
        worker = AnalysisWorker(AnalysisWorker.MODE_LOAD)
        worker.file_path = file_path
        worker.analyzer = self.analyzer
        worker.mapper = self.mapper
        worker.progress.connect(self._on_worker_progress)
        worker.load_complete.connect(self._on_load_complete)
        worker.error.connect(self._on_worker_error)

        self._worker = worker
        self._set_controls_enabled(False)
        worker.start()

    def _on_load_complete(self, result):
        """Called when background loading/analysis finishes."""
        self._set_controls_enabled(True)
        self.features = result["features"]

        # Show cached classification if available
        cached_class = result.get("classification")
        if cached_class:
            self.controls.set_classification(cached_class)
        else:
            self.controls.set_classification(None)

        self.player.load(
            self.analyzer.get_raw_audio(),
            self.analyzer.get_sample_rate())

        self.controls.set_duration(self.features["duration"])

        file_info = get_file_info(self._current_file_path)
        now_str = datetime.now().strftime('%d/%m/%Y %H:%M')
        from_cache = result.get("from_cache", False)
        info_text = (
            f"Developer: Msc. David Hospinal\n"
            f"Analysis: {now_str}\n"
            f"\n"
            f"Duration: {self.features['duration']:.1f}s\n"
            f"Frames: {self.features['n_frames']}\n"
            f"SR: {self.analyzer.get_sample_rate()} Hz\n"
            f"Hop: {config.HOP_LENGTH}"
        )
        if from_cache:
            info_text += "\n(Loaded from cache)"
        self.controls.set_analysis_info(info_text)

        self.scene.reset_camera()
        self.animation = AnimationController(
            self.player, self.mapper, self.features,
            self.point_cloud, self.trail, self.scene)
        self.animation.frame_updated.connect(
            self.controls.update_time_display)
        self.animation.playback_finished.connect(self._on_playback_finished)

        self.statusBar().showMessage(
            f"Loaded: {file_info['name']} | "
            f"Duration: {self.features['duration']:.1f}s | "
            f"Frames: {self.features['n_frames']}")

    # ------------------------------------------------------------------ #
    #  Playback Controls                                                 #
    # ------------------------------------------------------------------ #

    def _on_play(self):
        if self.animation is None:
            return
        if self.player.is_paused:
            self.player.resume()
        else:
            self.player.play()
        self.animation.start()
        self.statusBar().showMessage("Playing...")

    def _on_pause(self):
        if self.animation:
            self.player.pause()
            self.animation.stop()
            self.statusBar().showMessage("Paused")

    def _on_stop(self):
        if self.animation:
            self.animation.stop()
        self.player.stop()
        self.point_cloud.clear()
        self.trail.clear()
        self.scene.set_playing(False)
        self.controls.update_time_display(0.0)
        self.statusBar().showMessage("Stopped")

    # ------------------------------------------------------------------ #
    #  Classification (background thread)                                #
    # ------------------------------------------------------------------ #

    def _on_playback_finished(self):
        """Playback ended. Start AI classification on background thread."""
        self._cancel_worker()

        worker = AnalysisWorker(AnalysisWorker.MODE_CLASSIFY)
        worker.file_path = self._current_file_path
        worker.analyzer = self.analyzer
        worker.classifier = self.classifier
        worker.features = self.features
        worker.progress.connect(self._on_worker_progress)
        worker.classify_complete.connect(self._on_classify_complete)
        worker.error.connect(self._on_worker_error)

        self._worker = worker
        worker.start()

    def _on_classify_complete(self, result):
        """Called when background classification finishes."""
        self.controls.set_classification(result)
        self.statusBar().showMessage(
            f"Classification: {result['label']} "
            f"({result['confidence']:.0%})")

    # ------------------------------------------------------------------ #
    #  Worker Management                                                 #
    # ------------------------------------------------------------------ #

    def _on_worker_progress(self, message, percent):
        """Update status bar with worker progress."""
        self.statusBar().showMessage(f"{message} ({percent}%)")

    def _on_worker_error(self, error_msg):
        """Handle worker errors."""
        self._set_controls_enabled(True)
        QMessageBox.critical(
            self, "Processing Error",
            f"Could not process audio:\n{error_msg}")
        self.statusBar().showMessage("Processing error.")

    def _cancel_worker(self):
        """Safely disconnect and discard any running worker."""
        if self._worker is not None and self._worker.isRunning():
            try:
                self._worker.progress.disconnect()
                self._worker.load_complete.disconnect()
                self._worker.classify_complete.disconnect()
                self._worker.error.disconnect()
            except (TypeError, RuntimeError):
                pass
        self._worker = None

    def _set_controls_enabled(self, enabled):
        """Enable or disable interactive controls during processing."""
        self.controls.play_btn.setEnabled(enabled)
        self.controls.pause_btn.setEnabled(enabled)
        self.controls.stop_btn.setEnabled(enabled)
        self.controls.open_btn.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    #  Visualization Parameters                                          #
    # ------------------------------------------------------------------ #

    def _on_trail_length_changed(self, value):
        if self.animation:
            self.animation.set_trail_length(value)

    def _on_point_scale_changed(self, value):
        config.POINT_SIZE_SCALE = float(value)
        if self.features is not None:
            self.mapper.map_features(self.features)

    # ------------------------------------------------------------------ #
    #  Theme, Menu, Status Bar                                           #
    # ------------------------------------------------------------------ #

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QWidget { background-color: #1a1a1a; color: #cccccc; }
            QMenuBar {
                background-color: #252525;
                color: #cccccc;
                border-bottom: 1px solid #333333;
            }
            QMenuBar::item:selected { background-color: #3a3a3a; }
            QMenu {
                background-color: #252525;
                color: #cccccc;
                border: 1px solid #444444;
            }
            QMenu::item:selected { background-color: #3a3a3a; }
            QPushButton {
                background-color: #333333;
                border: 1px solid #555555;
                padding: 8px 16px;
                color: #ffffff;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #444444; }
            QPushButton:pressed { background-color: #222222; }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #555555;
                border: 1px solid #3a3a3a;
            }
            QSlider::groove:horizontal {
                background: #333333;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ccff;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QLabel { color: #aaaaaa; }
            QGroupBox {
                border: 1px solid #444444;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 18px;
                color: #cccccc;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                padding: 0 6px;
            }
            QStatusBar {
                background-color: #252525;
                color: #888888;
                border-top: 1px solid #333333;
            }
        """)

    def _build_menu(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("&Open Audio...", self.controls.open_file_dialog)
        file_menu.addSeparator()
        file_menu.addAction("&Exit", self.close)

        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction("&Reset Camera", self.scene.reset_camera)

    def _build_status_bar(self):
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage(
            "Ready. Open an audio file to begin.")

    def closeEvent(self, event):
        """Clean up resources on window close."""
        self._cancel_worker()
        if self.animation:
            self.animation.stop()
        self.player.stop()
        event.accept()

    def resizeEvent(self, event):
        """Resize the spectral legend with the viewport."""
        super().resizeEvent(event)
        gl_widget = self.scene.get_widget()
        legend_h = min(gl_widget.height() - 20, 600)
        self.spectral_legend.setGeometry(
            4, 10, 90, max(legend_h, 100))
