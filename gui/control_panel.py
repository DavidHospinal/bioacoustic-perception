"""
Side panel with playback controls, parameter sliders,
IA classification with confidence bar, and audio information.
Content wrapped in QScrollArea for full access.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QPushButton, QSlider, QLabel,
    QGroupBox, QHBoxLayout, QFileDialog, QProgressBar, QGridLayout,
    QFrame, QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal
from gui.file_loader import get_audio_filter_string
import config


class ControlPanel(QWidget):

    file_selected = pyqtSignal(str)
    play_clicked = pyqtSignal()
    pause_clicked = pyqtSignal()
    stop_clicked = pyqtSignal()
    trail_length_changed = pyqtSignal(int)
    point_scale_changed = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setFixedWidth(300)
        self._duration = 0.0

        # Layout externo: ScrollArea que envuelve TODO el contenido
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #444444;
                min-height: 30px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555555;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Widget interior con todo el contenido
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # File section
        file_group = QGroupBox("Audio File")
        file_layout = QVBoxLayout()
        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.open_btn = QPushButton("Open File...")
        self.open_btn.clicked.connect(self.open_file_dialog)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.open_btn)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Playback controls
        playback_group = QGroupBox("Playback")
        pb_layout = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.play_btn.clicked.connect(self.play_clicked.emit)
        self.pause_btn.clicked.connect(self.pause_clicked.emit)
        self.stop_btn.clicked.connect(self.stop_clicked.emit)
        for btn in [self.play_btn, self.pause_btn, self.stop_btn]:
            btn.setMinimumHeight(36)
        pb_layout.addWidget(self.play_btn)
        pb_layout.addWidget(self.pause_btn)
        pb_layout.addWidget(self.stop_btn)
        playback_group.setLayout(pb_layout)
        layout.addWidget(playback_group)

        # Time display
        time_group = QGroupBox("Time")
        time_layout = QVBoxLayout()
        self.time_label = QLabel("00:00.00 / 00:00.00")
        self.time_label.setStyleSheet(
            "font-size: 18px; font-family: monospace; color: #00ccff;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_layout.addWidget(self.time_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #222222;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background-color: #00ccff;
                border-radius: 4px;
            }
        """)
        time_layout.addWidget(self.progress_bar)
        time_group.setLayout(time_layout)
        layout.addWidget(time_group)

        # Visualization parameters
        viz_group = QGroupBox("Visualization Parameters")
        viz_layout = QVBoxLayout()

        viz_layout.addWidget(QLabel("Trail Length"))
        self.trail_slider = QSlider(Qt.Orientation.Horizontal)
        self.trail_slider.setRange(20, 500)
        self.trail_slider.setValue(config.TRAIL_LENGTH)
        self.trail_label = QLabel(str(config.TRAIL_LENGTH))
        self.trail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.trail_slider.valueChanged.connect(self._on_trail_changed)
        viz_layout.addWidget(self.trail_slider)
        viz_layout.addWidget(self.trail_label)

        viz_layout.addWidget(QLabel("Point Scale"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(10, 200)
        self.size_slider.setValue(int(config.POINT_SIZE_SCALE))
        self.size_label = QLabel(str(int(config.POINT_SIZE_SCALE)))
        self.size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        viz_layout.addWidget(self.size_slider)
        viz_layout.addWidget(self.size_label)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # ---- IA Classification ----
        self._build_classification_panel(layout)

        # Analysis information
        info_group = QGroupBox("Analysis Information")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(8, 14, 8, 8)
        info_layout.setSpacing(4)
        self.info_label = QLabel("--")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet(
            "font-size: 12px; color: #aaaaaa; line-height: 1.6;")
        info_layout.addWidget(self.info_label)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()

        scroll.setWidget(inner)
        outer_layout.addWidget(scroll)

    def _build_classification_panel(self, parent_layout):
        """Builds the IA Classification section with tabular layout."""
        class_group = QGroupBox("IA Classification")
        outer = QVBoxLayout()
        outer.setContentsMargins(8, 18, 8, 8)
        outer.setSpacing(10)

        # Categoria principal
        self.class_category_label = QLabel("--")
        self.class_category_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #00e5ff;"
            "padding: 4px 0px;")
        self.class_category_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.class_category_label.setWordWrap(True)
        self.class_category_label.setMinimumHeight(24)
        self.class_category_label.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        outer.addWidget(self.class_category_label)

        # Confidence bar
        conf_row = QHBoxLayout()
        conf_row.setSpacing(8)
        conf_label = QLabel("Confidence:")
        conf_label.setStyleSheet("font-size: 11px; color: #888888;")
        conf_label.setFixedWidth(78)
        conf_row.addWidget(conf_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setFixedHeight(14)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%p%")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 3px;
                color: #cccccc;
                font-size: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #005577, stop:1 #00ccff);
                border-radius: 2px;
            }
        """)
        conf_row.addWidget(self.confidence_bar)
        outer.addLayout(conf_row)

        # Separador
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #333333;")
        sep.setFixedHeight(1)
        outer.addWidget(sep)

        # Header "Scores"
        scores_header = QLabel("Secondary Scores")
        scores_header.setStyleSheet(
            "font-size: 10px; color: #666666; font-weight: bold;"
            "padding-top: 2px;")
        outer.addWidget(scores_header)

        # Grid tabulado para puntuaciones secundarias
        self._scores_grid = QGridLayout()
        self._scores_grid.setContentsMargins(0, 0, 0, 0)
        self._scores_grid.setHorizontalSpacing(6)
        self._scores_grid.setVerticalSpacing(3)
        self._scores_grid.setColumnStretch(0, 3)
        self._scores_grid.setColumnStretch(1, 1)

        # Pre-crear 5 filas de labels para puntuaciones (top 5)
        self._score_name_labels = []
        self._score_value_labels = []
        for i in range(5):
            name_lbl = QLabel("")
            name_lbl.setStyleSheet(
                "font-size: 10px; color: #999999; padding: 1px 0px;")
            value_lbl = QLabel("")
            value_lbl.setStyleSheet(
                "font-size: 10px; color: #aaaaaa; font-family: monospace;"
                "padding: 1px 0px;")
            value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            self._scores_grid.addWidget(name_lbl, i, 0)
            self._scores_grid.addWidget(value_lbl, i, 1)
            self._score_name_labels.append(name_lbl)
            self._score_value_labels.append(value_lbl)

        outer.addLayout(self._scores_grid)

        # Separador fino
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("background-color: #333333;")
        sep2.setFixedHeight(1)
        outer.addWidget(sep2)

        # Detalles descriptivos del patron detectado
        self.class_details_label = QLabel("")
        self.class_details_label.setWordWrap(True)
        self.class_details_label.setStyleSheet(
            "font-size: 10px; color: #777777; padding-top: 2px;")
        outer.addWidget(self.class_details_label)

        class_group.setLayout(outer)
        parent_layout.addWidget(class_group)

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            get_audio_filter_string())
        if path:
            self.file_label.setText(path.split("/")[-1])
            self.file_selected.emit(path)

    def update_time_display(self, current_time):
        """Update time display with the current position."""
        curr_min = int(current_time) // 60
        curr_sec = current_time % 60
        dur_min = int(self._duration) // 60
        dur_sec = self._duration % 60
        self.time_label.setText(
            f"{curr_min:02d}:{curr_sec:05.2f} / {dur_min:02d}:{dur_sec:05.2f}")

        if self._duration > 0:
            progress = int((current_time / self._duration) * 1000)
            self.progress_bar.setValue(min(progress, 1000))

    def set_duration(self, duration):
        """Set the total duration for the display."""
        self._duration = duration

    def set_analysis_info(self, info_text):
        """Display analysis information."""
        self.info_label.setText(info_text)

    def set_processing_status(self, message, percent):
        """Show real-time processing progress in the classification panel."""
        self.class_category_label.setText(message)
        self.class_category_label.setStyleSheet(
            "font-size: 12px; font-weight: bold; color: #ffaa00;"
            "padding: 4px 0px;")
        self.confidence_bar.setValue(percent)
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 3px;
                color: #cccccc;
                font-size: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #cc7700, stop:1 #ffaa00);
                border-radius: 2px;
            }
        """)
        self.class_details_label.setText("")
        for i in range(len(self._score_name_labels)):
            self._score_name_labels[i].setText("")
            self._score_value_labels[i].setText("")

    def _restore_confidence_style(self):
        """Restore the default cyan confidence bar style."""
        self.class_category_label.setStyleSheet(
            "font-size: 13px; font-weight: bold; color: #00e5ff;"
            "padding: 4px 0px;")
        self.confidence_bar.setStyleSheet("""
            QProgressBar {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 3px;
                color: #cccccc;
                font-size: 10px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #005577, stop:1 #00ccff);
                border-radius: 2px;
            }
        """)

    def set_classification(self, result):
        """Display IA classification results."""
        self._restore_confidence_style()

        if result is None:
            self.class_category_label.setText("Awaiting playback...")
            self.confidence_bar.setValue(0)
            self.class_details_label.setText("")
            for i in range(len(self._score_name_labels)):
                self._score_name_labels[i].setText("")
                self._score_value_labels[i].setText("")
            return

        # Categoria principal
        self.class_category_label.setText(result["label"])

        # Barra de confianza
        confidence_pct = int(result["confidence"] * 100)
        self.confidence_bar.setValue(confidence_pct)

        # Puntuaciones tabuladas (top 5 categorias)
        from core.audio_classifier import AudioClassifier
        cats = AudioClassifier.CATEGORIES
        scores = result.get("scores", {})
        top_items = list(scores.items())[:5]

        for i in range(5):
            if i < len(top_items):
                cat, score = top_items[i]
                label = cats.get(cat, cat)
                self._score_name_labels[i].setText(label)
                self._score_value_labels[i].setText(f"{score:.1f}")
                # Destacar la primera fila (ganadora)
                if i == 0:
                    self._score_name_labels[i].setStyleSheet(
                        "font-size: 10px; color: #00ccff; font-weight: bold;"
                        "padding: 1px 0px;")
                    self._score_value_labels[i].setStyleSheet(
                        "font-size: 10px; color: #00ccff; font-weight: bold;"
                        "font-family: monospace; padding: 1px 0px;")
                else:
                    self._score_name_labels[i].setStyleSheet(
                        "font-size: 10px; color: #999999;"
                        "padding: 1px 0px;")
                    self._score_value_labels[i].setStyleSheet(
                        "font-size: 10px; color: #aaaaaa;"
                        "font-family: monospace; padding: 1px 0px;")
            else:
                self._score_name_labels[i].setText("")
                self._score_value_labels[i].setText("")

        # Detalles del patron
        self.class_details_label.setText(result.get("details", ""))

    def _on_trail_changed(self, value):
        self.trail_label.setText(str(value))
        self.trail_length_changed.emit(value)

    def _on_size_changed(self, value):
        self.size_label.setText(str(value))
        self.point_scale_changed.emit(value)
