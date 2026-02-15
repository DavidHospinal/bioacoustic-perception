"""
Background worker for the audio analysis pipeline.
Runs heavy computation (audio decoding, feature extraction, AI classification)
on a QThread to keep the UI responsive.
"""
from PyQt6.QtCore import QThread, pyqtSignal
from core.feature_cache import FeatureCache


class AnalysisWorker(QThread):
    """
    Executes audio loading, feature extraction, or classification
    on a background thread with real-time progress reporting.

    Modes:
        MODE_LOAD     - Decode audio, extract features, map visualization.
        MODE_CLASSIFY - Run YAMNet inference and spectral classification.
    """

    progress = pyqtSignal(str, int)          # (stage_message, percent 0-100)
    load_complete = pyqtSignal(dict)         # analysis results
    classify_complete = pyqtSignal(dict)     # classification result
    error = pyqtSignal(str)

    MODE_LOAD = "load"
    MODE_CLASSIFY = "classify"

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.file_path = None
        self.analyzer = None
        self.mapper = None
        self.classifier = None
        self.features = None

    def run(self):
        try:
            if self.mode == self.MODE_LOAD:
                self._run_load()
            elif self.mode == self.MODE_CLASSIFY:
                self._run_classify()
        except Exception as e:
            self.error.emit(str(e))

    def _run_load(self):
        """Load audio file, extract features, compute 3D mapping."""
        # Check cache
        self.progress.emit("Checking cache...", 5)
        cached = FeatureCache.load(self.file_path)

        if cached is not None:
            self.progress.emit("Loading from cache...", 20)
            self.analyzer.y = cached["y"]
            self.analyzer.duration = cached["duration"]
            self.analyzer.sr = cached["sr"]
            self.analyzer._stft = None  # Recomputed on demand
            self.features = cached["features"]
            self.analyzer.features = self.features

            self.progress.emit("Mapping visualization...", 80)
            self.mapper.map_features(self.features)

            self.progress.emit("Ready", 100)
            self.load_complete.emit({
                "features": self.features,
                "from_cache": True,
                "classification": cached.get("classification"),
            })
            return

        # Fresh analysis
        self.progress.emit("Decoding audio...", 10)
        self.analyzer.load(self.file_path)

        self.progress.emit("Extracting spectral features...", 35)
        self.features = self.analyzer.analyze()

        self.progress.emit("Computing 3D mapping...", 65)
        self.mapper.map_features(self.features)

        self.progress.emit("Saving to cache...", 90)
        FeatureCache.save(self.file_path, {
            "y": self.analyzer.y,
            "duration": self.analyzer.duration,
            "sr": self.analyzer.sr,
            "features": self.features,
        })

        self.progress.emit("Ready", 100)
        self.load_complete.emit({
            "features": self.features,
            "from_cache": False,
            "classification": None,
        })

    def _run_classify(self):
        """Run AI classification on the full audio signal."""
        def on_progress(msg, pct):
            self.progress.emit(msg, pct)

        result = self.classifier.classify(
            self.features,
            self.analyzer.get_raw_audio(),
            self.analyzer.get_sample_rate(),
            S=self.analyzer.get_stft(),
            progress_callback=on_progress)

        # Persist classification to cache
        if self.file_path:
            FeatureCache.update_classification(self.file_path, result)

        self.progress.emit("Classification complete", 100)
        self.classify_complete.emit(result)
