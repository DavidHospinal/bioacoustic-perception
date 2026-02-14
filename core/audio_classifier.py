"""
Audio classifier using YAMNet (TensorFlow Lite).
Classifies audio into 3 categories:
- Musical Instruments
- Human Voice
- Bioacoustics

Uses a pre-trained YAMNet model (521 classes) mapping:
- Voice: Speech, Singing, Human sounds (indices 0-66)
- Instruments: Musical instruments, Music genres (indices 132-276)
- Bioacoustics: Animals (67-131) + Nature (277-293)
"""
import os
import csv
import numpy as np

class AudioClassifier:
    """
    Classifies audio using YAMNet TFLite model.
    Falls back to legacy rule-based system if model is missing.
    """

    CATEGORIES = {
        "instruments": "Musical Instruments",
        "voice": "Human Voice",
        "bioacoustics": "Bioacoustics",
    }

    # YAMNet expects 16kHz audio, 0.975s frames (15600 samples)
    TARGET_SR = 16000
    INPUT_SIZE = 15600

    def __init__(self):
        self.result = None
        self.scores = {}
        self.confidence = 0.0
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.yamnet_classes = []
        
        # Load YAMNet model
        self._load_model()

    def _load_model(self):
        """Load TFLite model and class map."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base_dir, "models", "yamnet.tflite")
        class_map_path = os.path.join(base_dir, "models", "yamnet_class_map.csv")

        if not os.path.exists(model_path) or not os.path.exists(class_map_path):
            print("YAMNet model not found. Using safe fallbacks.")
            return

        # Load class map
        try:
            with open(class_map_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                self.yamnet_classes = [row[2] for row in reader]
        except Exception as e:
            print(f"Error loading class map: {e}")
            return

        # Load TFLite interpreter
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                from tflite_runtime.interpreter import Interpreter
            
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("YAMNet model loaded successfully.")
        except Exception as e:
            print(f"Error loading YAMNet interpreter: {e}")
            self.interpreter = None

    def classify(self, features, y, sr):
        """
        Classify audio using YAMNet.
        
        Parameters
        ----------
        features : dict
            Legacy features (kept for compatibility/stats)
        y : np.ndarray
            Raw audio signal
        sr : int
            Sample rate
        """
        import librosa

        # Legacy stats for visualization/debug
        stats = self._compute_legacy_stats(features, y, sr)

        if self.interpreter is None:
            return self._fallback_classify(stats)

        # --- YAMNet Inference ---
        
        # 1. Resample to 16kHz
        if sr != self.TARGET_SR:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=self.TARGET_SR)
        else:
            y_16k = y

        # 2. Normalize (-1.0 to 1.0)
        max_val = np.max(np.abs(y_16k))
        if max_val > 0:
            y_16k = y_16k / max_val
        y_16k = y_16k.astype(np.float32)

        # 3. Frame audio into 15600 sample chunks (0.975s)
        # Use simple tiling without overlap for speed, or 50% overlap for accuracy.
        # Let's use no overlap for simplicity, discarding the last partial frame.
        num_samples = len(y_16k)
        num_frames = num_samples // self.INPUT_SIZE
        
        if num_frames == 0:
            # Pad if too short
            y_16k = np.pad(y_16k, (0, self.INPUT_SIZE - num_samples))
            num_frames = 1

        input_index = self.input_details[0]['index']
        output_index = self.output_details[0]['index']
        
        # Accumulate scores for our 3 categories
        category_scores = {
            "voice": 0.0,
            "instruments": 0.0,
            "bioacoustics": 0.0
        }

        # Process each frame
        valid_frames = 0
        for i in range(num_frames):
            start = i * self.INPUT_SIZE
            end = start + self.INPUT_SIZE
            frame = y_16k[start:end]
            
            # Reshape to [15600] (YAMNet generic model expects 1D input)
            input_tensor = frame
            
            self.interpreter.set_tensor(input_index, input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_index)[0] # [521]
            
            # Map YAMNet classes to our categories
            # Voice: 0-66
            # Instruments: 132-276
            # Bioacoustics: 67-131 (Animals) + 277-293 (Nature)
            
            # Sum probabilites for each group
            voice_prob = np.sum(output_data[0:67])
            bio_prob = np.sum(output_data[67:132]) + np.sum(output_data[277:294])
            instr_prob = np.sum(output_data[132:277])
            
            category_scores["voice"] += voice_prob
            category_scores["instruments"] += instr_prob
            category_scores["bioacoustics"] += bio_prob
            valid_frames += 1

        # Average scores
        if valid_frames > 0:
            for k in category_scores:
                category_scores[k] /= valid_frames

        # Determine winner
        if not category_scores:
            category = "instruments" 
            self.confidence = 0.0
        else:
            category = max(category_scores, key=category_scores.get)
            max_score = category_scores[category]
            total = sum(category_scores.values())
            # Normalize confidence based on total meaningful energy
            self.confidence = max_score / total if total > 0 else 0.0
            
            # Scale scores for display (0-10 range approximation)
            self.scores = {k: v * 10.0 for k, v in category_scores.items()}

        label = self.CATEGORIES.get(category, "Unclassified")
        
        # Build details string
        details = self._build_details(category, stats, category_scores)

        self.result = {
            "category": category,
            "label": label,
            "confidence": self.confidence,
            "scores": dict(sorted(self.scores.items(), key=lambda x: x[1], reverse=True)),
            "stats": stats,
            "details": details,
        }
        return self.result

    def _compute_legacy_stats(self, features, y, sr):
        """Compute legacy descriptors for visualization/stats."""
        import librosa
        
        centroid = features["centroid"]
        bandwidth = features["bandwidth"]
        rms = features["rms"]
        chroma = features["chroma"]
        mfcc = features["mfcc"]

        zcr = librosa.feature.zero_crossing_rate(y)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        return {
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "centroid_max": float(np.max(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "bandwidth_std": float(np.std(bandwidth)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "harmonic_ratio": float(self._harmonic_ratio(y, sr)),
            "vocal_band_ratio": float(self._vocal_band_energy(y, sr)),
            "high_freq_ratio": float(self._high_freq_energy(y, sr)),
            "zcr_mean": float(np.mean(zcr)),
            "mfcc_delta_std": float(self._mfcc_delta_stability(mfcc)),
            "chroma_entropy": float(self._chroma_entropy(chroma)),
            "chroma_dominance": float(self._chroma_dominance(chroma)),
            "chroma_std": float(self._chroma_temporal_variability(chroma)),
            "flatness_mean": float(np.mean(flatness)),
            "flatness_std": float(np.std(flatness)),
            "rolloff_mean": float(np.mean(rolloff)),
        }

    # ------------------------------------------------------------------ #
    #  Legacy Helper Methods (Kept for stats)                           #
    # ------------------------------------------------------------------ #
    
    def _chroma_entropy(self, chroma):
        chroma_mean = np.mean(chroma, axis=1)
        chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-10)
        return -np.sum(chroma_norm * np.log2(chroma_norm + 1e-10))

    def _chroma_dominance(self, chroma):
        chroma_mean = np.mean(chroma, axis=1)
        total = chroma_mean.sum()
        return float(chroma_mean.max() / total) if total > 1e-10 else 0.0

    def _chroma_temporal_variability(self, chroma):
        if chroma.shape[1] < 2: return 0.0
        return float(np.mean(np.std(chroma, axis=1)))

    def _harmonic_ratio(self, y, sr):
        import librosa
        y_harm, y_perc = librosa.effects.hpss(y)
        harm = np.sum(y_harm ** 2)
        total = harm + np.sum(y_perc ** 2)
        return float(harm / total) if total > 0 else 0.5

    def _vocal_band_energy(self, y, sr):
        import librosa
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = (freqs >= 80) & (freqs <= 1100)
        total = np.sum(S ** 2)
        return float(np.sum(S[mask, :] ** 2) / total) if total > 0 else 0.0

    def _high_freq_energy(self, y, sr):
        import librosa
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = freqs >= 3000
        total = np.sum(S ** 2)
        return float(np.sum(S[mask, :] ** 2) / total) if total > 0 else 0.0

    def _mfcc_delta_stability(self, mfcc):
        if mfcc.shape[1] < 3: return 0.0
        delta = np.diff(mfcc, axis=1)
        return float(np.mean(np.std(delta, axis=1)))

    def _build_details(self, category, stats, yamnet_scores):
        """Build descriptive text."""
        lines = []
        lines.append(f"YAMNet Prediction: {self.CATEGORIES.get(category, category)}")
        lines.append(f"Confidence: {self.confidence:.1%}")
        lines.append("")
        lines.append("Class Probabilities:")
        lines.append(f"  Voice: {yamnet_scores.get('voice', 0):.3f}")
        lines.append(f"  Instruments: {yamnet_scores.get('instruments', 0):.3f}")
        lines.append(f"  Bioacoustics: {yamnet_scores.get('bioacoustics', 0):.3f}")
        lines.append("")
        lines.append("Acoustic Features (Legacy):")
        lines.append(f"  Centroid: {stats['centroid_mean']/1000:.2f} KHz")
        lines.append(f"  Harmonic%: {stats['harmonic_ratio']:.0%}")
        lines.append(f"  ZCR: {stats['zcr_mean']:.3f}")
        lines.append(f"  MFCC Delta: {stats['mfcc_delta_std']:.1f}")
        return "\n".join(lines)

    def _fallback_classify(self, stats):
        """Minimal fallback using simple rules if model fails."""
        # Simple heuristic fallback
        scores = {"instruments": 0, "voice": 0, "bioacoustics": 0}
        
        # Bioacoustics: low harmonic, high centroid
        if stats["harmonic_ratio"] < 0.45: scores["bioacoustics"] += 5
        
        # Voice: high MFCC delta
        if stats["mfcc_delta_std"] > 20: scores["voice"] += 5
        
        # Instruments: high harmonic, low ZCR
        if stats["harmonic_ratio"] > 0.7 and stats["zcr_mean"] < 0.1: scores["instruments"] += 5

        category = max(scores, key=scores.get)
        self.result = {
            "category": category,
            "label": self.CATEGORIES.get(category),
            "confidence": 0.5,
            "scores": scores,
            "stats": stats,
            "details": "Model missing. Using fallback rules."
        }
        return self.result
