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

Performance optimizations:
- Accepts pre-computed STFT from AudioAnalyzer (avoids redundant computation)
- Uses librosa.decompose.hpss(S) instead of librosa.effects.hpss(y)
  to skip expensive ISTFT reconstruction
- Passes S= to spectral_flatness and spectral_rolloff
"""
import os
import csv
import numpy as np
import config


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

    def classify(self, features, y, sr, S=None, progress_callback=None):
        """
        Classify audio using YAMNet.

        Parameters
        ----------
        features : dict
            Extracted audio features (centroid, chroma, rms, etc.)
        y : np.ndarray
            Raw audio signal
        sr : int
            Sample rate
        S : np.ndarray, optional
            Pre-computed STFT magnitude from AudioAnalyzer.
            Avoids redundant STFT computation in legacy stats.
        progress_callback : callable, optional
            Function(message: str, percent: int) for progress reporting.
        """
        import librosa

        if progress_callback:
            progress_callback("Computing spectral descriptors...", 10)

        stats = self._compute_legacy_stats(features, y, sr, S=S)

        if self.interpreter is None:
            return self._fallback_classify(stats)

        # --- YAMNet Inference ---

        if progress_callback:
            progress_callback("Resampling audio for YAMNet...", 25)

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
        num_samples = len(y_16k)
        num_frames = num_samples // self.INPUT_SIZE

        if num_frames == 0:
            # Pad if too short
            y_16k = np.pad(y_16k, (0, self.INPUT_SIZE - num_samples))
            num_frames = 1

        input_index = self.input_details[0]['index']
        output_index = self.output_details[0]['index']

        # Initialize accumulator for 521 classes
        sum_outputs = np.zeros(521, dtype=np.float32)
        valid_frames = 0

        # Apply frame stride for faster inference (process every Nth frame)
        stride = max(1, config.YAMNET_FRAME_STRIDE)
        frames_to_process = list(range(0, num_frames, stride))
        total_to_process = len(frames_to_process)

        if progress_callback:
            progress_callback(
                f"Running YAMNet inference (0/{total_to_process})...", 35)

        # Process sampled frames
        for idx, i in enumerate(frames_to_process):
            start = i * self.INPUT_SIZE
            end = start + self.INPUT_SIZE
            frame = y_16k[start:end]

            self.interpreter.set_tensor(input_index, frame)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_index)[0]  # [521]

            sum_outputs += output_data
            valid_frames += 1

            # Report progress every 5 processed frames
            if progress_callback and (idx + 1) % 5 == 0:
                pct = 35 + int(((idx + 1) / total_to_process) * 45)
                progress_callback(
                    f"YAMNet inference ({idx + 1}/{total_to_process})...",
                    min(pct, 80))

        if valid_frames > 0:
            mean_output = sum_outputs / valid_frames
        else:
            mean_output = np.zeros(521)

        if progress_callback:
            progress_callback("Aggregating predictions...", 85)

        # --- Classification Logic ---

        # 1. Get Top-1 YAMNet prediction
        top_index = np.argmax(mean_output)
        top_score = float(mean_output[top_index])
        top_label = self.yamnet_classes[top_index] if self.yamnet_classes else "Unknown"

        # 2. Calculate Aggregated Category Scores
        voice_score = float(np.sum(mean_output[0:67]))
        instr_score = float(np.sum(mean_output[132:277]))
        bio_score = float(np.sum(mean_output[67:132]) + np.sum(mean_output[277:294]))

        category_scores = {
            "voice": voice_score,
            "instruments": instr_score,
            "bioacoustics": bio_score
        }

        # 3. Determine Final Category
        is_voice = 0 <= top_index <= 66
        is_bio = (67 <= top_index <= 131) or (277 <= top_index <= 293)
        is_instr = 132 <= top_index <= 276

        is_mapped_category = is_voice or is_bio or is_instr

        if is_mapped_category:
            winner = max(category_scores, key=category_scores.get)
            final_category = winner
            final_label = self.CATEGORIES[winner]
        else:
            # Fallback for unmapped sounds
            final_category = top_label.lower().replace(" ", "_")
            final_label = top_label
            category_scores[final_category] = top_score

        # 4. Normalize Scores for Display (0.0 - 1.0)
        total_score = sum(category_scores.values())
        if total_score > 0:
            for k in category_scores:
                category_scores[k] /= total_score

        if progress_callback:
            progress_callback("Finalizing result...", 95)

        # Prepare Result
        self.result = {
            "category": final_category,
            "label": final_label,
            "confidence": float(category_scores[final_category]),
            "scores": category_scores,
            "stats": stats,
            "details": f"Top: {top_label} ({top_score:.2f})"
        }

        return self.result

    def _compute_legacy_stats(self, features, y, sr, S=None):
        """
        Compute legacy descriptors for visualization and UI display.
        Reuses the pre-computed STFT magnitude from AudioAnalyzer
        to avoid redundant FFT computation.
        """
        import librosa

        centroid = features["centroid"]
        bandwidth = features["bandwidth"]
        rms = features["rms"]
        chroma = features["chroma"]
        mfcc = features["mfcc"]

        # Reuse pre-computed STFT if available
        if S is None:
            S = np.abs(librosa.stft(y, n_fft=2048))

        zcr = librosa.feature.zero_crossing_rate(y)[0]

        # Pass S= directly to avoid internal STFT recomputation
        flatness = librosa.feature.spectral_flatness(S=S)[0]
        rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)[0]

        # HPSS on magnitude spectrogram (limited to first N seconds for speed)
        hpss_max_frames = min(
            S.shape[1],
            int(config.HPSS_MAX_DURATION * sr / config.HOP_LENGTH))
        S_sub = S[:, :hpss_max_frames]
        S_harm, S_perc = librosa.decompose.hpss(S_sub)
        harm_energy = np.sum(S_harm ** 2)
        total_hpss = harm_energy + np.sum(S_perc ** 2)
        harmonic_ratio = float(harm_energy / total_hpss) if total_hpss > 0 else 0.5

        # Reuse S for frequency band energies
        n_fft = (S.shape[0] - 1) * 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        total_energy = np.sum(S ** 2)

        # Vocal band (80-1100 Hz)
        vocal_mask = (freqs >= 80) & (freqs <= 1100)
        vocal_band_ratio = float(
            np.sum(S[vocal_mask, :] ** 2) / total_energy
        ) if total_energy > 0 else 0.0

        # High frequency (> 3kHz)
        hf_mask = freqs >= 3000
        high_freq_ratio = float(
            np.sum(S[hf_mask, :] ** 2) / total_energy
        ) if total_energy > 0 else 0.0

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
            "harmonic_ratio": harmonic_ratio,
            "vocal_band_ratio": vocal_band_ratio,
            "high_freq_ratio": high_freq_ratio,
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
        scores = {"instruments": 0, "voice": 0, "bioacoustics": 0}

        if stats["harmonic_ratio"] < 0.45: scores["bioacoustics"] += 5
        if stats["mfcc_delta_std"] > 20: scores["voice"] += 5
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
