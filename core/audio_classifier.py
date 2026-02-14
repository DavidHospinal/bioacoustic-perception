"""
Audio classifier based on global spectral descriptors.
Processes the average of all descriptors from the complete audio file
before emitting a prediction, allowing the AI to 'listen' to all
audio variations for higher accuracy.

Classification into 3 categories:
- Musical Instruments (strings, piano, wind, orchestra, guitar)
- Human Voice (speech, singing)
- Bioacoustics (bird calls, animal sounds, nature sounds)

Key discriminators empirically calibrated (7 test files):
- harmonic_ratio: instruments > 0.70, voice ~ 0.70, birds < 0.25
- centroid_mean: instruments 1000-1700, voice 2500+, birds 3700+
- zcr_mean: instruments < 0.12, voice > 0.12, birds > 0.20
- mfcc_delta_std: instruments < 15, voice > 18
- vocal_band_ratio: instruments > 0.60, voice ~ 0.50, birds < 0.10
"""
import numpy as np


class AudioClassifier:
    """
    Classifies audio into 3 categories using global descriptors
    averaged over the entire file.

    Primary descriptors:
    - Spectral centroid: dominant frequency
    - Spectral bandwidth: spectral spread
    - Spectral flatness: tonal vs noisy
    - Harmonic ratio: harmonic vs percussive energy
    - Vocal band ratio (80-1100 Hz): vocal band concentration
    - High freq ratio (> 3 KHz): spectral brightness
    - Rolloff: frequency containing 85% of energy

    Temporal variability descriptors:
    - Chroma std: frame-to-frame chromatic variability
    - MFCC delta std: formant stability
    - ZCR: zero crossing rate (consonants/fricatives)
    """

    CATEGORIES = {
        "instruments": "Musical Instruments",
        "voice": "Human Voice",
        "bioacoustics": "Bioacoustics",
    }

    def __init__(self):
        self.result = None
        self.scores = {}
        self.confidence = 0.0

    def classify(self, features, y, sr):
        """
        Analyze features extracted over the COMPLETE file
        and classify the audio type using global averages.

        Parameters
        ----------
        features : dict
            Output from AudioAnalyzer.analyze()
        y : np.ndarray
            Raw audio signal (complete)
        sr : int
            Sample rate

        Returns
        -------
        dict with 'category', 'label', 'confidence', 'scores', 'stats', 'details'
        """
        import librosa

        centroid = features["centroid"]
        bandwidth = features["bandwidth"]
        rms = features["rms"]
        chroma = features["chroma"]
        mfcc = features["mfcc"]

        # Additional features computed over the full signal
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]

        # --- Global descriptors (averaged over the ENTIRE file) ---
        stats = {
            "centroid_mean": float(np.mean(centroid)),
            "centroid_std": float(np.std(centroid)),
            "centroid_max": float(np.max(centroid)),
            "centroid_median": float(np.median(centroid)),
            "bandwidth_mean": float(np.mean(bandwidth)),
            "bandwidth_std": float(np.std(bandwidth)),
            "rms_mean": float(np.mean(rms)),
            "rms_std": float(np.std(rms)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
            "flatness_mean": float(np.mean(flatness)),
            "flatness_std": float(np.std(flatness)),
            "rolloff_mean": float(np.mean(rolloff)),
            "mfcc_mean": np.mean(mfcc, axis=1).tolist(),
            "chroma_entropy": float(self._chroma_entropy(chroma)),
            "chroma_dominance": float(self._chroma_dominance(chroma)),
            "chroma_std": float(self._chroma_temporal_variability(chroma)),
            "harmonic_ratio": float(self._harmonic_ratio(y, sr)),
            "vocal_band_ratio": float(self._vocal_band_energy(y, sr)),
            "high_freq_ratio": float(self._high_freq_energy(y, sr)),
            "mfcc_delta_std": float(self._mfcc_delta_stability(mfcc)),
        }

        # Scoring system per category
        self.scores = {}
        self._score_instruments(stats)
        self._score_voice(stats)
        self._score_bioacoustics(stats)

        if not self.scores:
            category = "instruments"
            self.confidence = 0.0
        else:
            category = max(self.scores, key=self.scores.get)
            max_score = self.scores[category]
            total = sum(self.scores.values())
            self.confidence = max_score / total if total > 0 else 0.0

        label = self.CATEGORIES.get(category, "Unclassified")

        self.result = {
            "category": category,
            "label": label,
            "confidence": self.confidence,
            "scores": dict(sorted(
                self.scores.items(), key=lambda x: x[1], reverse=True)),
            "stats": stats,
            "details": self._build_details(category, stats),
        }
        return self.result

    # ------------------------------------------------------------------ #
    #  Derived global descriptors                                          #
    # ------------------------------------------------------------------ #

    def _chroma_entropy(self, chroma):
        """Average chromagram entropy. Low = tonal, high = noisy."""
        chroma_mean = np.mean(chroma, axis=1)
        chroma_norm = chroma_mean / (chroma_mean.sum() + 1e-10)
        entropy = -np.sum(chroma_norm * np.log2(chroma_norm + 1e-10))
        return entropy

    def _chroma_dominance(self, chroma):
        """Strength of the dominant average note (0-1)."""
        chroma_mean = np.mean(chroma, axis=1)
        total = chroma_mean.sum()
        if total < 1e-10:
            return 0.0
        return float(chroma_mean.max() / total)

    def _chroma_temporal_variability(self, chroma):
        """
        Frame-to-frame chromatic variability.
        High = polyphony (orchestra), low = monophony (solo voice).
        """
        if chroma.shape[1] < 2:
            return 0.0
        per_bin_std = np.std(chroma, axis=1)
        return float(np.mean(per_bin_std))

    def _harmonic_ratio(self, y, sr):
        """Harmonic vs percussive energy ratio over the full signal."""
        import librosa
        y_harm, y_perc = librosa.effects.hpss(y)
        harm_energy = np.sum(y_harm ** 2)
        perc_energy = np.sum(y_perc ** 2)
        total = harm_energy + perc_energy
        if total < 1e-10:
            return 0.5
        return float(harm_energy / total)

    def _vocal_band_energy(self, y, sr):
        """
        Energy ratio in the fundamental vocal band (80-1100 Hz)
        vs total energy.
        """
        import librosa
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        vocal_mask = (freqs >= 80) & (freqs <= 1100)
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-10:
            return 0.0
        vocal_energy = np.sum(S[vocal_mask, :] ** 2)
        return float(vocal_energy / total_energy)

    def _high_freq_energy(self, y, sr):
        """
        Energy ratio above 3 KHz vs total energy.
        """
        import librosa
        S = np.abs(librosa.stft(y, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

        high_mask = freqs >= 3000
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-10:
            return 0.0
        high_energy = np.sum(S[high_mask, :] ** 2)
        return float(high_energy / total_energy)

    def _mfcc_delta_stability(self, mfcc):
        """
        Temporal stability of MFCCs (deltas).
        Human voice has stable formants -> low deltas.
        """
        if mfcc.shape[1] < 3:
            return 0.0
        delta = np.diff(mfcc, axis=1)
        return float(np.mean(np.std(delta, axis=1)))

    # ------------------------------------------------------------------ #
    #  Scoring system - 3 categories                                       #
    #                                                                      #
    #  Empirically calibrated with 7 test files:                           #
    #  - Paganini violin:   harm=0.91 flat=0.0005 vb=0.96 cent=1044       #
    #  - Vivaldi strings:   harm=0.94 flat=0.008  vb=0.74 cent=1664       #
    #  - Bach organ:        harm=0.96 flat=0.021  vb=0.68 cent=1232       #
    #  - Stravinsky orch:   harm=0.72 flat=0.006  vb=0.78 cent=1127       #
    #  - Kasandr guitar:    harm=0.79 flat=0.290  vb=0.60 cent=1072       #
    #  - Freddie vocals:    harm=0.70 flat=0.032  vb=0.51 cent=2587       #
    #  - Morning birds:     harm=0.20 flat=0.029  vb=0.04 cent=3717       #
    # ------------------------------------------------------------------ #

    def _score_instruments(self, s):
        """
        Musical Instruments: strings, piano, wind, orchestra, guitar.

        Empirical key signals for instruments:
        - Low centroid (1000-1700 Hz) - fundamentals in mid-low range
        - Low ZCR (< 0.12) - no consonants or fricatives
        - High harmonic ratio (> 0.70) - pure tonal sound
        - Stable MFCC delta (< 15) - consistent timbre
        - High vocal band ratio (> 0.60) - energy in fundamentals
        """
        score = 0.0

        # --- LOW CENTROID: classical instruments 1000-1700 Hz ---
        # This is the STRONGEST discriminator vs voice (voice ~ 2587)
        if s["centroid_mean"] < 1200:
            score += 4.0
        elif s["centroid_mean"] < 1800:
            score += 3.0
        elif s["centroid_mean"] < 2200:
            score += 1.5

        # --- LOW ZCR: no consonants (instruments 0.04-0.11) ---
        # Voice has 0.165, instruments < 0.12
        if s["zcr_mean"] < 0.06:
            score += 3.5
        elif s["zcr_mean"] < 0.10:
            score += 2.5
        elif s["zcr_mean"] < 0.12:
            score += 1.0

        # --- HIGH HARMONIC RATIO: tonal/tuned sound ---
        # Instruments 0.72-0.96, Voice 0.70, Birds 0.20
        if s["harmonic_ratio"] > 0.90:
            score += 4.0
        elif s["harmonic_ratio"] > 0.80:
            score += 3.0
        elif s["harmonic_ratio"] > 0.70:
            score += 1.5

        # --- HIGH VOCAL BAND: concentrated fundamentals ---
        # Instruments 0.60-0.96, Voice 0.51
        if s["vocal_band_ratio"] > 0.80:
            score += 3.0
        elif s["vocal_band_ratio"] > 0.65:
            score += 2.0
        elif s["vocal_band_ratio"] > 0.55:
            score += 1.0

        # --- STABLE MFCC delta: consistent timbre ---
        # Instruments < 15, Voice > 18
        if s["mfcc_delta_std"] < 10:
            score += 2.5
        elif s["mfcc_delta_std"] < 14:
            score += 2.0
        elif s["mfcc_delta_std"] < 17:
            score += 0.5

        # Low rolloff - classical instruments < 3200 Hz
        if s["rolloff_mean"] < 2500:
            score += 1.5
        elif s["rolloff_mean"] < 3500:
            score += 1.0

        # PENALTY: very high ZCR = consonants = voice
        if s["zcr_mean"] > 0.15:
            score -= 4.0
        elif s["zcr_mean"] > 0.12:
            score -= 1.5

        # PENALTY: very high centroid = birds or energetic voice
        if s["centroid_mean"] > 3500:
            score -= 5.0
        elif s["centroid_mean"] > 2500:
            score -= 3.0

        # PENALTY: low harmonic ratio = bioacoustics/noise
        if s["harmonic_ratio"] < 0.30:
            score -= 5.0
        elif s["harmonic_ratio"] < 0.50:
            score -= 2.0

        self.scores["instruments"] = max(score, 0.0)

    def _score_voice(self, s):
        """
        Human Voice: speech or singing.

        Empirical key signals for voice:
        - Mid-high centroid (2000-3500 Hz) - distributed energy
        - High ZCR (> 0.12) - consonants and fricatives
        - High MFCC delta (> 18) - formant transitions
        - Moderate harmonic ratio (0.55-0.80) - harmonic with breath
        - Mid vocal band ratio (0.40-0.60) - not as concentrated
        """
        score = 0.0

        # --- MID-HIGH CENTROID: voice has more mid-high energy ---
        # Voice ~ 2587 Hz, instruments 1000-1700 Hz
        if 2000 < s["centroid_mean"] < 3200:
            score += 4.0
        elif 1800 < s["centroid_mean"] < 3500:
            score += 2.5
        elif 1500 < s["centroid_mean"] < 4000:
            score += 1.0

        # --- HIGH ZCR: consonants and fricatives ---
        # Voice = 0.165, instruments = 0.04-0.11
        if s["zcr_mean"] > 0.15:
            score += 4.0
        elif s["zcr_mean"] > 0.12:
            score += 2.5
        elif s["zcr_mean"] > 0.08:
            score += 0.5

        # --- HIGH MFCC delta = formant transitions (vowels) ---
        # Voice = 21.8, instruments = 8-14
        if s["mfcc_delta_std"] > 20:
            score += 3.5
        elif s["mfcc_delta_std"] > 17:
            score += 2.0
        elif s["mfcc_delta_std"] > 14:
            score += 0.5

        # --- MODERATE harmonic ratio: harmonic but with breath noise ---
        # Voice = 0.70, instruments = 0.72-0.96 (purer)
        if 0.55 < s["harmonic_ratio"] < 0.80:
            score += 2.5
        elif 0.50 < s["harmonic_ratio"] < 0.85:
            score += 1.0

        # Mid vocal band ratio
        if 0.40 < s["vocal_band_ratio"] < 0.65:
            score += 2.0
        elif 0.35 < s["vocal_band_ratio"] < 0.70:
            score += 1.0

        # Moderate flatness (voice ~ 0.032)
        if 0.01 < s["flatness_mean"] < 0.06:
            score += 1.5
        elif 0.005 < s["flatness_mean"] < 0.10:
            score += 0.5

        # Mid-high rolloff (voice ~ 4431 Hz)
        if 3500 < s["rolloff_mean"] < 5500:
            score += 1.5
        elif 3000 < s["rolloff_mean"] < 6000:
            score += 0.5

        # PENALTY: very high harmonic ratio = pure instrument
        if s["harmonic_ratio"] > 0.90:
            score -= 4.0
        elif s["harmonic_ratio"] > 0.85:
            score -= 2.0

        # PENALTY: very low centroid = probably instrument
        if s["centroid_mean"] < 1200:
            score -= 4.0
        elif s["centroid_mean"] < 1500:
            score -= 2.0

        # PENALTY: very low ZCR = no consonants
        if s["zcr_mean"] < 0.05:
            score -= 3.0
        elif s["zcr_mean"] < 0.08:
            score -= 1.5

        # PENALTY: very high centroid = birds, not voice
        if s["centroid_mean"] > 3500:
            score -= 3.0

        # PENALTY: very low harmonic ratio = bioacoustics
        if s["harmonic_ratio"] < 0.30:
            score -= 4.0

        # PENALTY: very low vocal band ratio = not human
        if s["vocal_band_ratio"] < 0.20:
            score -= 4.0

        self.scores["voice"] = max(score, 0.0)

    def _score_bioacoustics(self, s):
        """
        Bioacoustics: bird calls, animal sounds, nature sounds.

        Empirical key signals:
        - Very high centroid (> 3500 Hz) - birds sing at high frequencies
        - Very low harmonic ratio (< 0.25) - rapid modulations
        - Very low vocal band ratio (< 0.10) - not human
        - Very high ZCR (> 0.20) - rapid modulations/chirps
        - Moderate flatness - noisy environmental components
        """
        score = 0.0

        # --- VERY HIGH CENTROID: birds sing at high frequencies ---
        # Morning birds = 3717 Hz, instruments = 1000-1700
        if s["centroid_mean"] > 4500:
            score += 5.0
        elif s["centroid_mean"] > 3500:
            score += 4.0
        elif s["centroid_mean"] > 2500:
            score += 2.0

        # --- VERY LOW HARMONIC RATIO: modulations/chirps ---
        # Birds = 0.20, instruments = 0.72-0.96
        if s["harmonic_ratio"] < 0.25:
            score += 5.0
        elif s["harmonic_ratio"] < 0.35:
            score += 3.0
        elif s["harmonic_ratio"] < 0.45:
            score += 1.0

        # --- VERY LOW VOCAL BAND: not human voice ---
        # Birds = 0.04, instruments = 0.60-0.96
        if s["vocal_band_ratio"] < 0.10:
            score += 4.0
        elif s["vocal_band_ratio"] < 0.25:
            score += 2.5
        elif s["vocal_band_ratio"] < 0.35:
            score += 1.0

        # --- VERY HIGH ZCR: rapid modulations / trills ---
        # Birds = 0.32, voice = 0.17, instruments = 0.04-0.11
        if s["zcr_mean"] > 0.25:
            score += 4.0
        elif s["zcr_mean"] > 0.15:
            score += 2.0
        elif s["zcr_mean"] > 0.10:
            score += 1.0

        # High centroid variability (trills and song patterns)
        if s["centroid_std"] > 800:
            score += 2.0
        elif s["centroid_std"] > 500:
            score += 1.0

        # High energy in high frequencies (above 3 KHz)
        if s["high_freq_ratio"] > 0.25:
            score += 2.0
        elif s["high_freq_ratio"] > 0.15:
            score += 1.0

        # Moderate flatness (environmental/noisy sounds)
        if s["flatness_mean"] > 0.03:
            score += 1.0

        # PENALTY: very tonal with low centroid = instrument, not bird
        if s["harmonic_ratio"] > 0.80 and s["centroid_mean"] < 2000:
            score -= 5.0

        # PENALTY: high vocal band energy = voice/instrument
        if s["vocal_band_ratio"] > 0.55:
            score -= 5.0

        # PENALTY: high harmonic ratio = instrument
        if s["harmonic_ratio"] > 0.85:
            score -= 3.0

        self.scores["bioacoustics"] = max(score, 0.0)

    # ------------------------------------------------------------------ #
    #  Report                                                              #
    # ------------------------------------------------------------------ #

    def _build_details(self, category, stats):
        """Build descriptive text for the classification."""
        centroid_khz = stats["centroid_mean"] / 1000.0
        rolloff_khz = stats["rolloff_mean"] / 1000.0
        details = []

        details.append(f"Centroid: {centroid_khz:.2f} KHz")
        details.append(
            f"Harmonic ratio: {stats['harmonic_ratio']:.0%}")
        details.append(
            f"Flatness: {stats['flatness_mean']:.4f}")
        details.append(
            f"Vocal band (80-1100Hz): {stats['vocal_band_ratio']:.0%}")
        details.append(
            f"ZCR: {stats['zcr_mean']:.4f}")
        details.append(
            f"MFCC delta: {stats['mfcc_delta_std']:.1f}")
        details.append(
            f"Rolloff: {rolloff_khz:.1f} KHz")

        descriptors = {
            "instruments":
                "Pattern: Low centroid, high harmonicity, "
                "stable timbre without consonants",
            "voice":
                "Pattern: Mid-high centroid, consonants (ZCR), "
                "active formant transitions",
            "bioacoustics":
                "Pattern: High frequencies, rapid modulations, "
                "low vocal band energy",
        }
        if category in descriptors:
            details.append(descriptors[category])

        return "\n".join(details)
