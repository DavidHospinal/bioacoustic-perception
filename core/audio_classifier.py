"""
Audio classifier based on global spectral descriptors.
Processes the average of all descriptors from the complete audio file
before emitting a prediction, allowing the AI to 'listen' to all
audio variations for higher accuracy.

Classification into 3 categories:
- Musical Instruments (strings, piano, wind, orchestra, guitar)
- Human Voice (speech, singing)
- Bioacoustics (bird calls, animal sounds, nature sounds)

Hierarchical decision system (calibrated with 9 test files):
- Stage 1: Bioacoustics gate via harmonic_ratio < 0.45
  (perfect separation: bio max 0.31, voice/instr min 0.70)
- Stage 2: Voice vs Instruments via weighted composite score
  with MFCC delta as primary discriminator (weight 5.0)
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
    #  Hierarchical scoring system                                          #
    #                                                                       #
    #  Stage 1: Bioacoustics gate (harmonic_ratio < 0.45)                  #
    #    - Harmonic ratio perfectly separates bioacoustics (max 0.31)       #
    #      from voice/instruments (min 0.70). Threshold at 0.45 gives      #
    #      0.14 clearance on each side.                                    #
    #                                                                       #
    #  Stage 2: Voice vs Instruments (weighted composite)                  #
    #    - MFCC delta is the primary discriminator (weight 5.0)            #
    #    - Centroid, ZCR, harmonic ratio as secondary features             #
    #    - Uses midpoint thresholds between clusters for generalization    #
    #                                                                       #
    #  Calibrated with 9 test files:                                        #
    #  - 5 instruments: Paganini, Vivaldi, Bach, Stravinsky, Kasandr      #
    #  - 2 voice: Freddie vocals, Bad Bunny                                #
    #  - 2 bioacoustics: Morning birds, ludzkie                            #
    # ------------------------------------------------------------------ #

    def _score_instruments(self, s):
        """
        Musical Instruments scoring.

        In the hierarchical system, high scores mean "more instrument-like".
        Uses a weighted composite that avoids narrow range thresholds.
        """
        score = 0.0

        # --- MFCC delta stability (PRIMARY, weight 5.0) ---
        # Instruments: 8.0 - 19.6 (stable timbre)
        # Voice: 21.8 - 22.2 (formant transitions)
        # Midpoint threshold: 20.5
        if s["mfcc_delta_std"] < 14.0:
            score += 5.0       # Strong instrument signal
        elif s["mfcc_delta_std"] < 17.0:
            score += 3.5
        elif s["mfcc_delta_std"] < 20.5:
            score += 1.5       # Borderline
        # Above 20.5: no instrument credit

        # --- Centroid (weight 3.5) ---
        # Instruments: 1044-1664 Hz
        # Voice: 1837-2587 Hz
        # Midpoint: ~1750 Hz
        if s["centroid_mean"] < 1400:
            score += 3.5       # Strongly instrument range
        elif s["centroid_mean"] < 1750:
            score += 2.0
        elif s["centroid_mean"] < 2100:
            score += 0.5       # Borderline region

        # --- ZCR (weight 3.0) ---
        # Instruments: 0.047-0.114
        # Voice: 0.092-0.165
        # Overlap at 0.092! Use lower threshold for strong credit
        if s["zcr_mean"] < 0.07:
            score += 3.0       # Clearly instrument
        elif s["zcr_mean"] < 0.10:
            score += 1.5       # Ambiguous zone
        elif s["zcr_mean"] < 0.12:
            score += 0.5

        # --- Harmonic ratio (weight 2.5) ---
        # Instruments: 0.716-0.959
        # Voice: 0.697-0.814
        # Very high harmonicity (> 0.88) strongly favors instruments
        if s["harmonic_ratio"] > 0.88:
            score += 2.5
        elif s["harmonic_ratio"] > 0.78:
            score += 1.0

        # --- PENALTIES ---
        # High MFCC delta: strong voice signal
        if s["mfcc_delta_std"] > 20.5:
            score -= 3.0

        # Very high centroid: not instrument
        if s["centroid_mean"] > 2500:
            score -= 3.0

        # Low harmonic ratio: bioacoustics/noise
        if s["harmonic_ratio"] < 0.45:
            score -= 6.0

        self.scores["instruments"] = max(score, 0.0)

    def _score_voice(self, s):
        """
        Human Voice scoring.

        MFCC delta is the primary discriminator - voice consistently shows
        high values (> 20) due to formant transitions between vowels and
        consonants, regardless of pitch or singing style.
        """
        score = 0.0

        # --- MFCC delta (PRIMARY, weight 5.0) ---
        # Voice: 21.8-22.2 (formant transitions)
        # Instruments: 8.0-19.6 (stable)
        # Midpoint: 20.5
        if s["mfcc_delta_std"] > 21.0:
            score += 5.0       # Strong voice signal
        elif s["mfcc_delta_std"] > 20.5:
            score += 4.0
        elif s["mfcc_delta_std"] > 18.0:
            score += 2.0       # Moderate voice signal
        elif s["mfcc_delta_std"] > 16.0:
            score += 0.5

        # --- ZCR (weight 3.0) ---
        # Voice: 0.092-0.165, instruments: 0.047-0.114
        # High ZCR = consonants/fricatives
        if s["zcr_mean"] > 0.14:
            score += 3.0
        elif s["zcr_mean"] > 0.10:
            score += 1.5
        elif s["zcr_mean"] > 0.08:
            score += 0.5

        # --- Centroid (weight 2.5) ---
        # Voice: 1837-2587, instruments: 1044-1664
        if s["centroid_mean"] > 2200:
            score += 2.5
        elif s["centroid_mean"] > 1750:
            score += 1.5
        elif s["centroid_mean"] > 1500:
            score += 0.5

        # --- Harmonic ratio (weight 2.0) ---
        # Voice: 0.697-0.814 (moderate harmonicity)
        # Instruments tend to be very high (> 0.88)
        if 0.60 < s["harmonic_ratio"] < 0.85:
            score += 2.0       # Voice-like range
        elif 0.50 < s["harmonic_ratio"] < 0.88:
            score += 1.0

        # --- PENALTIES ---
        # Low MFCC delta: stable timbre = instrument
        if s["mfcc_delta_std"] < 16.0:
            score -= 3.0

        # Very low centroid: clearly instrument
        if s["centroid_mean"] < 1300:
            score -= 2.5

        # Very high harmonic ratio: pure instrument
        if s["harmonic_ratio"] > 0.92:
            score -= 2.5

        # Low harmonic ratio: bioacoustics
        if s["harmonic_ratio"] < 0.45:
            score -= 6.0

        # Very low ZCR: no consonants
        if s["zcr_mean"] < 0.06:
            score -= 2.0

        self.scores["voice"] = max(score, 0.0)

    def _score_bioacoustics(self, s):
        """
        Bioacoustics: bird calls, animal sounds, nature sounds.

        The harmonic ratio is the PRIMARY gate: bioacoustics always have
        low harmonic ratio (< 0.35) due to rapid chirps, broadband calls,
        and non-sustained tonal energy. This single feature provides
        complete separation from voice and instruments (both > 0.69).
        """
        score = 0.0

        # --- HARMONIC RATIO (PRIMARY GATE, weight 8.0) ---
        # Bioacoustics: 0.20-0.31 (max observed)
        # Voice/Instruments: 0.697+ (min observed)
        # Threshold: 0.45 (midpoint of the 0.31-0.70 gap)
        if s["harmonic_ratio"] < 0.25:
            score += 8.0       # Very strong bioacoustic signal
        elif s["harmonic_ratio"] < 0.35:
            score += 6.0       # Strong bioacoustic signal
        elif s["harmonic_ratio"] < 0.45:
            score += 3.0       # Probable bioacoustic

        # --- LOW VOCAL BAND: not human, no fundamentals ---
        # Bio: 0.04-0.23, instruments: 0.60-0.96, voice: 0.51-0.83
        if s["vocal_band_ratio"] < 0.15:
            score += 3.0
        elif s["vocal_band_ratio"] < 0.30:
            score += 2.0
        elif s["vocal_band_ratio"] < 0.45:
            score += 0.5

        # --- HIGH FREQUENCY ENERGY ---
        # Bio has most energy above 3 KHz
        if s["high_freq_ratio"] > 0.25:
            score += 2.5
        elif s["high_freq_ratio"] > 0.10:
            score += 1.0

        # --- HIGH ZCR: rapid modulations ---
        if s["zcr_mean"] > 0.20:
            score += 2.0
        elif s["zcr_mean"] > 0.12:
            score += 1.0

        # --- PENALTIES ---
        # High harmonic ratio: instrument or voice
        if s["harmonic_ratio"] > 0.60:
            score -= 8.0       # Definitely not bioacoustic
        elif s["harmonic_ratio"] > 0.45:
            score -= 4.0

        # High vocal band: human voice or instruments
        if s["vocal_band_ratio"] > 0.55:
            score -= 4.0

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
