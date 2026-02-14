"""
Audio feature extraction using librosa.
Computes all features in a single pass with consistent temporal alignment.
Applies Savitzky-Golay smoothing for fluid, organic trajectories.
"""
import numpy as np
import librosa
from scipy.signal import savgol_filter
import config


class AudioAnalyzer:
    """
    Loads an audio file and extracts temporally aligned features:
    - Spectral centroid (Hz)    -> Y axis (smoothed)
    - Chromagram (12 bins)      -> Color
    - RMS energy                -> Point size
    - Spectral bandwidth        -> Z axis (smoothed)
    - MFCCs                     -> Reserved for extensions
    """

    def __init__(self, sr=None, hop_length=None, n_fft=None):
        self.sr = sr or config.SAMPLE_RATE
        self.hop_length = hop_length or config.HOP_LENGTH
        self.n_fft = n_fft or config.N_FFT
        self.y = None
        self.duration = 0.0
        self.features = None

    def load(self, file_path):
        """Loads audio file, resamples to target SR. Returns raw signal."""
        self.y, _ = librosa.load(file_path, sr=self.sr, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        return self.y

    def _smooth(self, signal):
        """Applies Savitzky-Golay filter to smooth the signal."""
        win = config.SMOOTH_WINDOW
        if len(signal) < win:
            return signal
        if win % 2 == 0:
            win += 1
        return savgol_filter(signal, window_length=win,
                             polyorder=config.SMOOTH_POLYORDER)

    def analyze(self):
        """Runs full feature extraction. Returns dict of aligned arrays."""
        if self.y is None:
            raise ValueError("No audio loaded. Call load() first.")

        S = np.abs(librosa.stft(self.y, n_fft=self.n_fft,
                                hop_length=self.hop_length))

        centroid_raw = librosa.feature.spectral_centroid(
            S=S, sr=self.sr)[0]
        centroid = self._smooth(centroid_raw)

        chroma = librosa.feature.chroma_stft(
            S=S, sr=self.sr, n_chroma=config.N_CHROMA)

        rms_raw = librosa.feature.rms(S=S)[0]
        rms = self._smooth(rms_raw)
        rms = np.clip(rms, 0, None)

        bandwidth_raw = librosa.feature.spectral_bandwidth(
            S=S, sr=self.sr)[0]
        bandwidth = self._smooth(bandwidth_raw)

        mfcc = librosa.feature.mfcc(
            S=librosa.power_to_db(S ** 2),
            sr=self.sr, n_mfcc=config.N_MFCC)

        n_frames = centroid.shape[0]
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=self.sr, hop_length=self.hop_length)

        self.features = {
            "times": times,
            "centroid": centroid,
            "chroma": chroma,
            "rms": rms,
            "bandwidth": bandwidth,
            "mfcc": mfcc,
            "n_frames": n_frames,
            "duration": self.duration,
        }
        return self.features

    def get_raw_audio(self):
        """Returns raw audio signal for playback."""
        return self.y

    def get_sample_rate(self):
        """Returns the analysis sample rate."""
        return self.sr
