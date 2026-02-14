"""
Reproduccion de audio usando sounddevice con seguimiento de posicion en tiempo real.
Usa un OutputStream basado en callback para reproduccion no bloqueante.
"""
import threading
import time
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except OSError:
    sd = None
    SOUNDDEVICE_AVAILABLE = False


class AudioPlayer:
    """
    Reproduce audio via sounddevice y expone la posicion actual
    de reproduccion en segundos para sincronizacion con la visualizacion.
    """

    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        self.audio_data = None
        self.stream = None
        self._position = 0
        self._playing = False
        self._paused = False
        self._lock = threading.Lock()
        self._on_finished = None
        self._sim_start_time = 0.0
        self._sim_start_pos = 0
        self.audio_available = SOUNDDEVICE_AVAILABLE

    def load(self, audio_data, sample_rate):
        """Carga datos de audio para reproduccion."""
        self.audio_data = audio_data.astype(np.float32)
        self.sr = sample_rate
        self._position = 0

    def set_on_finished(self, callback):
        """Registra callback que se ejecuta cuando termina la reproduccion."""
        self._on_finished = callback

    def _callback(self, outdata, frames, time_info, status):
        """Callback de sounddevice. Se ejecuta desde el hilo de audio."""
        with self._lock:
            if self._paused or self.audio_data is None:
                outdata[:] = 0
                return

            start = self._position
            end = start + frames

            if end >= len(self.audio_data):
                valid = len(self.audio_data) - start
                if valid > 0:
                    outdata[:valid, 0] = self.audio_data[start:start + valid]
                outdata[valid:] = 0
                self._position = len(self.audio_data)
                self._playing = False
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = self.audio_data[start:end]
                self._position = end

    def _on_stream_finished(self):
        """Callback cuando el stream termina."""
        self._playing = False
        if self._on_finished:
            self._on_finished()

    def play(self):
        """Inicia o reanuda la reproduccion."""
        if self.audio_data is None:
            return
        if self._paused:
            self._paused = False
            self._sim_start_time = time.time()
            self._sim_start_pos = self._position
            return

        self.stop()
        self._position = 0
        self._playing = True
        self._paused = False

        if SOUNDDEVICE_AVAILABLE:
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=1024,
                finished_callback=self._on_stream_finished,
            )
            self.stream.start()

        self._sim_start_time = time.time()
        self._sim_start_pos = 0

    def pause(self):
        """Pausa la reproduccion (mantiene posicion)."""
        self._paused = True

    def resume(self):
        """Reanuda la reproduccion despues de una pausa."""
        if self._paused and self._playing:
            self._paused = False

    def stop(self):
        """Detiene la reproduccion y reinicia la posicion."""
        self._playing = False
        self._paused = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self._position = 0

    def seek(self, seconds):
        """Salta a una posicion temporal especifica."""
        with self._lock:
            if self.audio_data is not None:
                self._position = int(seconds * self.sr)
                self._position = max(0, min(self._position,
                                            len(self.audio_data) - 1))

    @property
    def current_time(self):
        """Posicion actual de reproduccion en segundos."""
        with self._lock:
            if self.audio_data is None:
                return 0.0
            if not SOUNDDEVICE_AVAILABLE and self._playing and not self._paused:
                elapsed = time.time() - self._sim_start_time
                sim_pos = self._sim_start_pos + int(elapsed * self.sr)
                if sim_pos >= len(self.audio_data):
                    self._position = len(self.audio_data)
                    self._playing = False
                else:
                    self._position = sim_pos
            return self._position / self.sr

    @property
    def duration(self):
        """Duracion total del audio en segundos."""
        if self.audio_data is None:
            return 0.0
        return len(self.audio_data) / self.sr

    @property
    def is_playing(self):
        return self._playing and not self._paused

    @property
    def is_paused(self):
        return self._paused

    @property
    def is_finished(self):
        if self.audio_data is None:
            return True
        return self._position >= len(self.audio_data)
