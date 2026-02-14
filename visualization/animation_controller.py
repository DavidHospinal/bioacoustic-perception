"""
Loop de animacion que sincroniza la constelacion 3D con la
reproduccion de audio. Revela puntos y conexiones armonicas
progresivamente conforme avanza la musica. Controla la
auto-rotacion de camara y el tracking del playhead.
"""
import time
from PyQt6.QtCore import QTimer, QObject, pyqtSignal
import numpy as np
import config


class AnimationController(QObject):
    """
    Controla el loop de la visualizacion.
    En cada tick:
    1. Lee el tiempo actual del reproductor
    2. Revela puntos acumulados (0..current_frame)
    3. Revela conexiones armonicas cuyos endpoints ya estan visibles
    4. Actualiza trail activo del playhead
    5. Actualiza auto-rotacion y tracking de camara
    """

    frame_updated = pyqtSignal(float)
    playback_finished = pyqtSignal()

    def __init__(self, audio_player, feature_mapper, features,
                 point_cloud, trail_renderer, scene_manager):
        super().__init__()
        self.player = audio_player
        self.mapper = feature_mapper
        self.features = features
        self.point_cloud = point_cloud
        self.trail = trail_renderer
        self.scene = scene_manager

        self.timer = QTimer()
        self.timer.setInterval(int(1000 / config.TARGET_FPS))
        self.timer.timeout.connect(self._tick)

        self.times = features["times"]
        self.trail_length = config.TRAIL_LENGTH
        self._conn_update_counter = 0
        self._last_tick_time = None

    def start(self):
        self._last_tick_time = time.monotonic()
        self.scene.set_playing(True)
        self.timer.start()

    def stop(self):
        self.timer.stop()
        self.scene.set_playing(False)
        self._last_tick_time = None

    def set_trail_length(self, length):
        """Actualiza la longitud del trail en tiempo real."""
        self.trail_length = max(10, length)

    def _tick(self):
        """Se ejecuta cada frame."""
        current_time = self.player.current_time
        self.frame_updated.emit(current_time)

        if self.player.is_finished:
            self.stop()
            self.playback_finished.emit()
            return

        # Delta time para auto-rotacion suave
        now = time.monotonic()
        dt = now - self._last_tick_time if self._last_tick_time else 1.0 / config.TARGET_FPS
        self._last_tick_time = now

        current_frame = self.mapper.time_to_frame(current_time, self.times)
        total_end = min(current_frame + 1, self.mapper.n_frames)

        if total_end <= 0:
            return

        # Todos los puntos acumulados (0..current_frame)
        data = self.mapper.get_range_data(0, total_end)

        # Actualizar constelacion de puntos
        self.point_cloud.update(
            data["positions"], data["colors"],
            data["sizes"], current_frame)

        # Conexiones armonicas (actualizar cada 5 frames para rendimiento)
        self._conn_update_counter += 1
        if self._conn_update_counter >= 5:
            self._conn_update_counter = 0
            conn_pos, conn_colors = self.mapper.get_visible_connections(
                current_frame)
            self.trail.update_connections(conn_pos, conn_colors)

        # Trail activo del playhead
        trail_start = max(0, current_frame - self.trail_length)
        self.trail.update_active_trail(
            data["positions"], data["colors"],
            trail_start, total_end)

        # Auto-rotacion y tracking de camara
        playhead_x = data["positions"][current_frame, 0] if current_frame < len(data["positions"]) else 0.0
        self.scene.update_camera(playhead_x, dt)
