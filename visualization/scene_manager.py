"""
Configura el GLViewWidget: fondo, camara centrada, grilla sutil
y marca de autor. Camara con auto-rotacion 360 grados durante
la reproduccion y tracking suave del playhead X.
"""
import numpy as np
import pyqtgraph.opengl as gl
import config


class SceneManager:
    """
    Gestiona el viewport 3D OpenGL con camara auto-rotante
    que orbita la constelacion durante la reproduccion.
    """

    def __init__(self):
        self.view = gl.GLViewWidget()
        self._setup_background()
        self._setup_camera()
        self._setup_grid()
        self._setup_author_mark()

        self._current_azimuth = config.CAMERA_AZIMUTH
        self._target_center_x = 0.0
        self._current_center_x = 0.0
        self._is_playing = False

    def _setup_background(self):
        self.view.setBackgroundColor(*config.BACKGROUND_COLOR)

    def _setup_camera(self):
        self.view.opts["distance"] = config.CAMERA_DISTANCE
        self.view.opts["elevation"] = config.CAMERA_ELEVATION
        self.view.opts["azimuth"] = config.CAMERA_AZIMUTH
        self.view.opts["fov"] = config.CAMERA_FOV
        self.view.opts["center"] = pg_vector(
            0.0, config.CAMERA_Y_CENTER, 0.0)
        self._current_azimuth = config.CAMERA_AZIMUTH
        self._current_center_x = 0.0

    def _setup_grid(self):
        self._grid = gl.GLGridItem()
        self._grid.setSize(60, 60, 1)
        self._grid.setSpacing(5, 5, 1)
        self._grid.setColor(config.GRID_COLOR)
        self.view.addItem(self._grid)

    def _setup_author_mark(self):
        """Agrega la marca de autor en esquina inferior-derecha de la escena."""
        try:
            self._author_label = gl.GLTextItem(
                pos=np.array([25, -3, 25], dtype=np.float32),
                text="Developer: Msc. David Hospinal",
                color=(255, 255, 255, 55),
            )
            self.view.addItem(self._author_label)
        except Exception:
            self._author_label = None

    def set_playing(self, playing):
        """Activa/desactiva la auto-rotacion."""
        self._is_playing = playing

    def update_camera(self, playhead_x, dt):
        """
        Actualiza la camara cada frame:
        1. Auto-rotacion: incrementa azimuth durante playback
        2. Tracking: sigue el playhead X con lerp suave
        """
        if not self._is_playing:
            return

        # Auto-rotacion 360 grados
        if config.CAMERA_AUTO_ROTATE:
            self._current_azimuth += config.CAMERA_ROTATION_SPEED * dt
            if self._current_azimuth > 360.0:
                self._current_azimuth -= 360.0
            self.view.opts["azimuth"] = self._current_azimuth

        # Tracking suave del playhead X
        self._target_center_x = playhead_x
        self._current_center_x += (
            (self._target_center_x - self._current_center_x)
            * config.CAMERA_SMOOTHING
        )
        self.view.opts["center"] = pg_vector(
            self._current_center_x, config.CAMERA_Y_CENTER, 0.0)

    def get_widget(self):
        return self.view

    def reset_camera(self):
        """Reinicia la camara a la posicion por defecto."""
        self._is_playing = False
        self._setup_camera()


def pg_vector(x, y, z):
    """Crea un QVector3D compatible con pyqtgraph."""
    from pyqtgraph import Vector
    return Vector(x, y, z)
