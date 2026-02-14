"""
Renderiza conexiones armonicas entre nodos de la misma nota musical.
Usa GL_LINES para segmentos individuales (no line_strip secuencial).
Lineas finas y semi-transparentes que forman la telarana/constelacion.
"""
import numpy as np
import pyqtgraph.opengl as gl
import config


class TrailRenderer:
    """
    Dibuja conexiones armonicas entre puntos que comparten
    la misma nota dominante y estan cercanos espacialmente.
    """

    def __init__(self, view):
        self.view = view
        self._last_count = 0

        # Conexiones armonicas (segmentos individuales)
        self.connection_lines = gl.GLLinePlotItem(
            mode="lines",
            width=config.CONNECTION_LINE_WIDTH,
            antialias=True,
            glOptions="translucent",
        )
        self.view.addItem(self.connection_lines)

        # Trail del punto actual (ultima secuencia reciente)
        self.active_trail = gl.GLLinePlotItem(
            mode="line_strip",
            width=config.TRAIL_LINE_WIDTH,
            antialias=True,
            glOptions="translucent",
        )
        self.view.addItem(self.active_trail)

    def update_connections(self, conn_positions, conn_colors):
        """
        Actualiza las conexiones armonicas visibles.
        Solo actualiza cuando el conteo cambia.
        """
        n = len(conn_positions)
        if n != self._last_count:
            self._last_count = n
            if n < 2:
                self.connection_lines.setData(
                    pos=np.zeros((0, 3), dtype=np.float32),
                    color=np.zeros((0, 4), dtype=np.float32),
                )
            else:
                self.connection_lines.setData(
                    pos=conn_positions,
                    color=conn_colors,
                )

    def update_active_trail(self, positions, colors, trail_start, trail_end):
        """
        Actualiza el trail reciente (ultimos N puntos) como linea continua
        blanca para mostrar la trayectoria actual del playhead.
        """
        t_start = max(trail_start, 0)
        t_end = min(trail_end, len(positions))
        trail_n = t_end - t_start

        if trail_n < 2:
            self.active_trail.setData(
                pos=np.zeros((0, 3), dtype=np.float32),
                color=np.zeros((0, 4), dtype=np.float32),
            )
            return

        trail_pos = positions[t_start:t_end]

        # Linea blanca con fade progresivo (mas reciente = mas brillante)
        trail_colors = np.ones((trail_n, 4), dtype=np.float32)
        indices = np.arange(trail_n, dtype=np.float32)
        fade = np.clip(indices / max(float(trail_n - 1), 1.0), 0.02, 0.5)
        trail_colors[:, 3] = fade

        self.active_trail.setData(
            pos=trail_pos,
            color=trail_colors,
        )

    def clear(self):
        """Limpia todas las lineas."""
        self._last_count = 0
        empty_pos = np.zeros((0, 3), dtype=np.float32)
        empty_color = np.zeros((0, 4), dtype=np.float32)
        self.connection_lines.setData(pos=empty_pos, color=empty_color)
        self.active_trail.setData(pos=empty_pos, color=empty_color)
