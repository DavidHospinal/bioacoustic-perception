"""
Renderiza la constelacion de puntos con colores vividos persistentes.
Un solo scatter para todos los puntos + glow reducido para el activo.
"""
import numpy as np
import pyqtgraph.opengl as gl
import config


class PointCloudRenderer:
    """
    Renderiza puntos de la constelacion armonica.
    - main_scatter: todos los puntos acumulados con colores vividos
    - current_scatter + glow: punto activo con brillo sutil
    """

    def __init__(self, view):
        self.view = view
        self._last_count = 0

        # Todos los puntos de la constelacion
        self.main_scatter = gl.GLScatterPlotItem(
            pxMode=True,
            glOptions="translucent",
        )
        self.view.addItem(self.main_scatter)

        # Nucleo brillante del punto actual
        self.current_scatter = gl.GLScatterPlotItem(
            pxMode=True,
            glOptions="additive",
        )
        self.view.addItem(self.current_scatter)

        # Halo del punto actual
        self._glow_halo = gl.GLScatterPlotItem(
            pxMode=True,
            glOptions="additive",
        )
        self.view.addItem(self._glow_halo)

        # Resplandor exterior
        self._glow_outer = gl.GLScatterPlotItem(
            pxMode=True,
            glOptions="additive",
        )
        self.view.addItem(self._glow_outer)

    def update(self, positions, colors, sizes, current_idx):
        """
        Actualiza la constelacion. Todos los puntos mantienen
        colores vividos. Solo el punto actual tiene glow.
        """
        n = len(positions)
        if n == 0:
            return

        # Actualizar scatter principal solo cuando hay nuevos puntos
        if n != self._last_count:
            self._last_count = n
            self.main_scatter.setData(
                pos=positions,
                color=colors,
                size=sizes,
            )

        # Glow del punto actual (reducido)
        if 0 <= current_idx < n:
            curr_pos = positions[current_idx:current_idx + 1]
            curr_color = colors[current_idx].copy()
            base_size = sizes[current_idx]

            # Nucleo blanco brillante
            self.current_scatter.setData(
                pos=curr_pos,
                color=np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
                size=np.array([base_size * config.GLOW_CORE_SCALE]),
            )

            # Halo con color de nota
            halo_color = curr_color.copy()
            halo_color[3] = config.GLOW_HALO_ALPHA
            self._glow_halo.setData(
                pos=curr_pos,
                color=np.array([halo_color], dtype=np.float32),
                size=np.array([base_size * config.GLOW_HALO_SCALE]),
            )

            # Resplandor exterior difuso
            outer_color = curr_color.copy()
            outer_color[3] = config.GLOW_OUTER_ALPHA
            self._glow_outer.setData(
                pos=curr_pos,
                color=np.array([outer_color], dtype=np.float32),
                size=np.array([base_size * config.GLOW_OUTER_SCALE]),
            )

    def clear(self):
        """Limpia todos los puntos."""
        self._last_count = 0
        empty = np.zeros((0, 3), dtype=np.float32)
        empty_color = np.zeros((0, 4), dtype=np.float32)
        empty_size = np.zeros(0, dtype=np.float32)
        for item in [self.main_scatter, self.current_scatter,
                     self._glow_halo, self._glow_outer]:
            item.setData(pos=empty, color=empty_color, size=empty_size)
