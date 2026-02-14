"""
Widget de leyenda espectral 2D superpuesto sobre el viewport 3D.
Barra vertical con degradado de colores mapeando Spectral Centroid
(0 - 12 KHz) a los matices HSV de las notas cromaticas.
"""
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QFont, QPen
from colorsys import hsv_to_rgb


# Frecuencias y colores para la leyenda (mapeo simplificado)
LEGEND_STEPS = [
    (0.0, 0 / 360),       # C  - rojo
    (1.5, 30 / 360),      # C# - naranja
    (3.0, 60 / 360),      # D  - amarillo
    (4.5, 120 / 360),     # E  - verde
    (6.0, 180 / 360),     # F# - cian
    (7.5, 210 / 360),     # G  - azul claro
    (9.0, 240 / 360),     # A  - azul
    (10.5, 300 / 360),    # A# - magenta
    (12.0, 330 / 360),    # B  - rosa
]


class SpectralLegend(QWidget):
    """
    Barra de leyenda vertical con degradado HSV representando
    el rango del Spectral Centroid de 0 a 12 KHz.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(90)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Dimensiones de la barra
        bar_x = 12
        bar_w = 22
        bar_top = 50
        bar_bottom = h - 30
        bar_h = bar_bottom - bar_top

        if bar_h < 50:
            painter.end()
            return

        # Fondo semitranslucido
        painter.fillRect(0, 0, w, h, QColor(0, 0, 0, 100))

        # Titulo
        title_font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(4, 15, "SPECTRAL")
        painter.drawText(4, 28, "CENTROID")

        # Degradado vertical (de arriba = alta freq a abajo = baja freq)
        gradient = QLinearGradient(bar_x, bar_top, bar_x, bar_bottom)

        for khz, hue in reversed(LEGEND_STEPS):
            # Posicion normalizada (0 = top = 12 KHz, 1 = bottom = 0 KHz)
            t = 1.0 - (khz / 12.0)
            r, g, b = hsv_to_rgb(hue, 0.9, 1.0)
            gradient.setColorAt(t, QColor(
                int(r * 255), int(g * 255), int(b * 255)))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawRoundedRect(
            int(bar_x), int(bar_top), int(bar_w), int(bar_h), 3, 3)

        # Borde de la barra
        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(
            int(bar_x), int(bar_top), int(bar_w), int(bar_h), 3, 3)

        # Etiquetas de frecuencia
        label_font = QFont("Arial", 7)
        painter.setFont(label_font)
        painter.setPen(QColor(180, 180, 180))

        label_x = bar_x + bar_w + 4
        for khz, _ in LEGEND_STEPS:
            # Posicion Y en la barra
            t = 1.0 - (khz / 12.0)
            y = bar_top + t * bar_h

            if khz == int(khz):
                text = f"{int(khz)}.0 KHz"
            else:
                text = f"{khz:.1f} KHz"

            painter.drawText(int(label_x), int(y + 4), text)

            # Marca horizontal pequena
            painter.drawLine(
                int(bar_x + bar_w - 3), int(y),
                int(bar_x + bar_w + 2), int(y))

        painter.end()
