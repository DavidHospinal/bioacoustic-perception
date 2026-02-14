"""
Sound-Visual: Audio-Visual Analysis Application
Punto de entrada de la aplicacion PyQt6.

Visualizacion geometrica 3D de propiedades de audio inspirada
en el trabajo de Lucio Arese. Mapea centroide espectral, cromatograma,
energia RMS y ancho de banda a una nube de puntos coloreados
sincronizada con la reproduccion del audio.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configurar variables de entorno WSLg si no estan definidas
if "WSL_DISTRO_NAME" in os.environ:
    os.environ.setdefault("DISPLAY", ":0")
    os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    os.environ.setdefault("XDG_RUNTIME_DIR", "/mnt/wslg/runtime-dir")
    os.environ.setdefault("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Sound-Visual")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
