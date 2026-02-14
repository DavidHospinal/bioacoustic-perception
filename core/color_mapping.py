"""
Convierte datos de cromatograma a colores RGBA usando la rueda HSV.
Cada una de las 12 notas cromaticas se mapea a un segmento de 30 grados del hue.
"""
import numpy as np
from colorsys import hsv_to_rgb

NOTE_HUES_NORMALIZED = np.array([
    0 / 360, 30 / 360, 60 / 360, 90 / 360, 120 / 360, 150 / 360,
    180 / 360, 210 / 360, 240 / 360, 270 / 360, 300 / 360, 330 / 360
])

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]


def chroma_to_rgba(chroma_vector, alpha=1.0):
    """
    Convierte un frame de cromatograma (12,) a un color RGBA (4,).

    La nota dominante determina el hue.
    La energia de la nota dominante determina la saturacion.
    El valor (brillo) siempre es 1.0 para colores vivos sobre fondo oscuro.
    """
    chroma_sum = chroma_vector.sum()
    if chroma_sum < 1e-10:
        return np.array([0.3, 0.3, 0.3, alpha], dtype=np.float32)

    chroma_norm = chroma_vector / chroma_sum
    dominant_idx = np.argmax(chroma_vector)

    hue = NOTE_HUES_NORMALIZED[dominant_idx]

    dominance = chroma_norm[dominant_idx]
    saturation = 0.4 + 0.6 * min(1.0, dominance * 3.0)

    value = 1.0

    r, g, b = hsv_to_rgb(hue, saturation, value)
    return np.array([r, g, b, alpha], dtype=np.float32)


def chroma_to_rgba_batch(chroma_matrix, alphas=None):
    """
    Convierte matriz de cromatograma (12, T) a array RGBA (T, 4).
    """
    n_frames = chroma_matrix.shape[1]
    if alphas is None:
        alphas = np.ones(n_frames, dtype=np.float32)

    colors = np.zeros((n_frames, 4), dtype=np.float32)
    for i in range(n_frames):
        colors[i] = chroma_to_rgba(chroma_matrix[:, i], alphas[i])

    return colors
