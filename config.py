"""
Application-wide constants and default parameters.
"""

# Audio analysis
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MFCC = 13
N_CHROMA = 12

# Smoothing (Savitzky-Golay filter for spectral centroid and bandwidth)
SMOOTH_WINDOW = 15
SMOOTH_POLYORDER = 3

# Visualization
TARGET_FPS = 30
TRAIL_LENGTH = 200
POINT_SIZE_MIN = 2.0
POINT_SIZE_MAX = 30.0
POINT_SIZE_SCALE = 35.0

# Axis ranges
SPECTRAL_CENTROID_MAX = 12000.0
Y_SCENE_MAX = 30.0
Z_SCENE_RANGE = 15.0

# Linear time mapping: X grows with time, no circular wrapping
TIME_SCALE = 0.5

# Harmonic connections
MAX_CONNECTIONS_PER_POINT = 5
MAX_CONNECTION_DISTANCE = 12.0
CONNECTION_LINE_ALPHA = 0.35
CONNECTION_LINE_WIDTH = 1.0

# Camera defaults
CAMERA_DISTANCE = 55
CAMERA_ELEVATION = 20
CAMERA_AZIMUTH = -45
CAMERA_FOV = 50
CAMERA_Y_CENTER = 12.0

# Camera auto-rotation during playback
CAMERA_AUTO_ROTATE = True
CAMERA_ROTATION_SPEED = 12.0

# Camera tracking (follows playhead X)
CAMERA_SMOOTHING = 0.06

# Color mapping: 12 chromatic notes to HSV hue (degrees)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]
NOTE_HUES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

# Rendering
BACKGROUND_COLOR = (0, 0, 0, 255)
TRAIL_LINE_WIDTH = 0.8
GRID_COLOR = (15, 15, 15, 25)

# Glow
GLOW_CORE_SCALE = 1.5
GLOW_HALO_SCALE = 3.0
GLOW_HALO_ALPHA = 0.30
GLOW_OUTER_SCALE = 5.0
GLOW_OUTER_ALPHA = 0.06

# Persistent alpha (vivid colors for all points)
PERSISTENT_ALPHA = 0.85

# Audio playback
AUDIO_BLOCK_SIZE = 1024
