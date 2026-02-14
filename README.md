# Sound-Visual: Audio-Visual Analysis

Python application that generates 3D geometric visualizations from audio file analysis.
Inspired by the work of Lucio Arese on multidimensional musical data visualization.

The application extracts physical properties of sound (spectral centroid, chromagram, RMS energy,
spectral bandwidth) and maps them to a 3D colored point cloud, synchronized in real time
with audio playback. After full playback, an AI classifier categorizes the audio into
Musical Instruments, Human Voice, or Bioacoustics, using rule-based scoring grounded in
empirically calibrated spectral descriptors.

**Author:** Oscar David Hospinal Bibiano
**Contact:** oscardavid.hospinal@uc.cl
**Institution:** Pontificia Universidad Catolica de Chile

## Technology Stack

- **PyQt6** - GUI framework
- **pyqtgraph** - 3D OpenGL rendering (GLViewWidget, GLScatterPlotItem, GLLinePlotItem)
- **librosa** - Scientific audio analysis (spectral centroid, chromagram, MFCCs)
- **sounddevice** - Audio playback with frame-accurate synchronization
- **NumPy / SciPy** - Numerical operations and signal processing

## Audio to Visual Mapping

| Visual Property | Audio Feature | Description |
|---|---|---|
| Position X | Time (s) | Temporal progression |
| Position Y | Spectral centroid (Hz) | Sound "brightness" (0-12 KHz) |
| Position Z | Spectral bandwidth | Spectral spread |
| Point size | RMS energy | Intensity / amplitude |
| Color | Chromagram | Dominant musical note (12 notes -> HSV wheel) |
| Trail | Temporal history | Past points with decreasing alpha |

## AI Classification

After full playback, the system classifies the audio into one of three categories
using global spectral descriptors averaged over the entire file:

| Category | Key Discriminators |
|---|---|
| Musical Instruments | Low centroid (1000-1700 Hz), high harmonic ratio (>0.70), low ZCR (<0.12), stable MFCC delta (<15) |
| Human Voice | Mid-high centroid (2000-3500 Hz), high ZCR (>0.12), high MFCC delta (>18), moderate harmonic ratio |
| Bioacoustics | Very high centroid (>3500 Hz), low harmonic ratio (<0.25), very low vocal band (<0.10), high ZCR (>0.20) |

## System Requirements

- Python 3.10+
- System with OpenGL support (WSLg on Windows, X11/Wayland on Linux)
- ffmpeg (for MP3 decoding)
- PortAudio (for audio playback)

## Installation

```bash
# System dependencies (Ubuntu/Debian)
sudo apt install python3-pip python3-venv python3-dev ffmpeg libportaudio2 libsndfile1

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### WSL2 Configuration

If running on WSL2 with WSLg, ensure these environment variables are set:

```bash
export DISPLAY=:0
export PULSE_SERVER=unix:/mnt/wslg/PulseServer
```

In PyCharm, add these variables in Run/Debug Configurations -> Environment Variables.

## Running

```bash
source .venv/bin/activate
python main.py
```

## Usage

1. Click "Open File..." or use the File -> Open Audio menu
2. Select an audio file (MP3, WAV, FLAC, OGG)
3. The application analyzes the audio and prepares the visualization
4. Press "Play" to start synchronized playback
5. After playback ends, the AI classifier displays the predicted category
6. Use the controls to adjust trail length and point scale
7. Rotate the 3D view with the mouse (left click + drag)

## Project Structure

```
sound-visual/
    main.py              # Entry point
    config.py            # Global constants
    requirements.txt     # Dependencies
    core/
        audio_analyzer.py    # Feature extraction (librosa)
        audio_player.py      # Playback (sounddevice)
        audio_classifier.py  # AI rule-based classifier (3 categories)
        feature_mapper.py    # Feature -> visual property mapping
        color_mapping.py     # Chromagram -> HSV colors
    visualization/
        scene_manager.py         # 3D scene (camera, background, grid)
        point_cloud_renderer.py  # Point cloud (scatter + glow)
        trail_renderer.py        # Connection lines
        animation_controller.py  # Synchronized animation loop
    gui/
        main_window.py   # Main window (dark theme)
        control_panel.py # Control panel
        file_loader.py   # File loading utilities
```

## Supported Audio Formats

- MP3 (requires ffmpeg)
- WAV
- FLAC
- OGG/Vorbis
