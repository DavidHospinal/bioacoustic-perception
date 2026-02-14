# Sound-Visual: Audio-Visual Analysis

Python application that generates 3D geometric visualizations from audio file analysis.
Inspired by the work of Lucio Arese on multidimensional musical data visualization.

The application extracts physical properties of sound (spectral centroid, chromagram, RMS energy,
spectral bandwidth) and maps them to a 3D colored point cloud, synchronized in real time
with audio playback. After full playback, an AI classifier categorizes the audio into
Musical Instruments, Human Voice, or Bioacoustics, using rule-based scoring grounded in
empirically calibrated spectral descriptors.

**Author:** Oscar David Hospinal R.
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

After full playback completes, the system classifies the audio into one of three categories:
Musical Instruments, Human Voice, or Bioacoustics.

### How the Classifier Works

The classifier operates in two stages:

**Stage 1 - Feature Extraction.** The complete audio signal is processed using librosa to
compute frame-level spectral descriptors. These per-frame values are then averaged over the
entire file duration to produce a single set of global statistics. This "listen to everything
first" approach ensures the classifier captures the full spectral profile of the audio,
rather than making premature decisions based on partial data.

**Stage 2 - Rule-Based Scoring.** Each of the three categories receives a score computed
from the global statistics. The category with the highest score wins. Scores are built
by adding points when a descriptor falls within empirically calibrated ranges, and
subtracting points (penalties) when values strongly contradict the category.

### Spectral Descriptors Used for Classification

The classifier extracts and evaluates the following descriptors from the raw audio signal:

| Descriptor | Extraction Method | Physical Meaning |
|---|---|---|
| Spectral centroid | Mean of `librosa.feature.spectral_centroid` | Dominant frequency (Hz). Indicates where most energy is concentrated. Low for instruments (~1000-1700 Hz), mid-high for voice (~2500 Hz), high for birds (~3700 Hz). |
| Zero crossing rate (ZCR) | Mean of `librosa.feature.zero_crossing_rate` | Rate at which the signal crosses zero amplitude. High values indicate consonants, fricatives, or rapid modulations. Voice (~0.165) has higher ZCR than instruments (~0.05-0.11) due to consonant phonemes. |
| Harmonic ratio | HPSS decomposition via `librosa.effects.hpss` | Ratio of harmonic energy to total energy. Tuned instruments produce highly harmonic signals (0.72-0.96), voice is moderately harmonic (~0.70), and bioacoustics have low harmonicity (~0.20) due to rapid chirps. |
| MFCC delta stability | Standard deviation of frame-to-frame MFCC differences | Measures how quickly the timbral characteristics change over time. Voice has high values (>18) due to transitions between vowels and consonants (formant shifts). Instruments maintain stable timbre (<15). |
| Vocal band ratio | Energy in 80-1100 Hz band vs total (via STFT) | Proportion of energy in the fundamental vocal frequency range. High for instruments with low fundamentals (>0.60) and voice (~0.50), very low for birds (<0.10) that sing above 3 KHz. |
| High frequency ratio | Energy above 3 KHz vs total (via STFT) | Proportion of energy in high frequencies. Bioacoustics have the highest values (>0.25), while instruments and voice concentrate energy below 3 KHz. |
| Spectral rolloff | Mean of `librosa.feature.spectral_rolloff` | Frequency below which 85% of spectral energy is contained. Low rolloff indicates energy concentrated in low frequencies (instruments, voice). High rolloff indicates broad spectral spread (bioacoustics). |
| Spectral flatness | Mean of `librosa.feature.spectral_flatness` | Measures how noise-like (flat) vs tonal (peaked) the spectrum is. Pure tonal instruments have very low flatness (<0.01), voice is moderate (~0.03), environmental sounds are higher. |
| Chromatic variability | Frame-to-frame standard deviation of chromagram | Indicates polyphonic complexity. Orchestral music changes notes frequently (high variability), solo voice is more monophonic (low variability). |

### Decision Thresholds per Category

| Category | Key Discriminators |
|---|---|
| Musical Instruments | Low centroid (1000-1700 Hz), high harmonic ratio (>0.70), low ZCR (<0.12), stable MFCC delta (<15) |
| Human Voice | Mid-high centroid (2000-3500 Hz), high ZCR (>0.12), high MFCC delta (>18), moderate harmonic ratio (0.55-0.80) |
| Bioacoustics | Very high centroid (>3500 Hz), low harmonic ratio (<0.25), very low vocal band (<0.10), high ZCR (>0.20) |

All thresholds were empirically calibrated against 7 test audio files spanning classical
orchestral music, solo violin, organ, guitar, singing voice, and bird/nature recordings.
The classifier does not use the filename or any metadata; it operates exclusively on
the acoustic properties of the signal.

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

## References

### Audio-Visual Analysis and Sound Cognition

McAdams, S. (1993). Recognition of sound sources and events. In S. McAdams & E. Bigand (Eds.),
*Thinking in Sound: The Cognitive Psychology of Human Audition* (pp. 146-198). Oxford University Press.
DOI: 10.1093/acprof:oso/9780198524897.003.0006.
PDF: https://www.mcgill.ca/mpcl/files/mpcl/mcadams_thinkingsound_1993.pdf

Wattenberg, M. (2002). Arc Diagrams: Visualizing Structure in Strings.
*IEEE Symposium on Information Visualization (InfoVis 2002)*.
DOI: 10.1109/INFVIS.2002.1173157.
PDF: http://hint.fm/papers/arc-diagrams.pdf

Mueller, M. (2007). *Information Retrieval for Music and Motion*. Springer.
ISBN: 978-3-540-74047-6. DOI: 10.1007/978-3-540-74048-3.
https://books.google.com/books/about/Information_Retrieval_for_Music_and_Moti.html?id=uhwIvgAACAAJ

Dieleman, S. & Schrauwen, B. (2014). End-to-end learning for music audio tagging at scale.
*Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2014)*.
DOI: 10.1109/ICASSP.2014.6853581.
PDF: http://dihana.cps.unizar.es/proceedings/ICASSP/2014/papers/p7014-dieleman.pdf

### Spectral Centroid and Rolloff

Ghisingh, L. & Pandey, K. K. (2018). Music Genre Classification Using Spectral Analysis.
*arXiv preprint arXiv:1803.04652*.
PDF: https://arxiv.org/pdf/1803.04652.pdf

Classification of Biological Sounds Using Spatial Directivity (2025).
*Forum Acusticum 2025*.
PDF: https://dael.euracoustics.org/confs/fa2025/data/articles/000816.pdf

### Zero Crossing Rate (ZCR)

Bachu, R. G. et al. (2011). Noise robust zero-crossing rate computation for audio signal classification.
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2011)*.
DOI: 10.1109/ICASSP.2011.5947564.

### Harmonic Ratio and Harmonicity

Marxer, R. et al. (2025). Bioacoustic fundamental frequency estimation.
*Accepted Manuscript*.
PDF: https://www.ricardmarxer.com/assets/f0-examples/Accepted_Manuscript_Best_Marxer_et_al_2025.pdf

### MFCC Delta Stability

Kumar, K., Kim, C. & Stern, R. M. (2011). Delta-spectral cepstral coefficients for robust speech recognition.
*IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2011)*.
PDF: https://www.cs.cmu.edu/afs/cs/user/robust/www/Papers/KumarKimSternICA11.pdf

Jothilakshmi, S. (2014). Delta-MFCC Based Text-Independent Speaker Recognition.
*International Journal of Engineering Development and Research (IJEDR), 2(3)*.
PDF: https://rjwave.org/ijedr/viewpaperforall.php?paper=IJEDR1403017

### Band Ratios and Spectral Flatness

Herrera, P. et al. Automatic Classification of Musical Instrument Sounds.
*Universitat Pompeu Fabra*.
PDF: https://repositori.upf.edu/bitstreams/2ddfbd22-c9be-428a-9277-

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation

If you use this project in your research or work, please cite it as follows:

```bibtex
@misc{sound-visual-bioacoustic-perception,
    title = {Sound-Visual: Audio-Visual Analysis with AI Classification},
    type = {Open Source Software},
    author = {Hospinal R., Oscar David},
    howpublished = {\url{https://github.com/DavidHospinal/-bioacoustic-perception}},
    url = {https://github.com/DavidHospinal/-bioacoustic-perception},
    institution = {Pontificia Universidad Catolica de Chile},
    year = {2026},
    month = {feb}
}
```
