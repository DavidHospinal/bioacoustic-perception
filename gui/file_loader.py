"""
Utilities for loading and validating audio files.
"""
import os

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg"}


def validate_audio_file(file_path):
    """Checks if the file exists and has a supported extension."""
    if not os.path.isfile(file_path):
        return False
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def get_file_info(file_path):
    """Returns basic file metadata."""
    return {
        "name": os.path.basename(file_path),
        "path": file_path,
        "size_mb": os.path.getsize(file_path) / (1024 * 1024),
        "extension": os.path.splitext(file_path)[1].lower(),
    }


def get_audio_filter_string():
    """Returns filter string for QFileDialog."""
    return (
        "Audio Files (*.mp3 *.wav *.flac *.ogg);;"
        "MP3 Files (*.mp3);;"
        "WAV Files (*.wav);;"
        "FLAC Files (*.flac);;"
        "OGG Files (*.ogg);;"
        "All Files (*)"
    )
