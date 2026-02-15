"""
Hash-based feature cache for instant reload of previously analyzed audio files.
Uses file path + size + modification time as cache key.
Stores raw audio, extracted features, and classification results.
"""
import os
import hashlib
import pickle


class FeatureCache:
    """Caches audio analysis results to avoid redundant processing."""

    CACHE_DIR = ".cache"

    @classmethod
    def _base_dir(cls):
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @classmethod
    def _cache_dir(cls):
        d = os.path.join(cls._base_dir(), cls.CACHE_DIR)
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def _key(cls, file_path):
        """Generate cache key from file identity (path + size + mtime)."""
        try:
            stat = os.stat(file_path)
            data = f"{os.path.abspath(file_path)}|{stat.st_size}|{stat.st_mtime_ns}"
            return hashlib.sha256(data.encode()).hexdigest()[:16]
        except OSError:
            return None

    @classmethod
    def _path(cls, file_path):
        key = cls._key(file_path)
        if key is None:
            return None
        return os.path.join(cls._cache_dir(), f"{key}.pkl")

    @classmethod
    def load(cls, file_path):
        """Load cached data for a file. Returns None if not cached."""
        cache_path = cls._path(file_path)
        if cache_path is None or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            return None

    @classmethod
    def save(cls, file_path, data):
        """Save analysis data to cache."""
        cache_path = cls._path(file_path)
        if cache_path is None:
            return
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    @classmethod
    def update_classification(cls, file_path, classification):
        """Add classification result to an existing cache entry."""
        data = cls.load(file_path)
        if data is None:
            return
        data["classification"] = classification
        cls.save(file_path, data)

    @classmethod
    def clear_classification(cls, file_path):
        """Remove classification from cache (keeps features for fast reload)."""
        data = cls.load(file_path)
        if data is None:
            return
        data.pop("classification", None)
        cls.save(file_path, data)
