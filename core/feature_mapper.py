"""
Maps audio features to linear 3D visual properties:
- Position X: time (grows linearly with playback)
- Position Y: spectral centroid (frequency = height)
- Position Z: spectral bandwidth (spread = depth)
- Size: RMS energy
- Color: dominant note from chromagram

Pre-computes harmonic connections between points of the same
musical note that are spatially close (cKDTree).

Linear mapping produces distinct visual signatures per source type:
- Human voice: vertical column with stable harmonic branches
- Orchestra: chaotic and wide explosion
- Birds: thin and rapid ascending traces
"""
import numpy as np
from scipy.spatial import cKDTree
from core.color_mapping import chroma_to_rgba_batch
import config


class FeatureMapper:
    """
    Transforms audio features into linear 3D positions
    and pre-computes harmonic connections between similar nodes.
    Each sound type produces a unique geometric signature.
    """

    def __init__(self):
        self.positions = None
        self.colors = None
        self.sizes = None
        self.n_frames = 0

        # Harmonic connections: index pairs (frame_a, frame_b)
        self.connections = None
        # Pre-computed arrays for line rendering
        self.connection_positions = None
        self.connection_colors = None

    def map_features(self, features):
        """
        Converts analysis results to linear 3D positions
        and pre-computes harmonic connections.
        """
        self.n_frames = features["n_frames"]
        times = features["times"]
        centroid = features["centroid"]
        bandwidth = features["bandwidth"]
        rms = features["rms"]
        chroma = features["chroma"]

        # --- Linear mapping (unique visual signature per sound type) ---

        # X axis: time grows linearly
        x = times * config.TIME_SCALE

        # Y axis: normalized spectral centroid (frequency = height)
        y = (centroid / config.SPECTRAL_CENTROID_MAX) * config.Y_SCENE_MAX
        y = np.clip(y, 0, config.Y_SCENE_MAX)

        # Z axis: spectral bandwidth (spread = depth)
        bw_max = bandwidth.max() if bandwidth.max() > 0 else 1.0
        bw_norm = bandwidth / bw_max
        z = bw_norm * config.Z_SCENE_RANGE - (config.Z_SCENE_RANGE / 2.0)

        self.positions = np.column_stack([x, y, z]).astype(np.float32)

        # --- Vivid colors from chromagram ---
        self.colors = chroma_to_rgba_batch(chroma)
        # Keep alpha high for all points
        self.colors[:, 3] = config.PERSISTENT_ALPHA

        # --- Sizes from RMS energy ---
        rms_max = rms.max() if rms.max() > 0 else 1.0
        rms_normalized = rms / rms_max
        self.sizes = (config.POINT_SIZE_MIN +
                      rms_normalized * config.POINT_SIZE_SCALE)
        self.sizes = self.sizes.astype(np.float32)

        # --- Pre-compute harmonic connections ---
        self._compute_harmonic_connections(chroma)

    def _compute_harmonic_connections(self, chroma):
        """
        Finds connections between points of the same dominant note
        that are spatially close using cKDTree.
        """
        dominant_notes = np.argmax(chroma, axis=0)  # (n_frames,)
        connections = []

        max_k = config.MAX_CONNECTIONS_PER_POINT
        max_dist = config.MAX_CONNECTION_DISTANCE

        for note in range(12):
            note_indices = np.where(dominant_notes == note)[0]
            if len(note_indices) < 2:
                continue

            note_positions = self.positions[note_indices]
            tree = cKDTree(note_positions)

            # K nearest neighbors for each point in the group
            k = min(max_k, len(note_indices) - 1)
            dists, neighbors = tree.query(note_positions, k=k + 1)

            for i in range(len(note_indices)):
                for j in range(1, k + 1):
                    if j < dists.shape[1] and dists[i, j] < max_dist:
                        frame_a = note_indices[i]
                        frame_b = note_indices[neighbors[i, j]]
                        # Avoid duplicates: only store if a < b
                        if frame_a < frame_b:
                            connections.append((frame_a, frame_b))

        if connections:
            self.connections = np.array(connections, dtype=np.int32)
        else:
            self.connections = np.zeros((0, 2), dtype=np.int32)

        # Pre-compute position and color arrays for lines
        self._build_connection_arrays()

    def _build_connection_arrays(self):
        """
        Builds position and color arrays for rendering
        connections as GL_LINES (vertex pairs).
        """
        n_conn = len(self.connections)
        if n_conn == 0:
            self.connection_positions = np.zeros((0, 3), dtype=np.float32)
            self.connection_colors = np.zeros((0, 4), dtype=np.float32)
            return

        # Each connection = 2 vertices (point A, point B)
        conn_pos = np.zeros((n_conn * 2, 3), dtype=np.float32)
        conn_colors = np.zeros((n_conn * 2, 4), dtype=np.float32)

        idx_a = self.connections[:, 0]
        idx_b = self.connections[:, 1]

        conn_pos[0::2] = self.positions[idx_a]
        conn_pos[1::2] = self.positions[idx_b]

        # Color: semi-transparent white for visibility on black background
        conn_colors[0::2, :3] = 1.0  # White RGB
        conn_colors[0::2, 3] = config.CONNECTION_LINE_ALPHA
        conn_colors[1::2, :3] = 1.0
        conn_colors[1::2, 3] = config.CONNECTION_LINE_ALPHA

        self.connection_positions = conn_pos
        self.connection_colors = conn_colors

    def get_range_data(self, start, end):
        """Gets visual properties for a range of frames."""
        s = max(0, start)
        e = min(end, self.n_frames)
        return {
            "positions": self.positions[s:e],
            "colors": self.colors[s:e],
            "sizes": self.sizes[s:e],
        }

    def get_visible_connections(self, max_frame):
        """
        Returns connections where both endpoints are revealed
        (both indices <= max_frame).
        """
        if len(self.connections) == 0:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32))

        # Filter: both endpoints must be <= max_frame
        mask = (self.connections[:, 0] <= max_frame) & \
               (self.connections[:, 1] <= max_frame)

        visible_idx = np.where(mask)[0]
        if len(visible_idx) == 0:
            return (np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0, 4), dtype=np.float32))

        # Indices into connection position/color arrays
        line_indices = np.empty(len(visible_idx) * 2, dtype=np.int64)
        line_indices[0::2] = visible_idx * 2
        line_indices[1::2] = visible_idx * 2 + 1

        return (self.connection_positions[line_indices],
                self.connection_colors[line_indices])

    def time_to_frame(self, seconds, times):
        """Converts time in seconds to the closest frame index."""
        idx = int(np.searchsorted(times, seconds))
        return min(idx, self.n_frames - 1)
