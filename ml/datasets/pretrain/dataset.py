import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import gc


def compute_and_save_stats(data_dir, cache_dir, force_recalc=False, batch_size=100):
    """
    Compute and save dataset statistics (memory-efficient version)
    Args:
        data_dir: Path to data directory
        cache_dir: Path to cache directory
        force_recalc: Whether to force recalculation
        batch_size: Number of files to process at once
    Returns:
        p1: 1st percentile values (channels,)
        p99: 99th percentile values (channels,)
    """
    stats_file = os.path.join(cache_dir, "normalization_stats.npz")

    if not force_recalc and os.path.exists(stats_file):
        print(f"Loading existing stats from {stats_file}")
        stats = np.load(stats_file)
        return stats["p1"], stats["p99"]

    print("Computing dataset statistics...")

    # Get all files list
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".h5")]
    total_files = len(all_files)

    # List to store values for each channel (memory-efficient with sampling)
    channel_values = {i: [] for i in range(10)}
    max_values_per_channel = 10000000  # Maximum number of samples per channel

    # Batch processing
    for i in tqdm(range(0, total_files, batch_size), desc="Processing files"):
        batch_files = all_files[i : min(i + batch_size, total_files)]

        for file in batch_files:
            try:
                with h5py.File(os.path.join(data_dir, file), "r") as f:
                    X = f["X"][:]
                    X = np.nan_to_num(X, 0)

                    for ch in range(10):
                        valid_pixels = X[ch][X[ch] != 0]

                        # Calculate sampling rate
                        if len(valid_pixels) > 0:
                            current_samples = len(channel_values[ch])
                            if current_samples < max_values_per_channel:
                                # Random sampling
                                sampling_rate = min(
                                    1.0,
                                    (max_values_per_channel - current_samples)
                                    / len(valid_pixels),
                                )
                                if sampling_rate < 1.0:
                                    mask = (
                                        np.random.random(len(valid_pixels))
                                        < sampling_rate
                                    )
                                    valid_pixels = valid_pixels[mask]
                                channel_values[ch].extend(valid_pixels)

            except Exception as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        # Free memory
        gc.collect()

    # Calculate percentiles
    p1 = np.zeros(10)
    p99 = np.zeros(10)

    for ch in range(10):
        values = np.array(channel_values[ch])
        if len(values) > 0:
            if ch < 9:  # AIA channels
                p1[ch] = np.percentile(values, 1)
                p99[ch] = np.percentile(values, 99)
            else:  # HMI channel
                p1[ch] = 0
                p99[ch] = 255

        # Free memory
        channel_values[ch] = []
        gc.collect()

    # Save statistics
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(stats_file, p1=p1, p99=p99)
    print(f"Saved stats to {stats_file}")

    return p1, p99


def normalize_solar_data(X, p1, p99):
    """
    Normalize data
    Args:
        X: Input data (channels, height, width)
        p1: 1st percentile values for each channel
        p99: 99th percentile values for each channel
    Returns:
        normalized: Normalized data
    """
    normalized = np.zeros_like(X, dtype=np.float32)

    # AIA channels (0-8): normalize to [-1, 1] range
    for ch in range(9):
        valid_mask = X[ch] != 0
        if valid_mask.any():
            normalized[ch][valid_mask] = np.clip(X[ch][valid_mask], p1[ch], p99[ch])
            normalized[ch][valid_mask] = (
                2 * (normalized[ch][valid_mask] - p1[ch]) / (p99[ch] - p1[ch]) - 1
            )

    # HMI channel (9): normalize to [0, 1] range
    valid_mask = X[9] != 0
    if valid_mask.any():
        normalized[9][valid_mask] = X[9][valid_mask] / 255.0

    return normalized


class SolarFlareDataset(Dataset):
    def __init__(
        self, data_dir, periods, split="train", cache_dirs=None, force_recalc=False
    ):
        """Initialization"""
        self.data_dir = data_dir
        self.periods = [
            (pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods
        ]
        self.split = split
        self.cache_dir = cache_dirs[split]

        self.indices_file = os.path.join(self.cache_dir, "data_indices.pkl")
        self.labels_file = os.path.join(self.cache_dir, "labels.npy")
        self.timestamps_file = os.path.join(self.cache_dir, "timestamps.pkl")
        self.valid_indices_file = os.path.join(self.cache_dir, "valid_indices.pkl")

        if force_recalc and os.path.exists(self.valid_indices_file):
            os.remove(self.valid_indices_file)

        if split == "train":
            self.p1, self.p99 = compute_and_save_stats(
                data_dir, self.cache_dir, force_recalc=force_recalc, batch_size=100
            )
        else:
            stats_file = os.path.join(cache_dirs["train"], "normalization_stats.npz")
            if not os.path.exists(stats_file):
                raise FileNotFoundError(
                    f"Training stats file not found at {stats_file}. "
                    "Please run training first to compute statistics."
                )
            stats = np.load(stats_file)
            self.p1, self.p99 = stats["p1"], stats["p99"]

        self.data_indices, self.labels, self.timestamps = self._process_data()
        self.valid_indices = self._get_valid_indices()
        print(f"{split.capitalize()} dataset: {len(self.valid_indices)} samples")

    def _process_data(self):
        """Process data (without statistics calculation)"""
        print(f"Processing data files for {self.split} period...")
        data_indices = []
        y_data = []
        timestamps = []

        filtered_files = [
            f
            for f in sorted(os.listdir(self.data_dir))
            if f.endswith(".h5")
            and self._is_in_periods(
                pd.to_datetime(f.split(".")[0], format="%Y%m%d_%H%M%S")
            )
        ]

        for file in tqdm(filtered_files, desc="Processing files"):
            file_path = os.path.join(self.data_dir, file)
            try:
                with h5py.File(file_path, "r") as f:
                    y = f["y"][()]
                    ts = f["timestamp"][()]

                    if isinstance(y, (bytes, np.bytes_)):
                        y_str = y.decode("utf-8")
                        label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                        y = label_map.get(y_str, 0)
                    elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                        label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                        y = label_map.get(y[0], 0)

                    if y == 0:
                        continue

                    data_indices.append(file_path)
                    if isinstance(ts, bytes):
                        ts_str = ts.decode("utf-8")
                        timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
                    else:
                        timestamp = pd.to_datetime(ts)
                    timestamps.append(timestamp)
                    y_data.append(y)

            except (OSError, KeyError) as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        return data_indices, y_data, timestamps

    def _is_in_periods(self, date):
        """Determine if the specified date is within the periods"""
        return any(start <= date <= end for start, end in self.periods)

    def _get_valid_indices(self):
        """Get valid indices"""
        print("self.valid_indices_file: ", self.valid_indices_file)
        if os.path.exists(self.valid_indices_file):
            print("Loading cached valid indices...")
            with open(self.valid_indices_file, "rb") as f:
                valid_indices = pickle.load(f)
                valid_indices = [
                    idx for idx in valid_indices if idx < len(self.data_indices)
                ]
                if len(valid_indices) == 0:
                    print("Warning: No valid indices found in cache, recalculating...")
                else:
                    return valid_indices

        print("Finding valid indices...")
        valid_indices = []
        for i in range(len(self.data_indices)):
            try:
                file_path = self.data_indices[i]
                with h5py.File(file_path, "r") as f:
                    y = f["y"][()]
                    if y != 0 and i < len(self.labels):
                        valid_indices.append(i)
            except Exception as e:
                print(f"Error processing index {i}: {str(e)}")
                continue

        if len(valid_indices) == 0:
            raise RuntimeError("No valid indices found in the dataset")

        with open(self.valid_indices_file, "wb") as f:
            pickle.dump(valid_indices, f)

        return valid_indices

    def __getitem__(self, idx):
        """Get data"""
        valid_idx = self.valid_indices[idx]
        file_path = self.data_indices[valid_idx]

        try:
            with h5py.File(file_path, "r") as f:
                X = f["X"][:]
                y = f["y"][()]
                ts = f["timestamp"][()]

            X = np.nan_to_num(X, 0)
            X = normalize_solar_data(X, self.p1, self.p99)
            X = torch.from_numpy(X).float()

            if isinstance(y, (bytes, np.bytes_)):
                y_str = y.decode("utf-8")
                label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                y = label_map.get(y_str, 0)
            elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                y = label_map.get(y[0], 0)

            y_onehot = torch.zeros(4, dtype=torch.float32)
            if y > 0:
                y_onehot[y - 1] = 1.0

            if isinstance(ts, bytes):
                ts_str = ts.decode("utf-8")
                timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
            else:
                timestamp = pd.to_datetime(ts, format="%Y%m%d_%H%M%S")

            return X, y_onehot, timestamp

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            raise

    def __len__(self):
        """Return the number of valid samples"""
        return len(self.valid_indices)


class AllDataDataset(Dataset):
    def __init__(self, data_dir, cache_dir):
        """
        Args:
            data_dir (str): Path to data directory
            cache_dir (str): Path to cache directory (containing training statistics)
        """
        self.data_dir = data_dir

        stats_file = os.path.join(cache_dir, "normalization_stats.npz")
        if not os.path.exists(stats_file):
            raise FileNotFoundError(
                f"Training stats file not found at {stats_file}. "
                "Please run training first to compute statistics."
            )
        stats = np.load(stats_file)
        self.p1, self.p99 = stats["p1"], stats["p99"]

        self.file_paths = sorted([f for f in os.listdir(data_dir) if f.endswith(".h5")])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_paths[idx])

        try:
            with h5py.File(file_path, "r") as f:
                X = f["X"][:]
                y = f["y"][()]
                ts = f["timestamp"][()]

            X = np.nan_to_num(X, 0)

            X = normalize_solar_data(X, self.p1, self.p99)
            X = torch.from_numpy(X).float()

            if isinstance(y, (bytes, np.bytes_)):
                y_str = y.decode("utf-8")
                label_map = {"O": 1, "C": 2, "M": 3, "X": 4}
                y = label_map.get(y_str, 0)
            elif isinstance(y, np.ndarray) and y.dtype.kind in ["S", "U"]:
                label_map = {b"O": 1, b"C": 2, b"M": 3, b"X": 4}
                y = label_map.get(y[0], 0)

            # one-hot encoding
            y_onehot = torch.zeros(4, dtype=torch.float32)
            if y > 0:
                y_onehot[y - 1] = 1.0

            # Process timestamp
            if isinstance(ts, bytes):
                ts_str = ts.decode("utf-8")
                timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
            else:
                timestamp = pd.to_datetime(ts)

            return X, y_onehot, timestamp

        except Exception as e:
            print(f"Error loading file {file_path}: {str(e)}")
            # If error occurs, return zero tensors
            X = torch.zeros((10, 256, 256), dtype=torch.float32)
            y_onehot = torch.zeros(4, dtype=torch.float32)
            timestamp = pd.to_datetime(
                self.file_paths[idx].split(".")[0], format="%Y%m%d_%H%M%S"
            )
            return X, y_onehot, timestamp
