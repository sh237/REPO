"""Dataset for Flare Transformer"""

import torch
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import h5py
import pandas as pd
import logging
import pickle
import random
from torchvision.transforms.functional import rotate
from multiprocessing import Pool, cpu_count
import gc

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SolarFlareDatasetWithFeatures(Dataset):
    def __init__(
        self, data_dir, periods, history=4, split="train", args=None, transform=None
    ):
        """
        Args:
            data_dir (str): Path to data directory
            periods (list): List of periods in the format [(start_date, end_date), ...]
            history (int): Length of history
            split (str): Dataset type ("train", "valid", "test")
            args: Additional arguments
            transform: Transform for data augmentation
        """
        self.data_dir = data_dir
        self.periods = [
            (pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods
        ]
        self.history = history
        self.split = split
        self.args = args
        self.transform = transform
        self.return_timestamp = False

        # Cache directory setup
        fold_num = getattr(args, "fold", 3)
        fold_dir = os.path.join(args.cache_root, f"fold{fold_num}")
        self.cache_dir = os.path.join(fold_dir, self.split)
        self.train_cache_dir = os.path.join(fold_dir, "train")

        # Features directory setup
        self.features_dir = args.features_path

        # Create directories
        for d in [self.cache_dir, self.train_cache_dir]:
            os.makedirs(d, exist_ok=True)

        # Cache file paths
        self.indices_file = os.path.join(self.cache_dir, "data_indices.pkl")
        self.means_file = os.path.join(self.train_cache_dir, "means.npy")
        self.stds_file = os.path.join(self.train_cache_dir, "stds.npy")
        self.labels_file = os.path.join(self.cache_dir, "labels.npy")
        self.timestamps_file = os.path.join(self.cache_dir, "timestamps.pkl")
        self.valid_indices_file = os.path.join(
            self.cache_dir, f"{self.history}_valid_indices.pkl"
        )

        if split == "train":
            self.data_indices, self.means, self.stds, self.labels, self.timestamps = (
                self._process_data()
            )
        else:
            if not os.path.exists(self.means_file) or not os.path.exists(
                self.stds_file
            ):
                raise FileNotFoundError(
                    f"Training statistics not found at {self.means_file} or {self.stds_file}. "
                    "Please process training data first."
                )
            self.means = np.load(self.means_file)
            self.stds = np.load(self.stds_file)
            self.data_indices, self.labels, self.timestamps = (
                self._process_data_without_stats()
            )

        self.valid_indices = self._get_valid_indices()

        print(f"Fold {fold_num} - {self.split}:")
        print("total samples: ", len(self.data_indices))
        print("valid samples: ", len(self.valid_indices))

        self.transform = transform

        # Changed channel count for statistics calculation from 12 to 10
        count = np.zeros(10)  # Changed from 12 to 10
        mean = np.zeros(10)  # Changed from 12 to 10
        M2 = np.zeros(10)  # Changed from 12 to 10

    def _is_in_periods(self, date):
        """Determine if the specified date is within the periods"""
        return any(start <= date <= end for start, end in self.periods)

    def _process_data(self):
        """Process and cache data"""
        if not self.args.dataset["force_recalc_stats"] and all(
            os.path.exists(f)
            for f in [
                self.indices_file,
                self.means_file,
                self.stds_file,
                self.labels_file,
                self.timestamps_file,
            ]
        ):
            print(f"Loading cached data for {self.split} period...")
            with open(self.indices_file, "rb") as f:
                data_indices = pickle.load(f)
            means = np.load(self.means_file)
            stds = np.load(self.stds_file)
            labels = np.load(self.labels_file)
            with open(self.timestamps_file, "rb") as f:
                timestamps = pickle.load(f)
            return data_indices, means, stds, labels.tolist(), timestamps

        print(f"Processing data files for {self.split} period...")
        data_indices = []
        y_data = []
        count = np.zeros(10)
        mean = np.zeros(10)
        M2 = np.zeros(10)
        timestamps = []

        filtered_files = [
            f
            for f in sorted(os.listdir(self.data_dir))
            if f.endswith(".h5")
            and self._is_in_periods(
                pd.to_datetime(f.split(".")[0], format="%Y%m%d_%H%M%S")
            )
        ]

        print(
            f"Found {len(filtered_files)} files for period {self.split} ({self.periods[0][0]} to {self.periods[-1][1]})"
        )

        for file in tqdm(filtered_files, desc=f"Processing {self.split} data"):
            file_path = os.path.join(self.data_dir, file)
            try:
                with h5py.File(file_path, "r") as f:
                    X = f["X"][:]
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

                    X = np.nan_to_num(X, 0)

                    valid_img = False
                    for j in range(10):
                        channel_data = X[j]
                        if np.any(channel_data):
                            valid_img = True
                            new_count = count[j] + 1
                            new_mean = (
                                mean[j] + (channel_data.mean() - mean[j]) / new_count
                            )
                            new_M2 = M2[j] + (channel_data.mean() - mean[j]) * (
                                channel_data.mean() - new_mean
                            )

                            if not np.isnan(new_mean) and not np.isnan(new_M2):
                                count[j] = new_count
                                mean[j] = new_mean
                                M2[j] = new_M2

                    if valid_img:
                        data_indices.append(file_path)
                        ts_str = ts.decode("utf-8")
                        try:
                            timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
                            timestamps.append(timestamp)
                            y_data.append(y)
                        except ValueError as e:
                            print(f"Error parsing timestamp {ts_str}: {str(e)}")
                            continue

            except (OSError, KeyError) as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        stds = np.where(count > 0, np.sqrt(M2 / count), 0)
        stds[stds == 0] = 1

        os.makedirs(os.path.dirname(self.indices_file), exist_ok=True)
        with open(self.indices_file, "wb") as f:
            pickle.dump(data_indices, f)
        np.save(self.means_file, mean)
        np.save(self.stds_file, stds)
        np.save(self.labels_file, y_data)
        with open(self.timestamps_file, "wb") as f:
            pickle.dump(timestamps, f)

        print(f"Processed {len(data_indices)} valid files for {self.split} period")
        return data_indices, mean, stds, y_data, timestamps

    def _process_data_without_stats(self):
        """Process data without calculating statistics (for validation/test data)"""
        if not self.args.dataset["force_recalc_stats"] and all(
            os.path.exists(f)
            for f in [self.indices_file, self.labels_file, self.timestamps_file]
        ):
            print(f"Loading cached data for {self.split} period...")
            with open(self.indices_file, "rb") as f:
                data_indices = pickle.load(f)
            labels = np.load(self.labels_file)
            with open(self.timestamps_file, "rb") as f:
                timestamps = pickle.load(f)
            return data_indices, labels.tolist(), timestamps

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

        print(
            f"Found {len(filtered_files)} files for period {self.split} ({self.periods[0][0]} to {self.periods[-1][1]})"
        )

        for file in tqdm(filtered_files, desc=f"Processing {self.split} data"):
            file_path = os.path.join(self.data_dir, file)
            try:
                with h5py.File(file_path, "r") as f:
                    X = f["X"][:]
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

                    X = np.nan_to_num(X, 0)

                    if np.any(X):
                        data_indices.append(file_path)
                        ts_str = ts.decode("utf-8")
                        try:
                            timestamp = pd.to_datetime(ts_str, format="%Y%m%d_%H%M%S")
                            timestamps.append(timestamp)
                            y_data.append(y)
                        except ValueError as e:
                            print(f"Error parsing timestamp {ts_str}: {str(e)}")
                            continue

            except (OSError, KeyError) as e:
                print(f"Error processing file {file}: {str(e)}")
                continue

        os.makedirs(os.path.dirname(self.indices_file), exist_ok=True)
        with open(self.indices_file, "wb") as f:
            pickle.dump(data_indices, f)
        np.save(self.labels_file, y_data)
        with open(self.timestamps_file, "wb") as f:
            pickle.dump(timestamps, f)

        print(f"Processed {len(data_indices)} valid files for {self.split} period")
        return data_indices, y_data, timestamps

    def _check_valid_index(self, i):
        """Internal method to check if an index is valid"""
        current_time = self.timestamps[i]
        total_missing_images = 0
        total_images = self.history * 10

        for h in range(self.history):
            if i - h < 0:
                return None

            history_time = self.timestamps[i - h]
            expected_time = current_time - pd.Timedelta(hours=h)

            if (
                history_time.hour != expected_time.hour
                or history_time.date() != expected_time.date()
            ):
                return None

            try:
                with h5py.File(self.data_indices[i - h], "r") as f:
                    x = f["X"][:]

                    for c in range(10):
                        channel_data = x[c]
                        if (
                            np.std(channel_data) < 1e-6
                            or np.all(channel_data == 0)
                            or np.all(np.isnan(channel_data))
                        ):
                            total_missing_images += 1

            except Exception as e:
                total_missing_images += 10

            if total_missing_images >= 10:
                return None

        return i if self.labels[i] > 0 else None

    def _get_valid_indices(self):
        """Get valid indices (serial processing version)"""
        cache_file = self.valid_indices_file

        if not self.args.dataset["force_recalc_indices"] and os.path.exists(cache_file):
            print(f"Loading cached valid indices for {self.split}...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        print(f"Calculating valid indices for {self.split}...")
        valid_indices = []

        for i in tqdm(range(len(self.data_indices)), desc="Checking valid indices"):
            result = self._check_valid_index(i)
            if result is not None:
                valid_indices.append(result)

        print(f"Found {len(valid_indices)} valid indices for {self.split}")

        # キャッシュを保存
        with open(cache_file, "wb") as f:
            pickle.dump(valid_indices, f)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        latest_idx = self.valid_indices[idx]
        history_indices = list(range(latest_idx - self.history + 1, latest_idx + 1))
        X = []

        # Get historical data in order
        for i in history_indices:
            try:
                with h5py.File(self.data_indices[i], "r") as f:
                    x = f["X"][:]
                    x = np.nan_to_num(x, 0)
                    x_normalized = (x - self.means[:, None, None]) / (
                        self.stds[:, None, None] + 1e-8
                    )
                    X.append(torch.from_numpy(x_normalized).float())
            except Exception as e:
                x = np.zeros((10, 256, 256), dtype=np.float32)
                X.append(torch.from_numpy(x).float())

        assert (
            len(X) == self.history
        ), f"Expected history length {self.history}, got {len(X)}"

        # Stack list to convert to Tensor
        X = torch.stack(X, dim=0)  # [history, channels, height, width]

        # Apply data augmentation (only for train)
        if self.transform is not None and self.split == "train":
            X = self.transform(X.to(self.args.device)).cpu()

        # Load feature data
        latest_timestamp = self.timestamps[latest_idx]
        feature_file = os.path.join(
            self.features_dir, f"{latest_timestamp.strftime('%Y%m%d_%H%M%S')}.h5"
        )

        try:
            with h5py.File(feature_file, "r") as f:
                h = f["features"][:]
                h = torch.from_numpy(h.astype(np.float32))
        except:
            h = torch.zeros((672, 128), dtype=torch.float32)

        # Check and process NaN and Inf
        if torch.isnan(h).any() or torch.isinf(h).any():
            h = torch.zeros_like(h)

        y = self.labels[latest_idx]
        y_onehot = torch.zeros(4, dtype=torch.float32)
        y_onehot[y - 1] = 1.0

        # Add check for NaN and Inf
        if torch.isnan(X).any() or torch.isinf(X).any():
            X = torch.nan_to_num(X, nan=0.0, posinf=1e5, neginf=-1e5)

        if self.return_timestamp:
            return X, h, y_onehot, latest_timestamp
        else:
            return X, h, y_onehot
