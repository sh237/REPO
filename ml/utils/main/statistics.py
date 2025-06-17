"""Scripts for evaluation metrics"""

import math
import torch
from sklearn import metrics
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from numpy import ndarray
import logging
import os
import pandas as pd
import h5py
from datetime import datetime


class Stat:
    """
    collect predictions and calculate some metrics
    """

    def __init__(self, climatology: np.ndarray, logger: logging.Logger):
        self.predictions = {"train": [], "valid": [], "test": []}
        self.observations = {"train": [], "valid": [], "test": []}
        self.climatology = climatology
        self.logger = logger
        self.gmgs_score_matrix = None

    @property
    def gmgs_score_matrix(self):
        return self._gmgs_score_matrix

    @gmgs_score_matrix.setter
    def gmgs_score_matrix(self, matrix):
        self._gmgs_score_matrix = matrix

    def collect(self, pred: torch.Tensor, ground_truth: torch.Tensor, mode: str):
        """
        Collect predictions and ground truth
        """
        assert mode in ["train", "valid", "test"], "Mode must be either 'train', 'valid' or 'test'"
        observation = torch.argmax(ground_truth, dim=1)
        self.predictions[mode].extend(pred.cpu().detach().numpy())
        self.observations[mode].extend(observation.cpu().detach().numpy())

    def aggregate(self, mode: str) -> Dict[str, Any]:
        """
        Aggregate collected data and calculate metrics
        """
        assert mode in ["train", "valid", "test"], "Mode must be either 'train', 'valid' or 'test'"
        score = {}
        y_pred, y_true = np.array(self.predictions[mode]), np.array(self.observations[mode])
        y_predl = [np.argmax(y) for y in y_pred]

        score["ACC"] = self.calc_accs(y_predl, y_true)
        score["TSS"] = self.calc_tss(y_predl, y_true, 2)
        score["BSS"] = self.calc_bss(y_pred, y_true, self.climatology)
        score["GMGS"] = self.calc_gmgs(y_predl, y_true)

        results = {f"{mode}_{k}": v for k, v in score.items()}

        return results

    def clear_all(self):
        """
        Clear all collected data
        """
        self.predictions = {"train": [], "valid": [], "test": []}
        self.observations = {"train": [], "valid": [], "test": []}

    def calc_mattheus(self, y_predl: ndarray, y_true: ndarray, flare_class: int):
        """
        Compute Matthews correlation coefficient
        """
        C = self.confusion_matrix(y_predl, y_true)
        c, s = np.diag(C).sum(), C.sum()
        p, t = np.sum(C, axis=0), np.sum(C, axis=1)
        mcc = (c * s - np.dot(p, t).sum()) / np.sqrt((s**2 - np.dot(p, p).sum()) - (s**2 - np.dot(t, t).sum()))
        return mcc

    def calc_tss(self, y_predl: ndarray, y_true: ndarray, flare_class: int) -> float:
        """
        Compute TSS
        """
        mtx = self.confusion_matrix(y_predl, y_true)
        tn, fp, fn, tp = self.binary_confusion_matrix(mtx, flare_class)
        print("tn, fp, fn, tp", tn, fp, fn, tp)
        # ゼロ除算を避けるためのチェックを追加
        if (tp + fn) == 0 or (fp + tn) == 0:
            return 0.0

        tss = (tp / (tp + fn)) - (fp / (fp + tn))
        return float(tss) if not math.isnan(tss) else 0.0

    def calc_gmgs(self, y_predl: ndarray, y_true: ndarray) -> float:
        """
        Compute GMGS (simplified version)
        """
        s = "XMCO"
        tss = np.empty(3, dtype=float)
        for i in range(3):
            tss[i] = self.calc_tss(y_predl, y_true, 3 - i)
            print(f"{s[i]}: {tss[i]}")

        return tss.mean()

    def calc_bss(self, y_pred: ndarray, y_true: ndarray, climatology: List[float]) -> float:
        """
        Compute BSS >= M
        """
        N = len(y_true)
        y_truel = np.array([self.convert_binary_onehot(y) for y in y_true])
        y_pred = np.vstack([y_pred[:, :2].sum(axis=1), y_pred[:, 2:].sum(axis=1)]).transpose()

        d = y_pred[:, 1] - y_truel[:, 1]
        bs = np.dot(d, d) / N
        bsc = climatology[0] * climatology[1]
        bss = (bsc - bs) / bsc

        return bss

    def convert_binary_onehot(self, flare_class: int) -> List[int]:
        """
        return 2-dimentional 1-of-K vector
        """
        return np.array([1, 0] if flare_class < 2 else [0, 1])

    def confusion_matrix(self, y_pred: ndarray, y_true: ndarray) -> ndarray:
        """
        return confusion matrix for 4 class
        """
        return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    def calc_accs(self, y_pred: ndarray, y_true: ndarray) -> float:
        """
        Compute classification accuracy for 4 class
        """
        cm = self.confusion_matrix(y_pred, y_true)
        accs = np.diag(cm).sum() / len(y_pred)
        print(cm)
        if self.logger is not None:
            self.logger.info(f"Confusion matrix: {cm}")

        return accs

    def binary_confusion_matrix(self, mtx: ndarray, target_class: int) -> Tuple[int, int, int, int]:
        """
        convert confusion matrix for 2 class ("< target_class" or ">= target_class")
        """
        tn, fp = mtx[:target_class, :target_class].sum(), mtx[:target_class, target_class:].sum()
        fn, tp = mtx[target_class:, :target_class].sum(), mtx[target_class:, target_class:].sum()

        return tn, fp, fn, tp


def calculate_climatology(labels):
    """
    Climatologyをクラスラベルから計算する関数。
    Climatologyは各クラスの出現頻度の長期平均です。
    """
    unique, counts = np.unique(labels, return_counts=True)
    climatology = counts / len(labels)
    return climatology


def calculate_gmgs_score_matrix(climatology):
    """
    ClimatologyからGMGS Score Matrixを計算する関数。
    """
    I = len(climatology)
    score_matrix = np.zeros((I, I))

    a = np.zeros(I)
    for i in range(I):
        a[i] = (1 - np.sum(climatology[: i + 1])) / np.sum(climatology[: i + 1])

    for i in range(I):
        for j in range(i, I):
            if i == j:
                score_matrix[i, j] = 1 / (I - 1) * (np.sum(a[:i]) + np.sum(a[i:]))
            else:
                score_matrix[i, j] = (
                    1
                    / (I - 1)
                    * (np.sum(a[:i]) + np.sum(-1 for k in range(i, j)) + np.sum(a[j:]))
                )

    # スコアマトリックスは対称行列なので、上三角を計算してから転置する
    for i in range(I):
        for j in range(i + 1, I):
            score_matrix[j, i] = score_matrix[i, j]

    return score_matrix

def read_labels_from_h5(file_path):
    """
    H5ファイルからラベルを読み込む関数。
    スカラーデータの場合も適切に処理します。
    """
    with h5py.File(file_path, "r") as f:
        try:
            # データセットが存在するか確認
            if "y" not in f:
                return None

            # データセットの取得
            dataset = f["y"]

            # スカラーデータの場合
            if dataset.shape == ():
                return np.array([dataset[()]])  # スカラー値を1要素の配列として返す

            # 通常の配列データの場合
            return dataset[:]

        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return None


def find_h5_files(directory, start_date, end_date):
    """
    ディレクトリ内の指定された期間に一致するすべてのH5ファイルを見つける関数。
    """
    files = [
        os.path.join(directory, fname)
        for fname in os.listdir(directory)
        if fname.endswith(".h5")
    ]
    filtered_files = []
    for file in files:
        date_str = os.path.basename(file).split(".")[0]  # ファイル名から日付部分を抽出
        try:
            date = pd.to_datetime(date_str, format="%Y%m%d_%H%M%S")
            if start_date <= date <= end_date:
                filtered_files.append(file)
        except ValueError:
            continue
    return filtered_files

def compute_statistics(
    data_dir: str,
    stats_dir: str,
    train_periods: List[Tuple[str, str]],
    force_recalc: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, Stat]:
    """統計量を計算する関数"""
    if logger:
        logger.info("=" * 50)
        logger.info("Statistics Computation Start")
        logger.info(f"Using statistics directory: {stats_dir}")
        logger.info("Training periods:")
        for start, end in train_periods:
            logger.info(f"  {start} to {end}")
        logger.info("=" * 50)

    os.makedirs(stats_dir, exist_ok=True)
    climatology_file = os.path.join(stats_dir, "climatology.npy")
    gmgs_matrix_file = os.path.join(stats_dir, "gmgs_score_matrix.npy")

    if not force_recalc and os.path.exists(climatology_file):
        if logger:
            logger.info(f"Loading cached statistics from {stats_dir}")
        full_climatology = np.load(climatology_file)
        gmgs_score_matrix = np.load(gmgs_matrix_file)
        if logger:
            logger.info(f"Loaded full Climatology: {full_climatology}")
    else:
        all_labels = []
        for start_date, end_date in train_periods:
            train_h5_files = find_h5_files(
                data_dir,
                pd.to_datetime(start_date),
                pd.to_datetime(end_date)
            )
            if logger:
                logger.info(
                    f"Found {len(train_h5_files)} training files for period {start_date} to {end_date}"
                )
            
            for file in train_h5_files:
                labels = read_labels_from_h5(file)
                if labels is not None:
                    all_labels.extend(labels)

        all_labels = np.array(all_labels)
        if logger:
            logger.info(f"Total labels collected: {len(all_labels)}")
            logger.info(f"Label distribution: {np.unique(all_labels, return_counts=True)}")

        filtered_labels = all_labels[(all_labels >= 1) & (all_labels <= 4)]
        if logger:
            logger.info(f"Filtered labels distribution: {np.unique(filtered_labels, return_counts=True)}")

        full_climatology = calculate_climatology(filtered_labels)
        if logger:
            logger.info(f"Computed full Climatology: {full_climatology}")

        np.save(climatology_file, full_climatology)
        if logger:
            logger.info(f"Climatology saved to {climatology_file}")

        gmgs_score_matrix = calculate_gmgs_score_matrix(full_climatology[::-1])
        if logger:
            logger.info("Computed GMGS Score Matrix:")
            for i in range(len(gmgs_score_matrix)):
                logger.info(f"Row {i}: {gmgs_score_matrix[i]}")

        np.save(gmgs_matrix_file, gmgs_score_matrix)
        if logger:
            logger.info(f"GMGS Score Matrix saved to {gmgs_matrix_file}")

    climatology = full_climatology[[2, 1]]
    climatology = climatology / climatology.sum()
    
    if logger:
        logger.info(f"Normalized M-X Climatology: {climatology}")
        logger.info(f"M class probability: {climatology[0]:.4f}")
        logger.info(f"X class probability: {climatology[1]:.4f}")
        logger.info("=" * 50)
        logger.info("Statistics Computation Complete")
        logger.info("=" * 50)

    stat = Stat(climatology, logger)
    stat.gmgs_score_matrix = gmgs_score_matrix

    return full_climatology, gmgs_score_matrix, stat