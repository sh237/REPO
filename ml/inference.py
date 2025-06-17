#!/usr/bin/env python
"""
inference.py

datasets/all_data_hours/ の画像データと datasets/all_features/ の特徴量データを組み合わせて推論を行います。

画像データ: 直近4時間分 (1, 4, 10, 256, 256)
特徴量データ: 直近672時間分 (1, 672, 128)

推論結果は ../data/pred_24.json に保存されます。

実行例:
$ python inference.py --params params/main/params.yaml --fold 1 --data_root ./datasets --cuda_device 0 --history 4 --trial_name 090 --mode test --resume_from_checkpoint checkpoints/main/216_stage1_best.pth
"""

import os
import sys
import json
import h5py
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace
import yaml
from models.main.models.ours import Ours

def simple_parse_params(dump=False):
    """
    Simplified parse_params function that only loads what we need for inference
    """
    class SimpleConfig:
        def __init__(self):
            self.cache_root = "ml/datasets/main/cache"  # Cache root path for main (not pretrain)
            self.data_root = "ml/datasets"  # Default data root
            self.fold = 1  # Default fold
            
            # モデル設定のモック
            self.model = SimpleNamespace()
            self.model.selected = "Ours"
            self.model.models = SimpleNamespace()
            self.model.models.Ours = SimpleNamespace()
            self.model.models.Ours.architecture_params = SimpleNamespace()
            # デフォルトのアーキテクチャパラメータ
            self.model.models.Ours.architecture_params.in_chans = 10
            self.model.models.Ours.architecture_params.embed_dim = 768
            self.model.models.Ours.architecture_params.depth = 12
            self.model.models.Ours.architecture_params.num_heads = 12
            self.model.models.Ours.architecture_params.mlp_ratio = 4.0
            self.model.models.Ours.architecture_params.drop_rate = 0.1
            self.model.models.Ours.architecture_params.attn_drop_rate = 0.1
    
    config = SimpleConfig()
    return config, {}

def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    # チェックポイントが "model_state_dict" を持っている場合
    if "model_state_dict" in state:
        state_dict = state["model_state_dict"]
    else:
        state_dict = state

    # "total_ops" や "total_params" を含むキーを除外
    filtered_state_dict = {k: v for k, v in state_dict.items() if "total_ops" not in k and "total_params" not in k}

    model.load_state_dict(filtered_state_dict)
    return model

def namespace_to_dict(obj):
    """再帰的に Namespace や list を辞書に変換する"""
    if isinstance(obj, (Namespace, SimpleNamespace)):
        return {k: namespace_to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: namespace_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [namespace_to_dict(v) for v in obj]
    else:
        return obj

def load_feature_data(feature_path):
    """特徴量H5ファイルから特徴量データを読み込む"""
    try:
        with h5py.File(feature_path, "r") as f:
            features = f["features"][:]  # shape: (128,)
            return features
    except Exception as e:
        print(f"Warning: 特徴量ファイル読み込みエラー {feature_path}: {e}")
        return None

def get_features_for_timerange(feature_dir, target_time, hours=672):
    """指定した時刻から過去hours時間分の特徴量を取得"""
    features_list = []
    
    for i in range(hours):
        # target_timeから i 時間前の時刻を計算
        current_time = target_time - timedelta(hours=i)
        feature_filename = current_time.strftime("%Y%m%d_%H0000.h5")
        feature_path = os.path.join(feature_dir, feature_filename)
        
        if os.path.exists(feature_path):
            features = load_feature_data(feature_path)
            if features is not None:
                features_list.append(features)
            else:
                # 読み込み失敗の場合は0埋め
                features_list.append(np.zeros(128, dtype=np.float32))
        else:
            # ファイルが存在しない場合は0埋め
            features_list.append(np.zeros(128, dtype=np.float32))
    
    # 時系列順に並べ直す（最古→最新）
    features_list.reverse()
    
    # スタックして (672, 128) にする
    features_array = np.stack(features_list, axis=0)
    return features_array

def main():
    parser = ArgumentParser()
    parser.add_argument("--params", required=True, help="YAML設定ファイルのパス")
    parser.add_argument("--fold", type=int, required=True, help="Fold番号")
    parser.add_argument("--data_root", default="./ml/datasets", help="datasetsディレクトリのルート")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--history", type=int, default=4, help="使用する履歴ファイル数（例: 4）")
    parser.add_argument("--trial_name", default="idxxxx")
    parser.add_argument("--mode", default="test")
    parser.add_argument("--resume_from_checkpoint", required=True, help="チェックポイントのパス")
    parser.add_argument("--debug", action="store_true", help="デバッグモード: 結果をファイルに保存せず、コマンドラインに出力します")
    parser.add_argument("--datetime", help="処理対象の日時 (YYYYMMDD_HHMMSS形式) - 特定の時刻のみ処理")
    args = parser.parse_args()

    # 既存の設定ファイルからパラメータを読み込む
    args_config, yaml_config = simple_parse_params()
    # コマンドラインの各パラメータで上書き
    args_config.fold = args.fold
    args_config.data_root = args.data_root
    args_config.cuda_device = args.cuda_device
    args_config.history = args.history
    args_config.trial_name = args.trial_name
    args_config.mode = args.mode
    args_config.resume_from_checkpoint = args.resume_from_checkpoint
    # 推論はCPU上で実施する
    args_config.device = "cpu"
    # h5ファイルの格納先は all_data_hours 以下
    args_config.data_path = os.path.join(args_config.data_root, "all_data_hours")
    # 特徴量ファイルの格納先
    feature_dir = os.path.join(args_config.data_root, "all_features")

    # キャッシュされた統計情報（means.npy, stds.npy）を読み込む
    fold_dir = os.path.join(args_config.cache_root, f"fold{args_config.fold}")
    train_cache_dir = os.path.join(fold_dir, "train")
    means_path = os.path.join(train_cache_dir, "means.npy")
    stds_path = os.path.join(train_cache_dir, "stds.npy")
    if not os.path.exists(means_path) or not os.path.exists(stds_path):
        print("Error: Cached statistics (means.npy or stds.npy) not found.")
        sys.exit(1)
    means = np.load(means_path)   # shape: (10,)
    stds = np.load(stds_path)       # shape: (10,)

    # 全ての h5 ファイル一覧を取得し、タイムスタンプでソート
    all_files = [f for f in os.listdir(args_config.data_path) if f.endswith(".h5")]
    file_dict = {}
    for f in all_files:
        # ファイル名例: "20250401_000000.h5"
        try:
            ts_str = os.path.splitext(f)[0]
            ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
            file_dict[ts] = os.path.join(args_config.data_path, f)
        except Exception as e:
            print(f"Warning: ファイル名のパースに失敗: {f} ({e})")
            continue

    if not file_dict:
        print("Error: 有効な h5 ファイルが見つかりませんでした。")
        sys.exit(1)

    sorted_times = sorted(file_dict.keys())

    # 推論可能なターゲット時刻を抽出
    if args.datetime:
        # 特定の日時が指定された場合
        try:
            target_time = datetime.strptime(args.datetime, "%Y%m%d_%H%M%S")
            # 指定された時刻に必要なファイルが存在するかチェック（1時間間隔）
            required = [target_time - timedelta(hours=i) for i in range(0, args_config.history)]
            # 欠損は0埋めで対応するので、ターゲット時刻のファイルが存在すれば処理可能
            if target_time in file_dict:
                valid_targets = [target_time]
            else:
                print(f"Error: 指定された時刻 {args.datetime} のファイルが見つかりません:")
                print(f"必要なファイル: {target_time.strftime('%Y%m%d_%H%M%S')}.h5")
                sys.exit(1)
        except ValueError:
            print(f"Error: 日時の形式が正しくありません: {args.datetime}")
            print("正しい形式: YYYYMMDD_HHMMSS (例: 20250615_180000)")
            sys.exit(1)
    else:
        # 全ての利用可能な時刻で推論
        # 各ターゲット時刻 T に対して、T, T-1h, T-2h, T-3h が存在すれば対象とする
        valid_targets = []
        for t in sorted_times:
            required = [t - timedelta(hours=i) for i in range(0, args_config.history)]
            if all(rt in file_dict for rt in required):
                valid_targets.append(t)

    if not valid_targets:
        print("Error: 推論可能なターゲット時刻が見つかりませんでした。")
        sys.exit(1)

    print("推論可能なターゲット時刻（最新時刻）一覧:")
    for vt in valid_targets:
        print(vt.strftime("%Y%m%d_%H%M%S"))

    # モデルのインスタンス化（config の設定から architecture_params を使用）
    # SimpleNamespaceから直接辞書を作成
    params_dict = {
        'D': 64,  # Feature dimension
        'drop_path_rate': 0.6,
        'layer_scale_init_value': 1.0e-6,
        'L': 128,  # Sequence length
        'L_SSE': 3,  # Number of SolarSpatialEncoder layers
        'L_LT': 1,  # Number of LongRangeTemporalSSM layers
        'L_mixing': 2,  # Number of MixingSSM layers
        'dropout_rates': {
            'sse': 0.6,
            'dwcm': 0.6,
            'stssm': 0.6,
            'ltssm': 0.6,
            'mixing_ssm': 0.6,
            'head': 0.6
        }
    }
    model = Ours(**params_dict)
    model.to("cpu")
    model.eval()
    # チェックポイントからモデルパラメータをロード
    load_checkpoint(model, args_config.resume_from_checkpoint)

    predictions = {}
    # 各ターゲット時刻について推論を実施
    for target in valid_targets:
        print(f"\n処理中: {target.strftime('%Y%m%d_%H%M%S')}")
        
        # === 画像データの準備 (直近4時間分) ===
        # required_times は [T-3h, T-2h, T-1h, T] の昇順のリスト
        required_times = [target - timedelta(hours=args_config.history - 1 - i) for i in range(args_config.history)]
        X_list = []
        for rt in required_times:
            file_path = file_dict.get(rt)
            if file_path is None:
                print(f"Warning: {rt.strftime('%Y%m%d_%H%M%S')} のファイルが存在しません。0埋めします。")
                # 欠損ファイルは0埋め（10, 256, 256）
                X_data = np.zeros((10, 256, 256), dtype=np.float32)
                X_list.append(X_data)
            else:
                with h5py.File(file_path, "r") as f:
                    X_data = f["X"][:]  # shape: (10,256,256) を想定
                    X_list.append(X_data)
        # 0埋めで対応するので、常にhistory分のデータが揃う
        if len(X_list) != args_config.history:
            print(f"Error: 予期しないデータ数: {len(X_list)} (期待値: {args_config.history})")
            continue
            
        # スタックして形状 (history, 10, 256,256) にする
        X_array = np.stack(X_list, axis=0)
        X_array = np.nan_to_num(X_array, 0)
        # 正規化：各チャネル毎に (x - mean) / (std + 1e-8)
        X_norm = (X_array - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-8)
        # テンソルに変換し、バッチ次元を追加 → (1, history, 10,256,256)
        sample_input1 = torch.from_numpy(X_norm).float().unsqueeze(0)
        
        # === 特徴量データの準備 (直近672時間分) ===
        features_array = get_features_for_timerange(feature_dir, target, hours=672)
        # テンソルに変換し、バッチ次元を追加 → (1, 672, 128)
        sample_input2 = torch.from_numpy(features_array).float().unsqueeze(0)
        
        print(f"画像データ形状: {sample_input1.shape}")
        print(f"特徴量データ形状: {sample_input2.shape}")
        
        with torch.no_grad():
            logits, _ = model(sample_input1, sample_input2)
            probs = torch.softmax(logits, dim=1)
            probs_list = probs[0].cpu().numpy().tolist()
        
        # キーはターゲット時刻（最新）の "YYYYMMDDHH" 形式
        key = target.strftime("%Y%m%d%H")
        predictions[key] = [round(p, 6) for p in probs_list]
        print(f"予測結果: {probs_list}")

    if args.debug:
        print("Prediction results (debug mode):")
        for ts, probs in predictions.items():
            print(f"{ts}: {probs}")
    else:
        # 結果を "../data/pred_24.json" に保存
        out_dir = os.path.join("..", "data")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "pred_24.json")
        with open(out_path, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"\n✅ Saved predictions for {len(predictions)} timestamps to {out_path}")
        print("Prediction results:")
        for ts, probs in predictions.items():
            print(f"{ts}: {probs}")

if __name__ == "__main__":
    main()