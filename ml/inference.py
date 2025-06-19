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
            # NaN値や無限大値を0に置換
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            return features
    except Exception as e:
        print(f"Warning: 特徴量ファイル読み込みエラー {feature_path}: {e}")
        return None

def get_features_for_timerange(feature_dir, target_time, hours=672):
    """指定した時刻から過去hours時間分の特徴量を取得"""
    features_list = []
    valid_features_count = 0
    
    for i in range(hours):
        # target_timeから i 時間前の時刻を計算
        current_time = target_time - timedelta(hours=i)
        feature_filename = current_time.strftime("%Y%m%d_%H0000.h5")
        feature_path = os.path.join(feature_dir, feature_filename)
        
        if os.path.exists(feature_path):
            features = load_feature_data(feature_path)
            if features is not None:
                # NaNや無限大を0に置換
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                features_list.append(features)
                valid_features_count += 1
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
    
    # 最終的にも一度NaN処理を実行
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_array, valid_features_count

def main():
    parser = ArgumentParser()
    parser.add_argument("--params", required=True, help="YAML設定ファイルのパス")
    parser.add_argument("--fold", type=int, required=True, help="Fold番号")
    parser.add_argument("--data_root", default="./datasets", help="datasetsディレクトリのルート")
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
        valid_images_count = 0
        
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
                    valid_images_count += 1
        
        # === 特徴量データの準備 (直近672時間分) ===
        features_array, valid_features_count = get_features_for_timerange(feature_dir, target, hours=672)
        
        # データ統計を表示
        image_missing_rate = (args_config.history - valid_images_count) / args_config.history * 100
        feature_missing_rate = (672 - valid_features_count) / 672 * 100
        
        print(f"📊 データ読み込み統計:")
        print(f"  画像データ: {valid_images_count}/{args_config.history} 個 (欠損率: {image_missing_rate:.1f}%)")
        print(f"  特徴量データ: {valid_features_count}/672 個 (欠損率: {feature_missing_rate:.1f}%)")
        
        # 0埋めで対応するので、常にhistory分のデータが揃う
        if len(X_list) != args_config.history:
            print(f"Error: 予期しないデータ数: {len(X_list)} (期待値: {args_config.history})")
            continue
            
        # スタックして形状 (history, 10, 256,256) にする
        X_array = np.stack(X_list, axis=0)
        
        # === デバッグ情報: 生データの確認 ===
        print(f"🔍 デバッグ情報:")
        print(f"  生画像データ統計: min={np.min(X_array):.4f}, max={np.max(X_array):.4f}, mean={np.mean(X_array):.4f}")
        print(f"  生画像データNaN数: {np.sum(np.isnan(X_array))}")
        print(f"  生画像データinf数: {np.sum(np.isinf(X_array))}")
        
        X_array = np.nan_to_num(X_array, 0)
        
        # === デバッグ情報: 正規化統計の確認 ===
        print(f"  正規化統計 means: min={np.min(means):.4f}, max={np.max(means):.4f}")
        print(f"  正規化統計 stds: min={np.min(stds):.4f}, max={np.max(stds):.4f}")
        print(f"  正規化統計にNaN: means={np.sum(np.isnan(means))}, stds={np.sum(np.isnan(stds))}")
        
        # 正規化：各チャネル毎に (x - mean) / (std + 1e-8)
        X_norm = (X_array - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-8)
        
        # === デバッグ情報: 正規化後データの確認 ===
        print(f"  正規化後画像データ統計: min={np.min(X_norm):.4f}, max={np.max(X_norm):.4f}, mean={np.mean(X_norm):.4f}")
        print(f"  正規化後画像データNaN数: {np.sum(np.isnan(X_norm))}")
        print(f"  正規化後画像データinf数: {np.sum(np.isinf(X_norm))}")
        
        # テンソルに変換し、バッチ次元を追加 → (1, history, 10,256,256)
        sample_input1 = torch.from_numpy(X_norm).float().unsqueeze(0)
        
        # === デバッグ情報: 特徴量データの確認 ===
        print(f"  特徴量データ統計: min={np.min(features_array):.4f}, max={np.max(features_array):.4f}, mean={np.mean(features_array):.4f}")
        print(f"  特徴量データNaN数: {np.sum(np.isnan(features_array))}")
        print(f"  特徴量データinf数: {np.sum(np.isinf(features_array))}")
        
        # テンソルに変換し、バッチ次元を追加 → (1, 672, 128)
        sample_input2 = torch.from_numpy(features_array).float().unsqueeze(0)
        
        print(f"画像データ形状: {sample_input1.shape}")
        print(f"特徴量データ形状: {sample_input2.shape}")
        
        # === デバッグ情報: モデル推論前の入力確認 ===
        print(f"  入力テンソル1 NaN数: {torch.sum(torch.isnan(sample_input1)).item()}")
        print(f"  入力テンソル1 inf数: {torch.sum(torch.isinf(sample_input1)).item()}")
        print(f"  入力テンソル2 NaN数: {torch.sum(torch.isnan(sample_input2)).item()}")
        print(f"  入力テンソル2 inf数: {torch.sum(torch.isinf(sample_input2)).item()}")
        
        with torch.no_grad():
            # === デバッグ情報: モデルの重みの確認 ===
            has_nan_weights = False
            for name, param in model.named_parameters():
                if torch.sum(torch.isnan(param)).item() > 0:
                    print(f"  ⚠️ モデル重み '{name}' にNaNが含まれています")
                    has_nan_weights = True
            if not has_nan_weights:
                print(f"  ✅ モデル重みにNaNは含まれていません")
            
            logits, _ = model(sample_input1, sample_input2)
            
            # === デバッグ情報: logitsの確認 ===
            print(f"  logits統計: min={torch.min(logits).item():.4f}, max={torch.max(logits).item():.4f}, mean={torch.mean(logits).item():.4f}")
            print(f"  logits NaN数: {torch.sum(torch.isnan(logits)).item()}")
            print(f"  logits inf数: {torch.sum(torch.isinf(logits)).item()}")
            print(f"  logits値: {logits[0].cpu().numpy()}")
            
            probs = torch.softmax(logits, dim=1)
            
            # === デバッグ情報: 確率の確認 ===
            print(f"  probs統計: min={torch.min(probs).item():.4f}, max={torch.max(probs).item():.4f}, sum={torch.sum(probs).item():.4f}")
            print(f"  probs NaN数: {torch.sum(torch.isnan(probs)).item()}")
            print(f"  probs inf数: {torch.sum(torch.isinf(probs)).item()}")
            
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