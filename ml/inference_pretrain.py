#!/usr/bin/env python
"""
inference_pretrain.py

指定された日付・日時の h5 ファイル群から、事前学習済みモデルを用いて中間特徴量を抽出し、保存します。
各ターゲット時刻 T の h5 ファイル (例: YYYYMMDD_HHMMSS.h5) を入力とし、
その特徴量を ml/datasets/all_features/YYYYMMDD_HHMMSS.h5 に保存します。

実行例:
$ python inference_pretrain.py --params params/main/params.yaml --datetime 20250504_080000 --fold 1 --pretrain_checkpoint checkpoints/pretrain/ours.pth
$ python inference_pretrain.py --params params/main/params.yaml --date 20250504 --fold 1 --pretrain_checkpoint checkpoints/pretrain/ours.pth
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
import yaml
# For pretrain models, import from the pretrain model definitions
from models.pretrain.mae import vit_for_FT128db, mae_vit_base_patch16_dec512d8b # Add other models if needed

def simple_parse_params(params_path, dump=False):
    """
    Simplified parse_params function that only loads what we need for inference_pretrain
    """
    class SimpleConfig:
        def __init__(self):
            self.cache_root = "datasets/pretrain/cache"  # Correct cache root path
            self.data_root = "./datasets"  # Default data root
            self.fold = 1  # Default fold
    
    config = SimpleConfig()
    
    # Try to load YAML file if it exists
    if params_path and os.path.exists(params_path):
        try:
            with open(params_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Update config with values from YAML if they exist
            if 'cache_root' in yaml_config:
                config.cache_root = yaml_config['cache_root']
            if 'data_root' in yaml_config:
                config.data_root = yaml_config['data_root']
            
            return config, yaml_config
        except:
            print(f"⚠️ Warning: Could not load YAML config from {params_path}, using defaults")
    
    return config, {}

def load_pretrain_checkpoint(model, checkpoint_path, map_location="cpu"):
    """事前学習済みモデルのチェックポイントをロードする"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Pretrain checkpoint file not found at {checkpoint_path}")
        sys.exit(1)
    state = torch.load(checkpoint_path, map_location=map_location)
    
    model_state_dict = None
    if "model" in state:
        model_state_dict = state["model"]
    elif "state_dict" in state:
        model_state_dict = state["state_dict"]
    elif "model_state_dict" in state: # From main inference.py style
         model_state_dict = state["model_state_dict"]
    else:
        model_state_dict = state # Assume the checkpoint is the state_dict itself

    if model_state_dict is None:
        print(f"❌ Error: Could not find model state_dict in checkpoint {checkpoint_path}")
        sys.exit(1)

    # Handle DataParallel/DistributedDataParallel prefix
    new_state_dict = {}
    for k, v in model_state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove `module.`
        else:
            new_state_dict[k] = v
    
    try:
        model.load_state_dict(new_state_dict)
        print(f"✅ Pretrain model loaded from {checkpoint_path}")
    except RuntimeError as e:
        print(f"❌ Error loading state_dict for pretrain model: {e}")
        print("This might be due to a mismatch in model architecture and checkpoint.")
        sys.exit(1)
    return model

def get_pretrain_model(model_arch, device):
    """指定されたアーキテクチャの事前学習モデルを初期化する"""
    # Define default parameters for pretrain models here
    # These should match the architecture used for 'ours.pth'
    # For vit_for_FT128db (example):
    model_params = {
        "in_chans": 10,
        "pyramid": True, # Example: ensure these match 'ours.pth'
        "sunspot": True, # Example: ensure these match 'ours.pth'
        # mask_ratio related params are not needed for feature extraction via forward_features
    }
    if model_arch == "vit_for_FT128db":
        model = vit_for_FT128db(**model_params)
    elif model_arch == "mae_vit_base_patch16_dec512d8b":
        # MAE model might need different default params
        model = mae_vit_base_patch16_dec512d8b(in_chans=10) # Simpler example
    else:
        print(f"❌ Error: Unknown pretrain model architecture: {model_arch}")
        sys.exit(1)
    return model.to(device)


def main():
    parser = ArgumentParser()
    parser.add_argument("--params", required=True, help="YAML設定ファイルのパス (主にキャッシュパス等で使用)")
    parser.add_argument("--date", help="処理対象の日付 (YYYYMMDD形式) - その日のすべてのファイルを処理")
    parser.add_argument("--datetime", help="処理対象の日時 (YYYYMMDD_HHMMSS形式) - 特定の時刻のファイルのみ処理")
    parser.add_argument("--fold", type=int, required=True, help="Fold番号 (means/stdsの読み込みに必要)")
    parser.add_argument("--data_root", default="./datasets", help="datasetsディレクトリのルート")
    parser.add_argument("--pretrain_checkpoint", default="checkpoints/pretrain/ours.pth", help="事前学習済みモデルのチェックポイントパス")
    parser.add_argument("--model_arch", default="vit_for_FT128db", help="事前学習モデルのアーキテクチャ (例: vit_for_FT128db, mae_vit_base_patch16_dec512d8b)")
    parser.add_argument("--output_feature_dir", default="datasets/all_features", help="抽出した特徴量の保存先ディレクトリ")
    parser.add_argument("--cuda_device", type=int, default=0, help="使用するCUDAデバイス番号 (CPUの場合は-1)")
    parser.add_argument("--debug", action="store_true", help="デバッグモード: 少数のファイルのみ処理し、詳細ログを出力")
    args = parser.parse_args()

    # --date と --datetime の排他チェック
    if not args.date and not args.datetime:
        print("❌ Error: Either --date (YYYYMMDD) or --datetime (YYYYMMDD_HHMMSS) must be specified.")
        sys.exit(1)
    if args.date and args.datetime:
        print("❌ Error: Cannot specify both --date and --datetime. Use one or the other.")
        sys.exit(1)

    # parse_paramsは主にargs_config.cache_root等の設定のために呼び出す
    args_config, _ = simple_parse_params(args.params, dump=False) # yaml_configは不要
    
    # コマンドライン引数でargs_configの関連値を上書き
    args_config.data_root = args.data_root
    args_config.fold = args.fold
    
    if args.cuda_device >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda_device}")
    else:
        device = torch.device("cpu")
    print(f"ℹ️ Using device: {device}")

    # h5ファイルの格納先 (入力元)
    input_h5_dir = os.path.join(args_config.data_root, "all_data_hours")
    if not os.path.isdir(input_h5_dir):
        print(f"❌ Error: Input H5 directory not found: {input_h5_dir}")
        sys.exit(1)

    # 特徴量の保存先ディレクトリを作成
    os.makedirs(args.output_feature_dir, exist_ok=True)
    print(f"ℹ️ Features will be saved to: {args.output_feature_dir}")

    # キャッシュされた統計情報（means.npy, stds.npy）を読み込む
    # `fold` は args_config ではなく args から直接使用
    fold_dir = os.path.join(args_config.cache_root, f"fold{args.fold}")
    train_cache_dir = os.path.join(fold_dir, "train")
    means_path = os.path.join(train_cache_dir, "means.npy")
    stds_path = os.path.join(train_cache_dir, "stds.npy")

    if not os.path.exists(means_path) or not os.path.exists(stds_path):
        print(f"❌ Error: Cached statistics (means.npy or stds.npy) not found in {train_cache_dir}")
        print("Please ensure pretraining/caching has been run for this fold.")
        sys.exit(1)
    means = np.load(means_path)   # shape: (C,) e.g. (10,)
    stds = np.load(stds_path)       # shape: (C,) e.g. (10,)
    print(f"✅ Loaded means (shape: {means.shape}) and stds (shape: {stds.shape}) for fold {args.fold}")

    # 処理対象のファイル一覧を取得
    target_files = []
    if args.datetime:
        # 特定の日時のファイルのみ処理
        target_filename = f"{args.datetime}.h5"
        target_filepath = os.path.join(input_h5_dir, target_filename)
        if os.path.exists(target_filepath):
            target_files.append(target_filepath)
        else:
            print(f"❌ Error: Specified datetime file not found: {target_filepath}")
            sys.exit(1)
    else:
        # 指定日付のすべてのファイルを処理
        for f_name in os.listdir(input_h5_dir):
            if f_name.startswith(args.date) and f_name.endswith(".h5"):
                try:
                    # ファイル名例: "20230101_000000.h5"
                    # 日付の妥当性チェック（オプション）
                    ts_str = os.path.splitext(f_name)[0]
                    datetime.strptime(ts_str, "%Y%m%d_%H%M%S") 
                    target_files.append(os.path.join(input_h5_dir, f_name))
                except ValueError:
                    print(f"⚠️ Warning: Could not parse timestamp from filename {f_name}. Skipping.")
    
    if not target_files:
        search_target = args.datetime if args.datetime else args.date
        print(f"ℹ️ No H5 files found for {search_target} in {input_h5_dir}")
        sys.exit(0)
    
    target_files.sort() # 時系列順に処理するためにソート
    print(f"ℹ️ Found {len(target_files)} files to process.")
    if args.debug:
        target_files = target_files[:3] # Debug: process only a few files
        print(f"🐛 Debug mode: processing first {len(target_files)} files.")

    # モデルのインスタンス化とチェックポイントのロード
    pretrain_model = get_pretrain_model(args.model_arch, device)
    pretrain_model = load_pretrain_checkpoint(pretrain_model, args.pretrain_checkpoint, map_location=device)
    pretrain_model.eval()

    # 各ファイルについて特徴量を抽出して保存
    for h5_file_path in target_files:
        file_basename = os.path.basename(h5_file_path)
        output_feature_filename = file_basename  # 同じファイル名で保存 (YYYYMMDD_HHMMSS.h5)
        output_feature_path = os.path.join(args.output_feature_dir, output_feature_filename)

        if os.path.exists(output_feature_path) and not args.debug: # Avoid re-computing if not in debug
            print(f"⏭️ Feature file already exists: {output_feature_path}. Skipping.")
            continue
        
        try:
            with h5py.File(h5_file_path, "r") as f:
                X_data = f["X"][:]  # Expected shape: (C, H, W), e.g., (10, 256, 256)
                # タイムスタンプも読み込み（存在する場合）
                # timestampがスカラー値の場合は[:]ではなく[()]を使用
                if "timestamp" in f:
                    timestamp_data = f["timestamp"]
                    if timestamp_data.shape == ():  # スカラー値の場合
                        timestamp = timestamp_data[()].decode('utf-8') if isinstance(timestamp_data[()], bytes) else str(timestamp_data[()])
                    else:  # 配列の場合
                        timestamp = timestamp_data[:].decode('utf-8') if isinstance(timestamp_data[:], bytes) else str(timestamp_data[:])
                else:
                    timestamp = os.path.splitext(file_basename)[0]
        except Exception as e:
            print(f"❌ Error reading H5 file {h5_file_path}: {e}")
            continue

        if X_data.ndim != 3 or X_data.shape[0] != means.shape[0]: # Basic sanity check
            print(f"❌ Error: Data in {h5_file_path} has unexpected shape {X_data.shape}. Expected ({means.shape[0]}, H, W). Skipping.")
            continue
            
        # 入力データの詳細情報を出力
        print(f"📊 Input data info for {file_basename}:")
        print(f"  - Input shape: {X_data.shape}")
        print(f"  - Input data type: {X_data.dtype}")
        print(f"  - Input min: {X_data.min():.6f}")
        print(f"  - Input max: {X_data.max():.6f}")
        print(f"  - Input mean: {X_data.mean():.6f}")
        print(f"  - Input std: {X_data.std():.6f}")
        
        # 正規化: (C, H, W) -> (C, H, W)
        # means/stds are (C,), need to reshape for broadcasting: (C, 1, 1)
        X_norm = (X_data - means[:, np.newaxis, np.newaxis]) / (stds[:, np.newaxis, np.newaxis] + 1e-8)
        
        # テンソルに変換し、バッチ次元を追加 -> (1, C, H, W)
        input_tensor = torch.from_numpy(X_norm).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            try:
                # Use forward_encoder_pyramid like in extract_mae_features
                batch_features, _, _ = pretrain_model.forward_encoder_pyramid(input_tensor, mask_ratio=0.0)
                # Extract mean features excluding CLS token (skip [:, 1:, :])
                mean_features = batch_features[:, 1:, :].mean(dim=1)
                
                # Convert to numpy - features should now be (1, embed_dim) -> (embed_dim,)
                features_np = mean_features.squeeze(0).cpu().numpy()
                
                # 特徴量の詳細情報を出力
                print(f"🔍 Features info for {file_basename}:")
                print(f"  - Shape: {features_np.shape}")
                print(f"  - Data type: {features_np.dtype}")
                print(f"  - Min value: {features_np.min():.6f}")
                print(f"  - Max value: {features_np.max():.6f}")
                print(f"  - Mean: {features_np.mean():.6f}")
                print(f"  - Std: {features_np.std():.6f}")
                print(f"  - Feature dimension: {features_np.shape[0] if len(features_np.shape) == 1 else features_np.shape}")
                
            except Exception as e:
                print(f"❌ Error during model inference for {h5_file_path}: {e}")
                continue
        
        try:
            # H5ファイルとして特徴量を保存
            with h5py.File(output_feature_path, 'w') as f:
                f.create_dataset("features", data=features_np)
                f.create_dataset("timestamp", data=timestamp.encode('utf-8'))
                # 元の形状情報も保存（デバッグ用）
                f.attrs["original_shape"] = X_data.shape
                f.attrs["features_shape"] = features_np.shape
            print(f"✅ Saved features for {file_basename} to {output_feature_path} (shape: {features_np.shape})")
        except Exception as e:
            print(f"❌ Error saving features to {output_feature_path}: {e}")

    print("🏁 Feature extraction complete.")

if __name__ == "__main__":
    main() 
    