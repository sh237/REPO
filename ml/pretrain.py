import os
import numpy as np
import torch
import pandas as pd
from torchinfo import summary
import gc
import argparse
import math
from models.pretrain.mae import (
    mae_vit_base_patch16_dec512d8b,
    vit_for_FT64d4b,
    vit_for_FT32d4b,
    vit_for_FT128db,
)

from datasets.pretrain.dataloader import (
    setup_datasets,
    setup_dataloaders,
    setup_all_data_loader,
    setup_visualization_loader,
    parse_time_range,
)
from utils.pretrain.losses import Losser
from utils.pretrain.logs import PretrainLogger
from utils.pretrain.io import setup_checkpoint_dir, load_model
from utils.pretrain.config import (
    load_config,
    update_args_from_config,
    get_periods_from_config,
)
from utils.pretrain.engine import (
    train_mae,
    eval_epoch,
    process_features,
    visualize_model_outputs,
    process_all_features,
    run_pretrain_workflow,
)

os.environ["NUMPY_EXPERIMENTAL_ARRAY_FUNCTION"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def setup_cache_dirs(data_root, fold):
    """Set up cache directory paths"""
    cache_base = os.path.join(data_root, f"pretrain/cache/fold{fold}")
    cache_dirs = {
        "train": os.path.join(cache_base, "train"),
        "val": os.path.join(cache_base, "val"),
        "test": os.path.join(cache_base, "test"),
    }
    for dir_path in cache_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return cache_dirs


def get_model_from_config(model_config, device):
    """Create model based on configuration"""
    model_type = model_config.get("type", "vit_for_FT128db")

    # Common parameters
    common_params = {
        "in_chans": model_config.get("in_chans", 10),
        "mask_ratio": model_config.get("mask_ratio", 0.75),
        "stdwise": model_config.get("stdwise", False),
        "pyramid": model_config.get("pyramid", True),
        "sunspot": model_config.get("sunspot", True),
        "base_mask_ratio": model_config.get("base_mask_ratio", 0.5),
        "sunspot_spatial_ratio": model_config.get("sunspot_spatial_ratio", 0.35),
        "feature_mask_ratio": model_config.get("feature_mask_ratio", 0.75),
    }

    # Create model based on type
    if model_type == "vit_for_FT128db":
        model = vit_for_FT128db(**common_params)
    elif model_type == "vit_for_FT64d4b":
        model = vit_for_FT64d4b(**common_params)
    elif model_type == "vit_for_FT32d4b":
        model = vit_for_FT32d4b(**common_params)
    elif model_type == "mae_vit_base_patch16_dec512d8b":
        model = mae_vit_base_patch16_dec512d8b(**common_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def pretrain_main(args):
    # Load configuration if specified
    config = {}
    if args.config:
        config = load_config(args.config)
        args = update_args_from_config(args, config)
    else:
        # If no config file is provided, ensure default values are set
        default_config = {
            "trial_name": "mae_default",
            "mode": "train",
            "fold": 1,
            "data_root": "data",
            "input_dir": "data/solar_flare",
            "output_dir": "results/features",
            "batch_size": 32,
            "num_workers": 4,
            "epochs": 20,
            "mask_ratio": 0.75,
            "cuda_device": 0,
        }
        args = update_args_from_config(args, default_config)

    # Set up logger
    logger = PretrainLogger(args.trial_name, args.fold, args.use_wandb)

    # Set up cache directories
    cache_dirs = setup_cache_dirs(args.data_root, args.fold)

    # Set up checkpoint directory
    checkpoint_dir = setup_checkpoint_dir(args.trial_name)

    # Log configuration
    logger.log_config(args)

    # Ensure cuda_device is set
    if not hasattr(args, "cuda_device") or args.cuda_device is None:
        args.cuda_device = 0

    # Set up device
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )
    logger.log_info(f"Using device: {device}")

    # Get periods from config or use defaults
    periods = get_periods_from_config(config, args.fold)
    train_periods = periods.get("train", [])
    val_periods = periods.get("val", [])
    test_periods = periods.get("test", [])
    vis_periods = test_periods  # Visualization periods are the same as test periods
    
    # Log PyTorch and CUDA versions
    logger.log_info(f"PyTorch Version: {torch.__version__}")
    logger.log_info(f"PyTorch CUDA Version: {torch.version.cuda}")
    logger.log_info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.log_info(f"CUDA Device Count: {torch.cuda.device_count()}")
        # current_device = torch.cuda.current_device() # This might fail if args.cuda_device is not yet set or invalid
        # To be safe, let's use args.cuda_device if it's set, otherwise 0 if CUDA is available.
        device_idx_to_check = 0
        if hasattr(args, 'cuda_device') and args.cuda_device is not None:
            device_idx_to_check = args.cuda_device
        
        # Ensure the device index is valid before querying
        if device_idx_to_check < torch.cuda.device_count():
            logger.log_info(f"Checking properties for CUDA Device: {device_idx_to_check}")
            logger.log_info(f"Device Name: {torch.cuda.get_device_name(device_idx_to_check)}")
            logger.log_info(f"Compute Capability: {torch.cuda.get_device_capability(device_idx_to_check)}")
        else:
            logger.log_info(f"Configured CUDA device index {device_idx_to_check} is out of range. Available devices: {torch.cuda.device_count()}.")
    else:
        logger.log_info("CUDA is not available to PyTorch.")


    # Set up model from config
    if hasattr(args, "model_config"):
        model = get_model_from_config(args.model_config, device)
    else:
        # Fallback to default model
        model = vit_for_FT128db(
            in_chans=10,
            mask_ratio=args.mask_ratio,
            stdwise=False,
            pyramid=True,
            sunspot=True,
            base_mask_ratio=0.5,
            sunspot_spatial_ratio=0.3,
            feature_mask_ratio=0.5,
        ).to(device)

    # Log model summary
    model_summary = summary(
        model, input_size=(args.batch_size, 10, 256, 256), verbose=0
    )
    logger.log_model_summary(model_summary)

    if args.mode == "train":
        # Set up datasets and dataloaders
        train_dataset, val_dataset, test_dataset = setup_datasets(
            args, cache_dirs, train_periods, val_periods, test_periods
        )
        train_loader, val_loader, test_loader = setup_dataloaders(
            args, train_dataset, val_dataset, test_dataset
        )

        # Run pretraining workflow
        model = run_pretrain_workflow(
            args, model, train_loader, val_loader, test_loader, checkpoint_dir, logger
        )

    elif args.mode == "test":
        # Set up datasets and dataloaders
        _, _, test_dataset = setup_datasets(
            args, cache_dirs, train_periods, val_periods, test_periods
        )
        _, _, test_loader = setup_dataloaders(args, None, None, test_dataset)

        # Load model
        model = load_model(model, checkpoint_dir, args.trial_name)
        model.eval()

        # Test evaluation
        losser = Losser(model, device=device)
        test_metrics = eval_epoch(model, test_loader, losser)
        logger.log_final_metrics(test_metrics)

    elif args.mode == "visualize":
        # Set up visualization dataloader
        test_loader = setup_visualization_loader(args, cache_dirs, vis_periods)

        # Filter by timestamp if specified
        time_range = parse_time_range(
            args.visualize_timestamp,
            hours_before=(
                args.visualize_config.get("hours_before", 12)
                if hasattr(args, "visualize_config")
                else 12
            ),
            hours_after=(
                args.visualize_config.get("hours_after", 12)
                if hasattr(args, "visualize_config")
                else 12
            ),
        )

        if time_range:
            start_time, end_time = time_range
            logger.log_info(
                f"Using time range around {args.visualize_timestamp}: {start_time} to {end_time}."
            )

        # Load model
        model = load_model(model, checkpoint_dir, args.trial_name).to(device)
        model.eval()

        # Run visualization
        visualize_model_outputs(
            model,
            test_loader,
            device,
            os.path.join("results", "reconstruct_images", args.trial_name),
            args.trial_name,
            num_images=(
                args.visualize_config.get("num_images", 30)
                if hasattr(args, "visualize_config")
                else 30
            ),
            use_sunspot_masking=(
                args.visualize_config.get("use_sunspot_masking", True)
                if hasattr(args, "visualize_config")
                else True
            ),
            time_range=time_range,
        )

    elif args.mode == "embed":
        # Set up datasets and dataloaders
        train_dataset, val_dataset, test_dataset = setup_datasets(
            args, cache_dirs, train_periods, val_periods, test_periods
        )
        train_loader, val_loader, test_loader = setup_dataloaders(
            args, train_dataset, val_dataset, test_dataset
        )

        # Load model
        model = load_model(model, checkpoint_dir, args.trial_name)
        model.eval()

        # Extract features
        process_all_features(
            model,
            [train_loader, val_loader, test_loader],
            ["train", "val", "test"],
            args.mask_ratio,
            device,
            args.output_dir,
            args.trial_name,
        )

    elif args.mode == "inference_all":
        # Set up all data loader
        all_loader = setup_all_data_loader(args, cache_dirs)

        # Load model
        model = load_model(model, checkpoint_dir, args.trial_name)
        model.eval()

        # Extract features for all data
        process_features(
            model,
            all_loader,
            args.mask_ratio,
            device,
            args.output_dir,
            "all_data",
            args.trial_name,
        )


def add_arguments(parser):
    """Set up command line arguments"""
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        help="Execution mode (train/test/visualize/embed/inference_all)",
    )
    parser.add_argument("--trial_name", type=str, default=None, help="Trial name")
    parser.add_argument(
        "--fold", type=int, default=None, help="Fold number to use (1-5)"
    )
    parser.add_argument(
        "--data_root", type=str, default=None, help="Data root directory"
    )
    parser.add_argument(
        "--input_dir", type=str, default=None, help="Input data directory"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of workers for data loader",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--mask_ratio", type=float, default=None, help="Mask ratio")
    parser.add_argument(
        "--cuda_device", type=int, default=None, help="CUDA device to use"
    )
    parser.add_argument(
        "--use_wandb", action="store_true", help="Whether to use Weights & Biases"
    )
    parser.add_argument(
        "--force_recalc",
        action="store_true",
        help="Whether to force recalculation of statistics",
    )
    parser.add_argument(
        "--visualize_timestamp",
        type=str,
        default=None,
        help="Specific timestamp to visualize (format: YYYYMMDD_HHMMSS)",
    )

    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_arguments(parser)
    args = parser.parse_args()
    pretrain_main(args)
