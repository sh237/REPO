"""
Training and evaluation functions for pre-training
"""

import os
import torch
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
import gc
import math

from utils.pretrain.losses import Losser
from utils.pretrain.io import save_model, load_model, apply_model_state


def train_epoch(
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_dl: DataLoader,
    losser: Losser,
) -> Dict[str, float]:
    """
    Perform training for one epoch

    Parameters:
        model: Model to train
        optimizer: Optimizer
        train_dl: Training data loader
        losser: Loss function

    Returns:
        metrics: Training metrics
    """
    model.train()
    losser.clear()
    total_loss = 0.0
    num_batches = 0

    for _, (x, y, _) in enumerate(tqdm(train_dl, desc="Training")):
        optimizer.zero_grad()
        imgs = x.to(losser.device).float()
        pred, mask = model(imgs)
        loss = losser(imgs, pred, mask)

        if loss is not None:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    metrics = {
        "mse": avg_loss,
        "mae": losser.get_metrics()["mae"],
        "solar_mse": losser.get_metrics()["solar_mse"],
        "solar_mae": losser.get_metrics()["solar_mae"],
    }

    return metrics


def eval_epoch(
    model: torch.nn.Module, val_dl: DataLoader, losser: Losser
) -> Dict[str, float]:
    """
    Evaluate the model

    Parameters:
        model: Model to evaluate
        val_dl: Evaluation data loader
        losser: Loss function

    Returns:
        metrics: Evaluation metrics
    """
    model.eval()
    losser.clear()

    with torch.no_grad():
        for _, (x, y, _) in enumerate(tqdm(val_dl, desc="Evaluating")):
            imgs = x.to(losser.device).float()
            pred, mask = model(imgs)
            losser.evaluate(imgs, pred, mask)

    return losser.get_metrics()


def train_mae(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    losser: Losser,
    trial_name: str,
    checkpoint_dir: str,
    num_epochs: int = 10,
    logger: Optional[Any] = None,
) -> torch.nn.Module:
    """
    Train the MAE model

    Parameters:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        losser: Loss function
        trial_name: Trial name
        checkpoint_dir: Checkpoint directory
        num_epochs: Number of epochs
        logger: Logger

    Returns:
        model: Trained model
    """
    best_val_loss = float("inf")
    best_model = None

    for e in range(num_epochs):
        # Training
        train_metrics = train_epoch(model, optimizer, train_loader, losser)
        lr_scheduler.step()

        # Validation
        val_metrics = eval_epoch(model, val_loader, losser)

        # Log using logger
        is_best = val_metrics["solar_mse"] < best_val_loss
        if logger:
            logger.log_train_step(
                e,
                train_metrics["mse"],
                val_metrics,
                optimizer.param_groups[0]["lr"],
                is_best=is_best,
            )

        # Save the best model
        if is_best:
            best_val_loss = val_metrics["solar_mse"]
            best_model = model.state_dict().copy()
            save_model(model, checkpoint_dir, trial_name, is_best=True)

            if logger:
                logger.log_info(
                    f"Epoch {e}: New best model with val_solar_mse = {best_val_loss:.6f}"
                )

    # Final test evaluation
    if best_model:
        model = apply_model_state(model, best_model)
        test_metrics = eval_epoch(model, val_loader, losser)

        if logger:
            logger.log_final_metrics(test_metrics)

    return model


def process_features(
    model: torch.nn.Module,
    data_loader: DataLoader,
    mask_ratio: float,
    device: torch.device,
    output_dir: str,
    dataset_type: str,
    trial_name: str,
) -> bool:
    """
    Extract and save features

    Parameters:
        model: Model to extract features
        data_loader: Data loader
        mask_ratio: Mask ratio
        device: Device
        output_dir: Output directory
        dataset_type: Dataset type
        trial_name: Trial name

    Returns:
        success: Whether the processing was successful
    """
    from utils.pretrain.feature_extraction import process_dataset_features

    try:
        process_dataset_features(
            model,
            data_loader,
            mask_ratio,
            device,
            output_dir,
            dataset_type,
            trial_name,
        )
        return True
    except Exception as e:
        print(f"Error processing features for {dataset_type}: {str(e)}")
        return False


def visualize_model_outputs(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_dir: str,
    trial_name: str,
    num_images: int = 30,
    use_sunspot_masking: bool = True,
    time_range: Optional[Tuple] = None,
) -> bool:
    """
    Visualize model outputs

    Parameters:
        model: Model to visualize
        data_loader: Data loader
        device: Device
        output_dir: Output directory
        trial_name: Trial name
        num_images: Number of images to visualize
        use_sunspot_masking: Whether to use sunspot masking
        time_range: Time range

    Returns:
        success: Whether the processing was successful
    """
    from utils.pretrain.visualize import visualize_reconstruction

    try:
        visualize_reconstruction(
            model,
            data_loader,
            device,
            output_dir,
            trial_name,
            num_images=num_images,
            use_sunspot_masking=use_sunspot_masking,
            time_range=time_range,
        )
        return True
    except Exception as e:
        print(f"Error visualizing model outputs: {str(e)}")
        return False


def process_all_features(
    model: torch.nn.Module,
    loaders: List[DataLoader],
    dataset_types: List[str],
    mask_ratio: float,
    device: torch.device,
    output_dir: str,
    trial_name: str,
) -> Dict[str, bool]:
    """
    Extract features for multiple datasets

    Parameters:
        model: Model to extract features
        loaders: List of data loaders
        dataset_types: List of dataset types
        mask_ratio: Mask ratio
        device: Device
        output_dir: Output directory
        trial_name: Trial name

    Returns:
        results: Processing results for each dataset
    """
    results = {}

    for loader, dataset_type in zip(loaders, dataset_types):
        print(f"Processing features for {dataset_type} dataset...")
        success = process_features(
            model, loader, mask_ratio, device, output_dir, dataset_type, trial_name
        )
        results[dataset_type] = success

        # Memory release
        gc.collect()
        torch.cuda.empty_cache()

    return results


def run_pretrain_workflow(
    args,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    checkpoint_dir: str,
    logger: Optional[Any] = None,
) -> torch.nn.Module:
    """
    Run the pre-training workflow

    Parameters:
        args: Command line arguments
        model: Model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        checkpoint_dir: Checkpoint directory
        logger: Logger

    Returns:
        model: Trained model
    """
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    )

    # Optimizer and scheduler settings
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=4e-3, betas=(0.9, 0.95), weight_decay=0.05
    )
    lr_func = lambda epoch: min(
        (epoch + 1) / (10 + 1e-8), 0.5 * (math.cos(epoch / 20 * math.pi) + 1)
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lr_func, verbose=True
    )

    # Loss function settings
    losser = Losser(model, device=device)

    # Model training
    model = train_mae(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        losser,
        args.trial_name,
        checkpoint_dir,
        num_epochs=args.epochs,
        logger=logger,
    )

    # Feature extraction
    process_all_features(
        model,
        [train_loader, val_loader, test_loader],
        ["train", "val", "test"],
        args.mask_ratio,
        device,
        args.output_dir,
        args.trial_name,
    )

    return model
