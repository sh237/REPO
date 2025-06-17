"""Train Model"""

import json
import argparse
import torch
import models.main.model
import torch.nn as nn
import numpy as np
import logging
import torch.profiler
import wandb
import os
import yaml
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from argparse import Namespace
from typing import Dict, Optional, Tuple, Any
from torchinfo import summary
from utils.main.statistics import Stat, compute_statistics

from utils.main.utils import fix_seed
from utils.main.losses import LossConfig, Losser
from utils.main.engine import train_epoch, eval_epoch
from utils.main.logs import Log, Logger, setup_logging
from utils.main.server import CallbackServer
from utils.main.config import parse_params

from datasets.main.dataloader import prepare_dataloaders
import torch.multiprocessing as mp
from schedulefree import RAdamScheduleFree
from thop import profile
from utils.main.io import save_checkpoint, load_checkpoint


class ExperimentManager:
    """
    Manager class for Model
    """

    def __init__(self, args: Namespace):
        # Setup logging
        self.logger = setup_logging(args.trial_name)
        self.logger.info(f"Using fold {args.fold}")
        self.logger.info("Dataset configuration:")
        self.logger.info(f"force_preprocess: {args.dataset.get('force_preprocess')}")
        self.logger.info(
            f"force_recalc_indices: {args.dataset.get('force_recalc_indices')}"
        )
        self.logger.info(
            f"force_recalc_stats: {args.dataset.get('force_recalc_stats')}"
        )

        self.log_writer = Logger(args, wandb=args.wandb)
        fix_seed(seed=42)

        if args.wandb:
            wandb.init(
                project="SolarFlarePrediction2",
                name=f"{args.trial_name}_fold{args.fold}",
            )

        self.current_stage = args.stage
        self.args = args

        # Calculate statistics
        stats_dir = os.path.join(args.cache_root, "statistics", f"fold{args.fold}")
        full_climatology, gmgs_score_matrix, stat = compute_statistics(
            data_dir=args.data_path,
            stats_dir=stats_dir,
            train_periods=args.train_periods,
            force_recalc=args.dataset["force_recalc_stats"],
            logger=self.logger,
        )
        self.stat = stat

        # Initialize dataloaders only if no checkpoint to resume from
        if args.mode == "train" and not args.resume_from_checkpoint:
            sample = self.load_dataloaders(args, args.imbalance)
        else:
            args.detail_summary = False
            sample = None

        # Prepare model and optimizer
        model, losser, optimizer, stat = self._build(args, sample)

        self.model = model
        self.losser = losser
        self.optimizer = optimizer
        self.stat = stat
        self.best_valid_gmgs = float("-inf")
        self.best_model_path = None
        self.best_train_loss = None
        self.best_valid_loss = None
        self.best_valid_score = None
        self.train_score = None
        self.valid_score = None
        self.test_score = None
        self.train_loss = None
        self.valid_loss = None
        self.test_loss = None

        # Add attributes for confusion matrix
        self.best_valid_predictions = None
        self.best_valid_observations = None
        self.test_predictions = None
        self.test_observations = None

        # Add variables for early stopping
        self.es_metric = args.early_stopping["metric"]
        self.patience = args.early_stopping["patience"]
        self.patience_counter = 0
        self.should_stop = False

        # Initialize best score based on metric
        self.best_metric_value = (
            float("-inf") if "GMGS" in self.es_metric else float("inf")
        )

        # Add variable for stage management
        self.current_stage = 1

        # Log configuration
        self.logger.info("\n=== Configuration ===")
        self.logger.info(yaml.dump(vars(args), default_flow_style=False))
        self.logger.info("==================\n")

    def _build(
        self, args: Namespace, sample: Any
    ) -> Tuple[nn.Module, Losser, RAdamScheduleFree, Stat]:
        """
        Build model, losser, optimizer, stat
        """
        print("Prepare model and optimizer", end="")
        loss_config = LossConfig(
            lambda_bss=args.factor["BS"],
            lambda_gmgs=args.factor["GMGS"],
            lambda_ce=args.factor["CE"],
            score_mtx=self.stat.gmgs_score_matrix,
            fold=args.fold,
            class_weights=args.class_weights,
            model_name=args.model.selected,
            stage=self.current_stage,
        )

        # Model
        Model = self._get_model_class(args.model.selected)

        # Convert Namespace to dict (including nested structures)
        def namespace_to_dict(ns):
            if isinstance(ns, Namespace):
                return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
            elif isinstance(ns, (list, tuple)):
                return [namespace_to_dict(x) for x in ns]
            else:
                return ns

        architecture_params = namespace_to_dict(
            args.model.models[args.model.selected].architecture_params
        )
        model = Model(**architecture_params).to(args.device)

        # Calculate computational complexity and number of parameters
        dummy_input1 = torch.randn(1, 4, 10, 256, 256).to(args.device)
        dummy_input2 = torch.randn(1, 672, 128).to(args.device)
        macs, params = profile(model, inputs=(dummy_input1, dummy_input2))

        # Record to log
        self.logger.info(
            "=========================================================================================="
        )
        self.logger.info(
            f"MACs: {macs/1e6:.2f}M"
        )  # MMACs (Million Multiply-Add operations)
        self.logger.info(
            "=========================================================================================="
        )

        # Set optimizer
        if args.optimizer == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay,
            )

            # Custom scheduler function
            def lr_func(epoch):
                return min(
                    (epoch + 1) / (args.warmup_epochs + 1e-8),
                    0.5 * (math.cos(epoch / args.cosine_epochs * math.pi) + 1),
                )

            scheduler = LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)
            self.scheduler = scheduler
        else:  # radam_free
            optimizer = RAdamScheduleFree(
                model.parameters(),
                lr=args.lr,
                betas=(0.9, 0.999),
                weight_decay=args.weight_decay,
            )

        losser = Losser(loss_config, device=args.device)
        stat = self.stat

        print(" ... ok")

        model_summary = summary(model, verbose=0)

        self.logger.info("Model Summary:\n" + str(model_summary))

        return model, losser, optimizer, stat

    def _get_model_class(self, name: str) -> nn.Module:
        mclass = models.main.model.__dict__[name]
        return mclass

    def is_better_score(self, current_value: float) -> bool:
        """
        Determine if current value is better than best score
        GMGS: higher is better, loss: lower is better
        """
        if "GMGS" in self.es_metric:
            return current_value > self.best_metric_value
        else:  # loss
            return current_value < self.best_metric_value

    def train_epoch(self, epoch):
        (train_dl, val_dl, _) = self.dataloaders

        # Set to training mode
        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

        # train
        train_score, train_loss = train_epoch(
            self.model,
            self.optimizer,
            train_dl,
            losser=self.losser,
            stat=self.stat,
            args=self.args,
        )

        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        # validation
        valid_score, valid_loss = eval_epoch(
            self.model,
            val_dl,
            losser=self.losser,
            stat=self.stat,
            args=self.args,
            mode="valid",
            optimizer=self.optimizer, 
        )

        if self.es_metric == "valid_GMGS":
            current_metric_value = valid_score["valid_GMGS"]
        else:  # valid_loss
            current_metric_value = np.mean(valid_loss)

        if self.is_better_score(current_metric_value):
            self.best_metric_value = current_metric_value
            self.patience_counter = 0 

            best_checkpoint_path = self.save_checkpoint(self.current_stage)
            self.logger.info(
                f"New best model (stage {self.current_stage}) saved with "
                f"{self.es_metric}: {current_metric_value:.4f}"
            )

            # Process when the best score is updated
            self.best_train_loss = np.mean(train_loss)
            self.best_valid_loss = np.mean(valid_loss)
            self.best_valid_score = valid_score
            self.best_valid_predictions = [
                np.argmax(p) for p in self.stat.predictions["valid"]
            ]
            self.best_valid_observations = self.stat.observations["valid"]
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                self.logger.info(
                    f"Early stopping triggered after {self.patience} epochs without "
                    f"improvement in {self.es_metric}"
                )

        self.log_writer.write(
            epoch,
            [
                Log("train", np.mean(train_loss), train_score),
                Log("valid", np.mean(valid_loss), valid_score),
            ],
        )

        self.logger.info(
            f"Epoch {epoch}: Train loss:{np.mean(train_loss):.4f}  Valid loss:{np.mean(valid_loss):.4f}"
        )
        self.logger.info(
            f"Epoch {epoch}: Train score:{train_score}  Valid score:{valid_score}"
        )

        # === wandb logging ===
        if self.args.wandb:
            log_data = {
                "epoch": epoch,
                "train_loss": np.mean(train_loss),
                **train_score,
                "valid_loss": np.mean(valid_loss),
                **valid_score,
            }

            # 混同行列の追加（validation）
            if "confusion_matrix" in valid_score:
                cm = np.array(valid_score["confusion_matrix"])  # 念のため numpy 配列に変換
                num_classes = cm.shape[0]  # クラス数を取得
                log_data["valid_confusion_matrix"] = wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=None,
                    preds=None,
                    class_names=[str(i) for i in range(num_classes)],
                    matrix_values=cm.tolist(),
                )

            wandb.log(log_data)

        self.train_score = train_score
        self.valid_score = valid_score
        self.train_loss = np.mean(train_loss)
        self.valid_loss = np.mean(valid_loss)

        # === 2エポックごとの test evaluation ===
        if epoch % 2 == 0:
            self.logger.info(f"====== Test evaluation at epoch {epoch} ======")
            self.test(save_qualitative=False)

            if self.args.wandb and self.test_score is not None:
                test_log_data = {
                    "epoch": epoch,
                    "test_loss": self.test_loss,
                    **self.test_score,
                }

                # 混同行列の追加（test）
                if "confusion_matrix" in self.test_score:
                    cm = np.array(self.test_score["confusion_matrix"])
                    num_classes = cm.shape[0]
                    test_log_data["test_confusion_matrix"] = wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=None,
                        preds=None,
                        class_names=[str(i) for i in range(num_classes)],
                        matrix_values=cm.tolist(),
                    )

                wandb.log(test_log_data)

    def train(self, lr: Optional[float] = None, epochs: Optional[int] = None):
        """
        Train the model for the specified number of epochs
        """
        if self.current_stage == 1:
            lr = lr or self.args.lr
            epochs = epochs or self.args.epochs
        else:
            lr = lr or self.args.lr_for_2stage
            epochs = epochs or self.args.epoch_for_2stage

        self.logger.info(f"Starting stage {self.current_stage} training")
        self.logger.info(f"Learning rate: {lr}, Epochs: {epochs}")

        for epoch in range(epochs):
            if self.should_stop:
                self.logger.info("Early stopping triggered")
                break

            self.logger.info(f"====== Epoch {epoch} ======")
            self.train_epoch(epoch)

            # Execute scheduler step for AdamW
            if self.args.optimizer == "adamw":
                self.scheduler.step()

        # Evaluate best model
        self.logger.info(f"\n=== Evaluating stage {self.current_stage} ===")
        best_checkpoint_path = os.path.join(
            "checkpoints",
            "main",
            f"{self.args.trial_name}_stage{self.current_stage}_best.pth",
        )
        self.load_checkpoint(best_checkpoint_path)
        self.test(save_qualitative=False)
        self.print_best_metrics(stage=f"{self.current_stage}st")

    def load(self, path: str):
        """
        Load model from path
        """
        self.model.load_state_dict(torch.load(path))

    def load_dataloaders(self, args: Namespace, imbalance: bool):
        dataloaders, sample = prepare_dataloaders(args, args.debug, imbalance)
        self.train_dl, self.valid_dl, self.test_dl = (
            dataloaders 
        )
        self.dataloaders = dataloaders
        return sample

    def test(self, save_qualitative: bool = False):
        """Test model and log results to wandb"""
        self.test_score, test_loss = eval_epoch(
            self.model,
            self.test_dl,
            losser=self.losser,
            stat=self.stat,
            args=self.args,
            mode="test",
            save_qualitative=save_qualitative,
            trial_name=self.args.trial_name,
            optimizer=self.optimizer,
        )
        self.test_loss = np.mean(test_loss)

        self.test_predictions = [np.argmax(p) for p in self.stat.predictions["test"]]
        self.test_observations = self.stat.observations["test"]

        if self.args.wandb:
            wandb.log(
                {
                    "test_loss": self.test_loss,
                    **self.test_score,
                }
            )

    def print_summary(self):
        """Print model summary"""
        self.log_writer.print_model_summary(
            self.model,
            args=self.args if hasattr(self, "mock_sample") else None,
            mock_sample=self.mock_sample if hasattr(self, "mock_sample") else None,
        )

    def freeze_feature_extractor(self):
        """
        Freeze feature extractor
        """
        self.model.freeze_feature_extractor()

    def reset_optimizer(self, lr: Optional[float] = None):
        """
        Reset optimizer with new lr
        """
        if self.args.optimizer == "adamw":
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=lr or self.args.lr,
                betas=(0.9, 0.95),
                weight_decay=self.args.weight_decay,
            )

            def lr_func(epoch):
                return min(
                    (epoch + 1) / (self.args.warmup_epochs + 1e-8),
                    0.5 * (math.cos(epoch / self.args.cosine_epochs * math.pi) + 1),
                )

            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_func, verbose=True)
        else:
            self.optimizer = RAdamScheduleFree(
                self.model.parameters(),
                lr=lr or self.args.lr,
                betas=(0.9, 0.999),
                weight_decay=self.args.weight_decay,
            )

        loss_config = LossConfig(
            lambda_bss=self.args.factor["BS"],
            lambda_gmgs=self.args.factor["GMGS"],
            lambda_ce=self.args.factor["CE"],
            score_mtx=self.stat.gmgs_score_matrix,
            fold=self.args.fold,
            class_weights=self.args.class_weights,
            model_name=self.args.model.selected,
            stage=2,
        )

        self.losser = Losser(loss_config, self.args.device)

    def print_best_metrics(self, stage: str = "1st"):
        """Output evaluation metrics for the best model"""
        self.log_writer.print_best_metrics(self, stage)

    def save_checkpoint(self, stage: int):
        return save_checkpoint(
            self.model, self.optimizer, self.best_valid_gmgs, stage, self.args
        )

    def load_checkpoint(self, checkpoint_path):
        config, self.best_valid_gmgs, self.start_epoch = load_checkpoint(
            self.model,
            self.optimizer,
            checkpoint_path,
            self.args.device,
            scheduler=getattr(self, "scheduler", None),
        )
        return config
    
    def predict_one_shot(self, sample_input1, sample_input2):
        """
        1サンプルの入力を受け取り、4クラスの尤度（確率分布）を返す関数（CPU推論用）

        :param sample_input1: 1サンプルの画像データテンソル（例：[1, 4, 10, 256, 256]）
        :param sample_input2: 1サンプルの特徴量データテンソル（例：[1, 672, 128]）
        :return: 4クラスの尤度リスト（例：[0.1, 0.1, 0.7, 0.1]）
        """
        # CPU上での推論のため、明示的にCPUのデバイスを使用
        device = torch.device("cpu")
        self.model.to(device)
        sample_input1 = sample_input1.to(device)
        sample_input2 = sample_input2.to(device)
        
        # 評価モードに切替え、勾配計算を無効化
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(sample_input1, sample_input2)
            probabilities = torch.softmax(logits, dim=1)
        
        # バッチサイズが1の場合、1サンプル分の尤度リストを返す
        return probabilities[0].cpu().numpy().tolist()

    def infer_all_valid_samples(self, valid_dataset):
        """
        valid_dataset 内の全サンプルについて、1ショット推論を行い4クラスの尤度を返す関数。

        :param valid_dataset: SolarFlareDatasetWithFeatures の 'valid' 分割（__getitem__ が valid_indices に沿ったサンプルを返す）
        :return: 各サンプルごとの4クラスの尤度のリスト（例：[[0.1,0.1,0.7,0.1], [ ... ], ...]）
        """
        self.model.to("cpu")
        self.model.eval()
        all_probabilities = []
        
        # 各サンプルに対して推論を実施
        for idx in range(len(valid_dataset)):
            X, h, y = valid_dataset[idx]
            # Xは [history, channels, height, width] なので、バッチ次元を追加
            sample_input1 = X.unsqueeze(0)
            sample_input2 = h.unsqueeze(0)
            
            with torch.no_grad():
                # predict_one_shotは内部でソフトマックスを適用して4クラスの尤度を返す
                probabilities = self.predict_one_shot(sample_input1, sample_input2)
            
            all_probabilities.append(probabilities)
        
        return all_probabilities


def main():
    args, _ = parse_params(dump=True)
    experiment = ExperimentManager(args)

    if args.mode == "train":
        if args.resume_from_checkpoint:
            experiment.logger.info(
                f"Resuming from checkpoint: {args.resume_from_checkpoint}"
            )
            config = experiment.load_checkpoint(args.resume_from_checkpoint)

            # Set based on specified stage
            experiment.current_stage = args.stage

            # Apply special settings only for stage 2
            if args.stage == 2:
                experiment.freeze_feature_extractor()
                experiment.reset_optimizer(lr=args.lr_for_2stage)

            # Set imbalance based on stage
            imbalance = args.stage == 1

            # Initialize dataloaders
            experiment.load_dataloaders(args, imbalance=imbalance)

            # Training settings based on stage
            if args.stage == 2:
                experiment.train(lr=args.lr_for_2stage, epochs=args.epoch_for_2stage)
            else:
                experiment.train()

        else:
            # Stage 1 training
            experiment.train()

            if args.imbalance:
                # Stage 2 settings
                experiment.current_stage = 2
                experiment.should_stop = False
                experiment.patience_counter = 0
                experiment.best_valid_gmgs = float("-inf")

                experiment.freeze_feature_extractor()
                experiment.reset_optimizer(lr=args.lr_for_2stage)
                experiment.load_dataloaders(args, imbalance=False)

                # Stage 2 training
                experiment.train(lr=args.lr_for_2stage, epochs=args.epoch_for_2stage)

    elif args.mode == "test":
        # Get checkpoint path from arguments
        if not args.resume_from_checkpoint:
            experiment.logger.error(
                "Please specify checkpoint path with --resume_from_checkpoint"
            )
            return

        # Prepare dataloaders (for test mode)
        experiment.load_dataloaders(args, imbalance=True)

        # Load checkpoint
        experiment.logger.info(
            f"Loading checkpoint from: {args.resume_from_checkpoint}"
        )
        experiment.load_checkpoint(args.resume_from_checkpoint)
        experiment.model.eval().to(args.device)

        # Evaluate on test data
        test_score, test_loss = eval_epoch(
            experiment.model,
            experiment.test_dl,
            experiment.losser,
            experiment.stat,
            args,
            mode="test",
            save_qualitative=True,
            trial_name=args.trial_name,
            optimizer=experiment.optimizer,
        )

        # Save predictions and ground truth
        results_dir = os.path.join("results", "analysis_results")
        os.makedirs(results_dir, exist_ok=True)

        # Get dataset indices
        test_indices = (
            experiment.test_dl.dataset.indices
            if hasattr(experiment.test_dl.dataset, "indices")
            else range(len(experiment.test_dl.dataset))
        )

        # Organize predictions
        predictions = [np.argmax(p) for p in experiment.stat.predictions["test"]]
        observations = experiment.stat.observations["test"]

        # Sort by index
        results = []
        for idx, pred, obs in zip(test_indices, predictions, observations):
            results.append(
                {"index": int(idx), "prediction": int(pred), "ground_truth": int(obs)}
            )

        # Sort by index
        results.sort(key=lambda x: x["index"])

        # Save as JSON
        stage = "stage1" if "stage1" in args.resume_from_checkpoint else "stage2"
        results_path = os.path.join(
            results_dir, f"{args.trial_name}_{stage}_results.json"
        )
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        experiment.logger.info(f"Results saved to: {results_path}")
        experiment.logger.info(f"Test Loss: {np.mean(test_loss):.4f}")
        for metric, value in test_score.items():
            if isinstance(value, list):
                experiment.logger.info(f"{metric}: {value}")  # リストはそのまま出力
            else:
                experiment.logger.info(f"{metric}: {value:.4f}")  # 数値の場合はフォーマット

    elif args.mode == "server":
        if not os.path.exists(experiment.best_model_path):
            experiment.logger.error(
                "Best model not found. Please train the model first."
            )
            return
        experiment.load(experiment.best_model_path)
        CallbackServer.start(callback=experiment.predict_one_shot)
    else:
        assert False, "Unknown mode"

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()