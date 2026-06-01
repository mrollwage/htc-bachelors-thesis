import csv
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager
from src.models.backbone import HierarchicalClassifier
from src.training.loss import compute_loss
from src.training.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for all hierarchical text classification models.

    Handles the full training loop including:
        - Per-epoch training and validation
        - Automatic Mixed Precision (AMP) when CUDA is available
        - Gradient clipping
        - Linear learning rate schedule with warmup
        - Early stopping based on validation hierarchical accuracy
        - Best model checkpointing
        - Per-epoch metric logging to CSV

    Designed to be model-agnostic: loss is taken directly from the model's
    forward() output if present (e.g. LCPN computes it internally), otherwise
    computed via the central compute_loss() function.

    Args:
        model: Any subclass of HierarchicalClassifier.
        config: Dataset-specific config.
        hierarchy: Fully built HierarchyManager.
        train_loader: DataLoader for the training split.
        val_loader: DataLoader for the validation split.
        experiment_name: Identifier used for checkpoint and log filenames.
    """

    def __init__(
        self,
        model: HierarchicalClassifier,
        config: BaseConfig,
        hierarchy: HierarchyManager,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_name: str,
    ):
        self.model = model
        self.config = config
        self.hierarchy = hierarchy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.experiment_name = experiment_name

        self.device = torch.device(config.device)
        self.model.to(self.device)

        # AMP: enabled automatically when training on CUDA
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Optimiser
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Linear schedule with warmup over the full training budget
        total_steps = len(train_loader) * config.max_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimiser,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps,
        )

        # Early stopping state
        self.best_val_ha: float = -1.0
        self.epochs_without_improvement: int = 0
        self.best_epoch: int = 0

        # Paths
        config.models_dir.mkdir(parents=True, exist_ok=True)
        config.logs_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = config.models_dir / f"{experiment_name}_best.pt"
        self.log_path = config.logs_dir / f"{experiment_name}_training_log.csv"

        # Initialise CSV log
        self._init_csv_log()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def train(self) -> dict[str, float]:
        """
        Runs the full training loop until max_epochs or early stopping.

        Returns:
            Validation metrics of the best checkpoint.
        """
        logger.info(
            f"Starting training: {self.experiment_name} | "
            f"device={self.device} | AMP={self.use_amp}"
        )

        best_val_metrics: dict[str, float] = {}

        for epoch in range(1, self.config.max_epochs + 1):
            epoch_start = time.time()

            train_loss = self._train_epoch(epoch)
            val_loss, val_metrics = self._validate_epoch(epoch)
            val_ha = val_metrics["hierarchical_accuracy"]

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.config.max_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_ha={val_ha:.4f} | "
                f"time={elapsed:.1f}s"
            )

            self._log_to_csv(epoch, train_loss, val_loss, val_metrics)

            # Checkpoint if best so far
            if val_ha > self.best_val_ha:
                self.best_val_ha = val_ha
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                best_val_metrics = val_metrics
                self._save_checkpoint()
                logger.info(
                    f"  New best hierarchical_accuracy: {val_ha:.4f} "
                    f"— checkpoint saved."
                )
            else:
                self.epochs_without_improvement += 1
                logger.info(
                    f"  No improvement for "
                    f"{self.epochs_without_improvement}/"
                    f"{self.config.early_stopping_patience} epochs."
                )

            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best epoch: {self.best_epoch}."
                )
                break

        logger.info(
            f"Training complete. Best epoch: {self.best_epoch} | "
            f"Best val HA: {self.best_val_ha:.4f}"
        )
        return best_val_metrics

    def evaluate(self, data_loader: DataLoader) -> tuple[float, dict[str, float]]:
        """
        Evaluates the model on an arbitrary DataLoader.

        Intended for final test-set evaluation after loading the best checkpoint.

        Args:
            data_loader: DataLoader to evaluate on (typically test_loader).

        Returns:
            Tuple of (mean_loss, metrics_dict).
        """
        return self._run_eval(data_loader)

    def load_best_checkpoint(self) -> None:
        """Loads the best saved checkpoint into the model."""
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        logger.info(f"Loaded best checkpoint from {self.checkpoint_path}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, epoch: int) -> float:
        """Runs one full training epoch and returns mean loss."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.max_epochs} [Train]",
            leave=True,
            dynamic_ncols=True  # Adjusts to your terminal width
        )

        for batch in progress_bar:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["labels"].to(self.device)

            self.optimiser.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                output = self.model(input_ids, attention_mask, labels=labels)

                # Use model-internal loss if available (e.g. LCPN),
                # otherwise compute centrally
                if "loss" in output:
                    loss = output["loss"]
                else:
                    loss = compute_loss(output["logits"], labels, self.config)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimiser)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_grad_norm
            )
            self.scaler.step(self.optimiser)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"train_loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def _validate_epoch(
        self, epoch: int
    ) -> tuple[float, dict[str, float]]:
        """Runs validation and returns (mean_loss, metrics_dict)."""
        return self._run_eval(self.val_loader)

    def _run_eval(
        self, data_loader: DataLoader
    ) -> tuple[float, dict[str, float]]:
        """
        Shared evaluation logic for validation and test splits.

        Collects all predictions and labels, then computes the full
        metrics suite in one pass after all batches are processed.

        Returns:
            Tuple of (mean_loss, metrics_dict).
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        progress_bar = tqdm(
            data_loader,
            desc=f"Validating",
            leave=True,
            dynamic_ncols=True  # Adjusts to your terminal width
        )

        with torch.no_grad():
            for batch in progress_bar:
                input_ids      = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels         = batch["labels"].to(self.device)

                with autocast('cuda', enabled=self.use_amp):
                    output = self.model(input_ids, attention_mask, labels=labels)

                    if "loss" in output:
                        loss = output["loss"]
                    else:
                        loss = compute_loss(output["logits"], labels, self.config)

                total_loss += loss.item()
                progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})

                # Use predict() for taxonomy-consistent predictions
                predictions = self.model.predict(input_ids, attention_mask)
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        predictions_np = np.concatenate(all_predictions, axis=0)
        labels_np      = np.concatenate(all_labels, axis=0)
        metrics        = compute_all_metrics(predictions_np, labels_np)

        return total_loss / len(data_loader), metrics

    def _save_checkpoint(self) -> None:
        """Saves the current model state dict to the checkpoint path."""
        torch.save(self.model.state_dict(), self.checkpoint_path)

    def _init_csv_log(self) -> None:
        """Creates the CSV log file and writes the header row."""
        fieldnames = self._get_csv_fieldnames()
        with open(self.log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def _log_to_csv(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metrics: dict[str, float],
    ) -> None:
        """Appends one row to the CSV log for the current epoch."""
        fieldnames = self._get_csv_fieldnames()
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        row.update({f"val_{k}": v for k, v in val_metrics.items()})

        with open(self.log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    def _get_csv_fieldnames(self) -> list[str]:
        """Returns the CSV column names based on the config."""
        num_levels = self.config.num_levels
        fields = ["epoch", "train_loss", "val_loss"]

        for level in range(1, num_levels + 1):
            fields += [f"val_acc_l{level}", f"val_f1_l{level}"]

        fields += ["val_hierarchical_accuracy", "val_hP", "val_hR", "val_hF"]

        for level in range(2, num_levels + 1):
            fields.append(f"val_cascade_l{level}")

        return fields
