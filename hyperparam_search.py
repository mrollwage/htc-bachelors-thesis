import argparse
import logging
import random

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from transformers import AutoTokenizer

from src.config import get_config
from src.data.dataset import (
    build_hierarchy_from_split,
    load_and_preprocess,
    HierarchicalDataset,
)
from src.data.dataloader import create_dataloaders
from src.models.baseline_flat import FlatClassifier
from src.models.global_chaining import ClassifierChainingModel
from src.models.global_multi import GlobalMultiHeadClassifier
from src.models.local_lcl import LocalClassifierPerLevel
from src.models.local_lcpn import LocalClassifierPerParentNode
from src.training.trainer import Trainer
from src.utils.logger import save_experiment_config, setup_logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "flat": FlatClassifier,
    "global_multi": GlobalMultiHeadClassifier,
    "lcl": LocalClassifierPerLevel,
    "lcpn": LocalClassifierPerParentNode,
    "chaining": ClassifierChainingModel,
}

N_TRIALS = 10
N_STARTUP_TRIALS = 3  # Trials before pruning is active


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for hierarchical text classifiers."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture to optimise.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wos", "dbpedia", "gpsd"],
        help="Dataset to optimise on.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def objective(
        trial: optuna.Trial,
        model_type: str,
        base_experiment_name: str,
        dataset_dict,
        config,
        hierarchy,
        tokenizer,
) -> float:
    """
    Optuna objective function for one hyperparameter trial.

    Samples a hyperparameter configuration, trains the model for up to
    max_epochs with early stopping, and reports the best validation
    hierarchical accuracy to Optuna after each epoch for pruning.

    Args:
        trial: Optuna trial object used for hyperparameter sampling.
        model_type: Architecture key into MODEL_REGISTRY.
        base_experiment_name: Prefix for checkpoint and log filenames.
        dataset_dict: Preprocessed HuggingFace DatasetDict.
        config: Dataset-specific config (will be mutated per trial).
        hierarchy: Fully built HierarchyManager.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Best validation hierarchical accuracy achieved in this trial.
    """
    # --- Sample hyperparameters ---
    config.learning_rate = trial.suggest_float(
        "learning_rate", 1e-5, 1e-4, log=True
    )
    config.batch_size = trial.suggest_categorical(
        "batch_size", [32, 48, 64, 96, 128]
    )
    config.warmup_steps = trial.suggest_categorical(
        "warmup_steps", [500, 1000, 2000]
    )
    config.weight_decay = trial.suggest_float(
        "weight_decay", 0.0, 0.1
    )

    logger.info(
        f"Trial {trial.number} | "
        f"lr={config.learning_rate:.2e} | "
        f"bs={config.batch_size} | "
        f"warmup={config.warmup_steps} | "
        f"wd={config.weight_decay:.4f}"
    )

    set_seed(config.random_seed)

    # --- DataLoaders (batch_size may differ per trial) ---
    train_ds = HierarchicalDataset(
        dataset_dict["train"], config, hierarchy, tokenizer
    )
    val_ds = HierarchicalDataset(
        dataset_dict["validation"], config, hierarchy, tokenizer
    )
    test_ds = HierarchicalDataset(
        dataset_dict["test"], config, hierarchy, tokenizer
    )
    train_loader, val_loader, _ = create_dataloaders(
        train_ds, val_ds, test_ds, config
    )

    # --- Model ---
    model = MODEL_REGISTRY[model_type](config, config.access_token, hierarchy, 0)

    experiment_name = f"{base_experiment_name}_trial{trial.number}"
    trainer = Trainer(
        model=model,
        config=config,
        hierarchy=hierarchy,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
    )

    # --- Training loop with per-epoch Optuna reporting for pruning ---
    best_val_ha = 0.0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = trainer._train_epoch(epoch)
        val_loss, val_metrics = trainer._validate_epoch(epoch)
        val_ha = val_metrics["hierarchical_accuracy"]

        logger.info(
            f"  Epoch {epoch} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_ha={val_ha:.4f}"
        )

        # Log CSV and handle checkpointing via trainer internals
        trainer._log_to_csv(epoch, train_loss, val_loss, val_metrics)

        if val_ha > best_val_ha:
            best_val_ha = val_ha
            trainer.best_val_ha = val_ha
            trainer.best_epoch = epoch
            trainer.epochs_without_improvement = 0
            trainer._save_checkpoint()
        else:
            trainer.epochs_without_improvement += 1

        # Report to Optuna for pruning
        trial.report(val_ha, epoch)
        if trial.should_prune():
            logger.info(f"  Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.TrialPruned()

        # Early stopping
        if trainer.epochs_without_improvement >= config.early_stopping_patience:
            logger.info(
                f"  Early stopping at epoch {epoch}. "
                f"Best val HA: {best_val_ha:.4f}"
            )
            break

    return best_val_ha


def main() -> None:
    setup_logging()
    args = parse_args()

    base_experiment_name = f"optuna_{args.model_type}_{args.dataset}"
    logger.info(f"=== Hyperparameter Search: {base_experiment_name} ===")
    logger.info(f"Trials: {N_TRIALS} | Startup trials: {N_STARTUP_TRIALS}")

    # --- Data (loaded once, shared across all trials) ---
    config = get_config(args.dataset)
    dataset = load_and_preprocess(config, config.access_token)

    if args.dataset == "dbpedia":
        fraction = 0.25  # 25% der Daten
        logger.info(f"Applying {fraction * 100}% sub-sampling for DBpedia to reduce runtime.")

        train_size = int(fraction * len(dataset["train"]))
        val_size = int(fraction * len(dataset["validation"]))

        # Deterministic Shuffling using config.random_seed
        dataset["train"] = dataset["train"].shuffle(seed=config.random_seed).select(range(train_size))
        dataset["validation"] = dataset["validation"].shuffle(seed=config.random_seed).select(range(val_size))

        logger.info(f"New DBpedia sizes - Train: {train_size}, Val: {val_size}")

    hierarchy = build_hierarchy_from_split(dataset["train"], config)
    config.level_weights = [1.0] * config.num_levels

    tokenizer = AutoTokenizer.from_pretrained(
        config.backbone_name,
        token=config.access_token,
    )

    # --- Optuna study ---
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=config.random_seed)
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=3,  # Don't prune before epoch 3
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=base_experiment_name,
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            model_type=args.model_type,
            base_experiment_name=base_experiment_name,
            dataset_dict=dataset,
            config=config,
            hierarchy=hierarchy,
            tokenizer=tokenizer,
        ),
        n_trials=N_TRIALS,
    )

    # --- Report best trial ---
    best = study.best_trial
    logger.info(f"=== Best Trial: {best.number} ===")
    logger.info(f"Best val hierarchical_accuracy: {best.value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in best.params.items():
        logger.info(f"  {key}: {value}")

    # --- Save best hyperparameters to config and log ---
    config.learning_rate = best.params["learning_rate"]
    config.batch_size = best.params["batch_size"]
    config.warmup_steps = best.params["warmup_steps"]
    config.weight_decay = best.params["weight_decay"]
    save_experiment_config(config, f"{base_experiment_name}_best", args.model_type)

    # --- Save Optuna summary as JSON ---
    import json
    summary = {
        "study_name": base_experiment_name,
        "best_trial": best.number,
        "best_val_ha": best.value,
        "best_params": best.params,
        "all_trials": [
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": str(t.state),
            }
            for t in study.trials
        ],
    }
    summary_path = config.logs_dir / f"{base_experiment_name}_summary.json"
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Optuna summary saved: {summary_path}")

    logger.info(f"=== Search complete: {base_experiment_name} ===")


if __name__ == "__main__":
    main()
