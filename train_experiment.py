import argparse
import logging
import random

import numpy as np
import torch
from transformers import AutoTokenizer

from src.config import BaseConfig, get_config
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
from src.utils.logger import save_experiment_config, save_test_results, setup_logging

logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "flat":         FlatClassifier,
    "global_multi": GlobalMultiHeadClassifier,
    "lcl":          LocalClassifierPerLevel,
    "lcpn":          LocalClassifierPerParentNode,
    "chaining":     ClassifierChainingModel,
}


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a hierarchical text classifier."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture to train.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wos", "dbpedia", "gpsd"],
        help="Dataset to train on.",
    )
    parser.add_argument(
        "--freeze_n_layers",
        type=int,
        required=True,
        choices=list(range(-1, 7)),
        help="Number of layers to freeze for training.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Optional experiment identifier for checkpoint and log filenames. "
            "Defaults to '{model_type}_{dataset}'."
        ),
    )
    # Allow overriding key hyperparameters from CLI for convenience
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--max_epochs",    type=int,   default=None)
    parser.add_argument("--warmup_steps",  type=int,   default=None)
    parser.add_argument("--weight_decay",  type=float, default=None)
    return parser.parse_args()


def apply_cli_overrides(config: BaseConfig, args: argparse.Namespace) -> None:
    """
    Applies optional CLI hyperparameter overrides to the config.

    Only overrides fields that were explicitly passed on the command line.

    Args:
        config: Config to mutate in place.
        args: Parsed CLI arguments.
    """
    overrides = {
        "learning_rate": args.learning_rate,
        "batch_size":    args.batch_size,
        "max_epochs":    args.max_epochs,
        "warmup_steps":  args.warmup_steps,
        "weight_decay":  args.weight_decay,
    }
    for field, value in overrides.items():
        if value is not None:
            setattr(config, field, value)
            logger.info(f"CLI override: {field}={value}")


def set_seed(seed: int) -> None:
    """
    Sets random seeds for Python, NumPy and PyTorch for reproducibility.

    Args:
        seed: Integer seed value from config.random_seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    config: BaseConfig,
    hierarchy,
    freeze_n_layers: int
) -> torch.nn.Module:
    """
    Instantiates the requested model class.

    Args:
        model_type: Key into MODEL_REGISTRY.
        config: Fully initialized config (classes_per_level must be set).
        hierarchy: Fully built HierarchyManager.
        freeze_n_layers: Define n number of layers to freeze for training.

    Returns:
        Instantiated model.
    """
    model_class = MODEL_REGISTRY[model_type]
    return model_class(config, config.access_token, hierarchy, freeze_n_layers)


def main() -> None:
    setup_logging()
    args = parse_args()

    experiment_name = args.experiment_name or f"{args.model_type}_{args.dataset}"
    logger.info(f"=== Experiment: {experiment_name} ===")

    # --- Config ---
    config = get_config(args.dataset)
    apply_cli_overrides(config, args)
    set_seed(config.random_seed)
    logger.info(f"Device: {config.device}")

    # --- Data ---
    dataset = load_and_preprocess(config, config.access_token)

    hierarchy = build_hierarchy_from_split(dataset["train"], config)
    config.level_weights = [1.0] * config.num_levels

    tokenizer = AutoTokenizer.from_pretrained(
        config.backbone_name,
        token=config.access_token,
    )

    train_ds = HierarchicalDataset(dataset["train"], config, hierarchy, tokenizer)
    val_ds   = HierarchicalDataset(dataset["validation"], config, hierarchy, tokenizer)
    test_ds  = HierarchicalDataset(dataset["test"], config, hierarchy, tokenizer)

    train_loader, val_loader, test_loader = create_dataloaders(
        train_ds, val_ds, test_ds, config
    )

    # --- Model ---
    model = build_model(args.model_type, config, hierarchy, args.freeze_n_layers)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model_type} | Trainable params: {n_params:,}")

    # --- Save config before training starts ---
    save_experiment_config(config, experiment_name, args.model_type)

    # --- Train ---
    trainer = Trainer(
        model=model,
        config=config,
        hierarchy=hierarchy,
        train_loader=train_loader,
        val_loader=val_loader,
        experiment_name=experiment_name,
    )
    trainer.train()

    # --- Evaluate on test set using best checkpoint ---
    logger.info("Evaluating best checkpoint on test set...")
    trainer.load_best_checkpoint()
    test_loss, test_metrics = trainer.evaluate(test_loader)

    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info(f"Test hierarchical_accuracy: "
                f"{test_metrics['hierarchical_accuracy']:.4f}")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    save_test_results(test_metrics, experiment_name, config)
    logger.info(f"=== Experiment {experiment_name} complete ===")


if __name__ == "__main__":
    main()
