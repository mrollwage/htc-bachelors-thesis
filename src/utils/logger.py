import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.config import BaseConfig


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Configures root logger to write to stdout with a consistent format.

    Should be called once at the start of train_experiment.py before
    any other imports that use logging.

    Args:
        log_level: Logging level, e.g. logging.INFO or logging.DEBUG.
    """
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def save_experiment_config(
    config: BaseConfig,
    experiment_name: str,
    model_type: str,
) -> None:
    """
    Saves the full experiment configuration as a JSON file.

    Useful for reproducing experiments and tracking which hyperparameters
    were used for a given run.

    Args:
        config: The dataset-specific config used for the experiment.
        experiment_name: Identifier matching the trainer's experiment name.
        model_type: Model type string, e.g. 'global_multi' or 'lcl'.
    """
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.logs_dir / f"{experiment_name}_config.json"

    # Serialise config fields — convert Path objects to strings
    config_dict = {
        "experiment_name": experiment_name,
        "model_type": model_type,
        "timestamp": datetime.now().isoformat(),
        "dataset": config.dataset_name,
        "backbone": config.backbone_name,
        "num_levels": config.num_levels,
        "classes_per_level": config.classes_per_level,
        "level_weights": config.level_weights,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "weight_decay": config.weight_decay,
        "warmup_steps": config.warmup_steps,
        "max_grad_norm": config.max_grad_norm,
        "early_stopping_patience": config.early_stopping_patience,
        "max_seq_length": config.max_seq_length,
        "dropout": config.dropout,
        "random_seed": config.random_seed,
        "device": config.device,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    logging.getLogger(__name__).info(f"Config saved: {output_path}")


def save_test_results(
    metrics: dict[str, float],
    experiment_name: str,
    config: BaseConfig,
) -> None:
    """
    Saves final test-set evaluation results as a JSON file.

    Args:
        metrics: Dict of metric names to values, as returned by
                 compute_all_metrics().
        experiment_name: Identifier matching the trainer's experiment name.
        config: Config carrying the logs_dir path.
    """
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.logs_dir / f"{experiment_name}_test_results.json"

    results = {
        "experiment_name": experiment_name,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logging.getLogger(__name__).info(f"Test results saved: {output_path}")
