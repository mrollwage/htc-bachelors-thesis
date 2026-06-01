from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass
class BaseConfig:
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "data")
    models_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "models_saved")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent / "logs")

    backbone_name: str = "distilbert-base-uncased"
    max_seq_length: int = 128
    dropout: float = 0.1

    # LR adjustement based on the freezing state
    #   n=-1 => LR=5e-4
    #   n=4  => LR=2e-4
    #   n=0  => LR=2e-5

    learning_rate: float = 8e-5
    batch_size: int = 128
    max_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    random_seed: int = 42
    # Seeds for n = 4 multi-seed experiment: s = {123, 456, 789}

    freeze_encoder_layers: int = 0
    # 0 = no freezing (train on full ~67M parameters)
    # -1 = freeze all encoder layers
    # n = freeze first n transformer layers

    num_levels: int = 0
    classes_per_level: list[int] = field(default_factory=list)
    level_weights: list[float] = field(default_factory=lambda: [])

    primary_metric: str = "hierarchical_accuracy"

    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    num_workers: int = 0
    log_every_n_steps: int = 50

    access_token = "HF_TOKEN" # here would be the corresponding HF access token; for security reasons, a placeholder is placed here instead


@dataclass
class WOSConfig(BaseConfig):
    dataset_name: str = "wos"
    hf_dataset_id: str = "jesse-tong/wos46985"
    num_levels: int = 2
    text_column: str = "text"
    raw_label_column: str = "label_description"
    label_columns: list[str] = field(default_factory=lambda: ["l1", "l2"])


@dataclass
class DBpediaConfig(BaseConfig):
    dataset_name: str = "dbpedia"
    hf_dataset_id: str = "DeveloperOats/DBPedia_Classes"
    num_levels: int = 3
    text_column: str = "text"
    raw_label_column: str = ""
    label_columns: list[str] = field(default_factory=lambda: ["l1", "l2", "l3"])


@dataclass
class GoogleTaxonomyConfig(BaseConfig):
    dataset_name: str = "gpsd"
    # Trage hier entweder deine HF Repo-ID ein oder den Pfad zu deiner lokalen Datei (z.B. "data/google_taxonomy.csv")
    hf_dataset_id: str = "data/rollwage2026gpsd_47k.csv"
    num_levels: int = 5
    text_column: str = "text"  # Passe dies an, falls deine Textspalte anders heißt
    raw_label_column: str = ""
    label_columns: list[str] = field(default_factory=lambda: ["l1", "l2", "l3", "l4", "l5"])


def get_config(dataset: str) -> BaseConfig:
    configs = {
        "wos": WOSConfig,
        "dbpedia": DBpediaConfig,
        "gpsd": GoogleTaxonomyConfig,
    }
    if dataset not in configs:
        raise ValueError(
            f"unknown dataset: '{dataset}'. "
            f"available: {list(configs.keys())}"
        )
    return configs[dataset]()
