import logging
from torch.utils.data import DataLoader

from src.config import BaseConfig
from src.data.dataset import HierarchicalDataset

logger = logging.getLogger(__name__)


def create_dataloaders(
    train_dataset: HierarchicalDataset,
    val_dataset: HierarchicalDataset,
    test_dataset: HierarchicalDataset,
    config: BaseConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation and test splits.

    The train DataLoader shuffles data each epoch. Validation and test
    DataLoaders are deterministic (no shuffle) for reproducible evaluation.

    Args:
        train_dataset: HierarchicalDataset for the training split.
        val_dataset: HierarchicalDataset for the validation split.
        test_dataset: HierarchicalDataset for the test split.
        config: Config carrying batch_size, num_workers and random_seed.

    Returns:
        A tuple of (train_loader, val_loader, test_loader).
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.device == "cuda",
    )

    logger.info(
        f"DataLoaders created | "
        f"train: {len(train_loader)} batches | "
        f"val: {len(val_loader)} batches | "
        f"test: {len(test_loader)} batches"
    )

    return train_loader, val_loader, test_loader
