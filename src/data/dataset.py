import ast
import logging
from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
import torch
from torch.utils.data import Dataset

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager

logger = logging.getLogger(__name__)


def _parse_wos_labels(example: dict) -> dict:
    """
    Parses the WOS raw label_description field into separate l1 and l2 columns.

    The raw field contains a string representation of a list, e.g.:
        "['CS', 'Symbolic computation']"
    This is converted into two separate fields l1 and l2.

    Args:
        example: A single HuggingFace dataset row.

    Returns:
        The example dict with added 'l1' and 'l2' keys.
    """
    raw = example["label_description"]
    if isinstance(raw, str):
        parsed = ast.literal_eval(raw)
    else:
        parsed = raw
    example["l1"] = parsed[0].strip()
    example["l2"] = parsed[1].strip()
    return example


def _split_wos(dataset: DatasetDict, config: BaseConfig) -> DatasetDict:
    """
    Splits the WOS train-only dataset into train, validation and test sets.

    Uses a 70/10/20 ratio, consistent with the DBpedia split proportions.
    The split is seeded via config.random_seed for reproducibility.

    Args:
        dataset: Raw HuggingFace DatasetDict containing only a 'train' split.
        config: Dataset-specific config carrying the random seed.

    Returns:
        A DatasetDict with 'train', 'validation' and 'test' splits.
    """
    train_test = dataset["train"].train_test_split(
        test_size=0.2,
        seed=42,  # unaffected by random_seed change
    )
    train_val = train_test["train"].train_test_split(
        test_size=0.125,
        seed=42,  # unaffected by random_seed change
    )
    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_test["test"],
    })


def _split_google(dataset: DatasetDict, config: BaseConfig) -> DatasetDict:
    """
    Splits the Google Taxonomy dataset into train, validation and test sets.
    Uses a 70/10/20 ratio.
    """
    # Find the primary split key (usually 'train' if loaded from a single file)
    split_key = "train" if "train" in dataset else list(dataset.keys())[0]

    dataset_split = dataset[split_key].class_encode_column("l5")

    train_test = dataset_split.train_test_split(
        test_size=0.2,
        seed=42,  # hard-set to 42 and not to seed-variable from config to ensure constant splits for multi-seed experiments
        stratify_by_column="l5"
    )
    train_val = train_test["train"].train_test_split(
        test_size=0.125,
        seed=42,
        stratify_by_column="l5"
    )
    return DatasetDict({
        "train": train_val["train"],
        "validation": train_val["test"],
        "test": train_test["test"],
    })


def load_and_preprocess(config: BaseConfig, token: str) -> DatasetDict:
    """
    Loads a HuggingFace dataset and applies dataset-specific preprocessing.

    For WOS: parses label_description into l1/l2 columns and creates
             train/validation/test splits (70/10/20).
    For DBpedia: no preprocessing needed, predefined splits are used directly.

    After this function, both datasets share the same structure:
        - Splits: 'train', 'validation', 'test'
        - Columns: config.text_column + config.label_columns

    Args:
        config: Dataset-specific config (WOSConfig or DBpediaConfig).

    Returns:
        A HuggingFace DatasetDict with unified train/validation/test splits.
        :param token:
    """
    logger.info(f"Loading dataset '{config.hf_dataset_id}'...")

    if str(config.hf_dataset_id).endswith(".csv"):
        dataset = load_dataset("csv",
                               data_files=config.hf_dataset_id,
                               delimiter=';',
                               )
    elif str(config.hf_dataset_id).endswith(".json"):
        dataset = load_dataset("json", data_files=config.hf_dataset_id)
    else:
        dataset = load_dataset(config.hf_dataset_id, token=token)

    if config.dataset_name == "wos":
        logger.info("Preprocessing WOS: parsing label_description into l1, l2...")
        dataset = dataset.map(_parse_wos_labels)
        logger.info("Splitting WOS into train/validation/test (70/10/20)...")
        dataset = _split_wos(dataset, config)
    elif config.dataset_name == "gpsd":
        logger.info("Splitting Google Taxonomy into train/validation/test (70/10/20)...")
        dataset = _split_google(dataset, config)

    # Keep only the columns we need
    columns_to_keep = [config.text_column] + config.label_columns
    dataset = dataset.select_columns(columns_to_keep)

    # Log split sizes for verification
    for split_name, split_data in dataset.items():
        logger.info(f"  {split_name}: {len(split_data)} samples")

    return dataset


def build_hierarchy_from_split(
        dataset_split,
        config: BaseConfig,
) -> HierarchyManager:
    """
    Builds a HierarchyManager from a dataset split (typically the training split).

    Extracts all label sequences from the split and passes them to
    HierarchyManager.build_or_load(), using the config's data directory as cache.
    Also dynamically updates config.classes_per_level with the discovered counts.

    Args:
        dataset_split: A HuggingFace dataset split (e.g. dataset["train"]).
        config: Dataset-specific config, will be mutated to set classes_per_level.

    Returns:
        A fully initialised HierarchyManager.
    """
    label_sequences = [
        [example[col] for col in config.label_columns]
        for example in dataset_split
    ]

    hierarchy = HierarchyManager.build_or_load(
        num_levels=config.num_levels,
        dataset_name=config.dataset_name,
        cache_dir=config.data_dir,
        label_sequences=label_sequences,
    )

    config.classes_per_level = hierarchy.classes_per_level
    logger.info(f"Classes per level: {config.classes_per_level}")
    return hierarchy


class HierarchicalDataset(Dataset):
    """
    PyTorch Dataset for hierarchical text classification.

    Tokenizes input text and encodes hierarchical labels into integer indices.
    Compatible with both WOS (2 levels) and DBpedia (3 levels).

    Args:
        data: A HuggingFace dataset split.
        config: Dataset-specific config.
        hierarchy: A fully built HierarchyManager instance.
        tokenizer: A HuggingFace tokenizer matching the backbone model.
    """

    def __init__(
            self,
            data,
            config: BaseConfig,
            hierarchy: HierarchyManager,
            tokenizer: PreTrainedTokenizerBase,
    ):
        self.data = data
        self.config = config
        self.hierarchy = hierarchy
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a single tokenized and label-encoded sample.

        Returns:
            A dict with keys:
                - input_ids: Tensor of token ids, shape (max_seq_length,)
                - attention_mask: Tensor of attention mask, shape (max_seq_length,)
                - labels: Tensor of encoded label indices, shape (num_levels,)
        """
        example = self.data[idx]

        encoding = self.tokenizer(
            example[self.config.text_column],
            max_length=self.config.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        label_sequence = [example[col] for col in self.config.label_columns]
        encoded_labels = self.hierarchy.encode_labels(label_sequence)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(encoded_labels, dtype=torch.long),
        }
