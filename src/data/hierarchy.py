import json
from pathlib import Path
from collections import defaultdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class HierarchyManager:
    def __init__(self, num_levels: int, dataset_name: str, cache_dir: Path):
        self.num_levels = num_levels
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.cache_path = cache_dir / f"{dataset_name}_hierarchy.json"

        # Label Encoding: one mapping-dict per level
        # label_to_idx[level] = {"CS": 0, "Medical": 1, ...}
        self.label_to_idx: list[dict[str, int]] = [dict() for _ in range(num_levels)]
        self.idx_to_label: list[dict[int, str]] = [dict() for _ in range(num_levels)]

        # Hierarchy structure: valid_children[level][parent_idx] = set of valid child_idxs
        # level=0 means: valid children from l1-classes on l2-level
        self.valid_children: list[dict[int, set[int]]] = [
            defaultdict(set) for _ in range(num_levels - 1)
        ]

        self._is_built = False

    @property
    def classes_per_level(self) -> list[int]:
        return [len(m) for m in self.label_to_idx]

    def build_from_data(self, label_sequences: list[list[str]]) -> None:
        logger.info(f"build hierarchy for '{self.dataset_name}' from {len(label_sequences)} samples...")

        # Step 1: Collect all unique labels per level
        labels_per_level: list[set[str]] = [set() for _ in range(self.num_levels)]
        for sequence in label_sequences:
            if len(sequence) != self.num_levels:
                raise ValueError(
                    f"label-sequence has length {len(sequence)}, "
                    f"expected {self.num_levels}: {sequence}"
                )
            for level, label in enumerate(sequence):
                labels_per_level[level].add(label)

        # Step 2: Build sorted encoding (sorted for reproducibility)
        for level, label_set in enumerate(labels_per_level):
            sorted_labels = sorted(label_set)
            self.label_to_idx[level] = {label: idx for idx, label in enumerate(sorted_labels)}
            self.idx_to_label[level] = {idx: label for idx, label in enumerate(sorted_labels)}

        # Step 3: Build valid parent-child relations
        for sequence in label_sequences:
            for level in range(self.num_levels - 1):
                parent_label = sequence[level]
                child_label = sequence[level + 1]
                parent_idx = self.label_to_idx[level][parent_label]
                child_idx = self.label_to_idx[level + 1][child_label]
                self.valid_children[level][parent_idx].add(child_idx)

        self._is_built = True
        logger.info(f"hierarchy built: {self.classes_per_level} classes per level.")

    def get_valid_children(self, level: int, parent_idx: int) -> list[int]:
        if level >= self.num_levels - 1:
            raise ValueError(f"level {level} has no children (max: {self.num_levels - 2}).")
        return list(self.valid_children[level][parent_idx])

    def encode_labels(self, label_sequence: list[str]) -> list[int]:
        return [
            self.label_to_idx[level][label]
            for level, label in enumerate(label_sequence)
        ]

    def decode_labels(self, idx_sequence: list[int]) -> list[str]:
        return [
            self.idx_to_label[level][idx]
            for level, idx in enumerate(idx_sequence)
        ]

    def save_to_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Sets müssen für JSON in Listen konvertiert werden
        serializable_children = [
            {str(k): list(v) for k, v in level_dict.items()}
            for level_dict in self.valid_children
        ]

        cache_data = {
            "dataset_name": self.dataset_name,
            "num_levels": self.num_levels,
            "label_to_idx": self.label_to_idx,
            "valid_children": serializable_children,
        }

        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)

        logger.info(f"saved hierarchy: {self.cache_path}")

    def load_from_cache(self) -> bool:
        if not self.cache_path.exists():
            return False

        with open(self.cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        if (cache_data["dataset_name"] != self.dataset_name or
                cache_data["num_levels"] != self.num_levels):
            logger.warning("cache file does not match current dataset, ignoring cache.")
            return False

        self.label_to_idx = cache_data["label_to_idx"]
        self.idx_to_label = [
            {int(k): v for k, v in level_dict.items()}
            for level_dict in [
                {str(idx): label for label, idx in level.items()}
                for level in self.label_to_idx
            ]
        ]

        self.valid_children = [
            defaultdict(set, {int(k): set(v) for k, v in level_dict.items()})
            for level_dict in cache_data["valid_children"]
        ]

        self._is_built = True
        logger.info(f"loaded hierarchy in cache: {self.cache_path} "
                    f"| classes: {self.classes_per_level}")
        return True

    @classmethod
    def build_or_load(
        cls,
        num_levels: int,
        dataset_name: str,
        cache_dir: Path,
        label_sequences: Optional[list[list[str]]] = None,
        force_rebuild: bool = False,
    ) -> "HierarchyManager":
        manager = cls(num_levels, dataset_name, cache_dir)

        if not force_rebuild and manager.load_from_cache():
            return manager

        if label_sequences is None:
            raise ValueError(
                "label_sequences must not be empty, "
                "if no data is cached yet."
            )

        manager.build_from_data(label_sequences)
        manager.save_to_cache()
        return manager
