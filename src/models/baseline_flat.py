import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager
from src.models.backbone import HierarchicalClassifier


class FlatClassifier(HierarchicalClassifier):
    """
    Flat baseline classifier for hierarchical text classification.

    Ignores the hierarchical structure entirely and treats each leaf node
    as an independent class. A single classification head maps the CLS token
    representation directly to the leaf-level classes.

    During inference, the predicted leaf index is mapped back to a full
    hierarchy path using a precomputed lookup table derived from the
    HierarchyManager. This allows fair comparison with hierarchical approaches
    using the same evaluation metrics.

    Args:
        config: Dataset-specific config.
        hierarchy: Fully built HierarchyManager used for path reconstruction.
    """

    def __init__(self, config: BaseConfig, token: str, hierarchy: HierarchyManager, freeze_n_layers: int):
        super().__init__(config, token, freeze_n_layers)
        self.hierarchy = hierarchy

        # Single head over leaf-level classes only
        self.leaf_level = config.num_levels - 1
        num_leaf_classes = config.classes_per_level[self.leaf_level]
        self.head = self._build_classification_head(num_leaf_classes)

        # Precompute leaf_idx -> full path tensor for inference
        # Shape: (num_leaf_classes, num_levels)
        self.register_buffer(
            "leaf_to_path",
            self._build_leaf_to_path_table(),
        )

    def _build_leaf_to_path_table(self) -> torch.Tensor:
        """
        Builds a lookup table mapping each leaf class index to its full path.

        Iterates over all known label sequences in the hierarchy to reconstruct
        the full path (indices at all levels) for each leaf node.

        Returns:
            A tensor of shape (num_leaf_classes, num_levels) where each row
            contains the label indices for all levels of that leaf's path.
        """
        num_leaf_classes = self.classes_per_level[self.leaf_level]
        table = torch.zeros(num_leaf_classes, self.num_levels, dtype=torch.long)

        # Reconstruct paths from valid_children structure
        # We traverse the hierarchy top-down to find all valid paths
        def traverse(level: int, path: list[int]) -> None:
            if level == self.num_levels:
                leaf_idx = path[-1]
                table[leaf_idx] = torch.tensor(path, dtype=torch.long)
                return
            if level == 0:
                for parent_idx in range(self.classes_per_level[0]):
                    traverse(level + 1, [parent_idx])
            else:
                parent_idx = path[-1]
                children = self.hierarchy.get_valid_children(
                    level=level - 1,
                    parent_idx=parent_idx,
                )
                for child_idx in children:
                    traverse(level + 1, path + [child_idx])

        traverse(0, [])
        return table

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the encoder and single flat classification head.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).
            labels: Ground truth indices, shape (batch_size, num_levels).
                    Only the leaf-level labels (last column) are used for loss.

        Returns:
            A dict with:
                - 'logits': list containing a single tensor of shape
                             (batch_size, num_leaf_classes)
                - 'loss': cross-entropy on leaf labels (if labels given)
        """
        cls_output = self.get_cls_output(input_ids, attention_mask)
        logits = self.head(cls_output)

        result = {"logits": [logits]}

        if labels is not None:
            leaf_labels = labels[:, self.leaf_level]
            loss = F.cross_entropy(logits, leaf_labels)
            result["loss"] = loss

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predicts the leaf class and reconstructs the full hierarchy path.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            Predicted label indices, shape (batch_size, num_levels).
            Parent levels are inferred from the predicted leaf via lookup table.
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)["logits"][0]

        leaf_predictions = logits.argmax(dim=-1)

        # Map each leaf prediction to its full path using the lookup table
        return self.leaf_to_path[leaf_predictions]
