import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager
from src.models.backbone import HierarchicalClassifier


class GlobalMultiHeadClassifier(HierarchicalClassifier):
    """
    Global End-to-End Multi-Head classifier for hierarchical text classification.

    Uses a single shared DistilBERT encoder with one linear classification head
    per hierarchy level. All heads operate in parallel on the same CLS token
    representation. During training, a weighted sum of per-level cross-entropy
    losses is optimised. During inference, hierarchical masking ensures that
    only taxonomy-valid paths are predicted.

    This approach models inter-level dependencies implicitly through the shared
    encoder, without explicit conditioning between levels.

    Args:
        config: Dataset-specific config.
        hierarchy: Fully built HierarchyManager for masking during inference.
    """

    def __init__(self, config: BaseConfig, token: str, hierarchy: HierarchyManager, freeze_n_layers: int):
        super().__init__(config, token, freeze_n_layers)
        self.hierarchy = hierarchy

        # One linear classification head per hierarchy level
        self.heads = nn.ModuleList([
            self._build_classification_head(num_classes)
            for num_classes in config.classes_per_level
        ])

        # Uniform loss weights across all levels
        # Stored as a buffer so they move to the correct device with .to(device)
        level_weights = torch.ones(config.num_levels)
        self.register_buffer("level_weights", level_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the shared encoder and all classification heads.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).
            labels: Ground truth indices, shape (batch_size, num_levels).
                    If provided, loss is computed and included in the output.

        Returns:
            A dict with:
                - 'logits': list of tensors, one per level,
                             each shape (batch_size, classes_per_level[i])
                - 'loss': weighted sum of per-level cross-entropy (if labels given)
        """
        cls_output = self.get_cls_output(input_ids, attention_mask)

        # All heads run in parallel on the same CLS representation
        logits = [head(cls_output) for head in self.heads]

        result = {"logits": logits}

        if labels is not None:
            loss = self._compute_loss(logits, labels)
            result["loss"] = loss

        return result

    def _compute_loss(
        self,
        logits: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the weighted sum of cross-entropy losses across all levels.

        Args:
            logits: List of logit tensors, one per level.
            labels: Ground truth indices, shape (batch_size, num_levels).

        Returns:
            Scalar loss tensor.
        """
        total_loss = torch.tensor(0.0, device=labels.device)

        for level, (level_logits, weight) in enumerate(
            zip(logits, self.level_weights)
        ):
            level_labels = labels[:, level]
            level_loss = F.cross_entropy(level_logits, level_labels)
            total_loss = total_loss + weight * level_loss

        return total_loss

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs inference with hierarchical masking.

        For each level beyond the first, only classes that are valid children
        of the predicted parent class are considered. Invalid classes are masked
        to -inf before argmax, ensuring taxonomy-consistent predictions.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            Predicted label indices, shape (batch_size, num_levels).
        """
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)["logits"]

        batch_size = input_ids.shape[0]
        predictions = torch.zeros(
            batch_size, self.num_levels, dtype=torch.long, device=input_ids.device
        )

        # Level 0: free prediction, no masking needed
        predictions[:, 0] = logits[0].argmax(dim=-1)

        # Levels 1+: mask out invalid children based on parent prediction
        for level in range(1, self.num_levels):
            level_logits = logits[level].clone()

            for sample_idx in range(batch_size):
                parent_idx = predictions[sample_idx, level - 1].item()
                valid_children = self.hierarchy.get_valid_children(
                    level=level - 1,
                    parent_idx=parent_idx,
                )

                # Mask all classes to -inf, then unmask valid children
                mask = torch.full(
                    (self.classes_per_level[level],),
                    float("-inf"),
                    device=input_ids.device,
                )
                mask[valid_children] = 0.0
                level_logits[sample_idx] += mask

            predictions[:, level] = level_logits.argmax(dim=-1)

        return predictions
