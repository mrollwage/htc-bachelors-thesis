import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager
from src.models.backbone import HierarchicalClassifier


class LocalClassifierPerLevel(HierarchicalClassifier):
    """
    Local Classifier per Level (LCL) for hierarchical text classification.

    Trains one independent classification head per hierarchy level, each with
    its own dropout layer. All heads share the same DistilBERT encoder, in line
    with the controlled experimental setup of this thesis (identical backbone
    across all approaches).

    The key distinction from GlobalMultiHeadClassifier is conceptual: each head
    is treated as an independent local classifier for its level, without any
    explicit or implicit conditioning on other levels during training. During
    inference, hierarchical masking is applied to ensure taxonomy-consistent
    predictions, which is the only point where hierarchy knowledge is introduced.

    This mirrors the standard LCL formulation in the literature, adapted to use
    a shared encoder for fair comparison under identical parameter budgets.

    Args:
        config: Dataset-specific config.
        hierarchy: Fully built HierarchyManager for masking during inference.
    """

    def __init__(self, config: BaseConfig, token: str, hierarchy: HierarchyManager, freeze_n_layers: int):
        super().__init__(config, token, freeze_n_layers)
        self.hierarchy = hierarchy

        # One independent dropout + classification head per level
        # Separate dropouts reinforce the local, level-independent training
        # philosophy: each head receives its own regularised representation
        self.level_dropouts = nn.ModuleList([
            nn.Dropout(config.dropout)
            for _ in range(config.num_levels)
        ])
        self.heads = nn.ModuleList([
            self._build_classification_head(num_classes)
            for num_classes in config.classes_per_level
        ])

        # Uniform loss weights across all levels
        level_weights = torch.ones(config.num_levels)
        self.register_buffer("level_weights", level_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass treating each level as an independent classification problem.

        The encoder runs once to produce a shared CLS representation. Each level
        then applies its own dropout independently before its classification head,
        reinforcing level-local behaviour during training.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).
            labels: Ground truth indices, shape (batch_size, num_levels).
                    If provided, loss is computed as a sum of independent
                    per-level cross-entropy losses.

        Returns:
            A dict with:
                - 'logits': list of tensors, one per level,
                             each shape (batch_size, classes_per_level[i])
                - 'loss': weighted sum of independent per-level losses
                           (only if labels are provided)
        """
        # Run encoder once — shared representation for all levels
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]

        # Each level applies its own dropout to the shared CLS representation,
        # treating itself as an independent local classifier
        logits = [
            head(dropout(cls_hidden))
            for dropout, head in zip(self.level_dropouts, self.heads)
        ]

        result = {"logits": logits}

        if labels is not None:
            result["loss"] = self._compute_loss(logits, labels)

        return result

    def _compute_loss(
        self,
        logits: list[torch.Tensor],
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the weighted sum of independent per-level cross-entropy losses.

        Each level is treated as a standalone classification problem — there is
        no conditioning on other levels during loss computation.

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
        Runs sequential inference with hierarchical masking.

        Although each level was trained independently, hierarchical masking is
        applied at inference time to ensure taxonomy-consistent predictions.
        This is architecturally meaningful for LCL: the hierarchy knowledge
        introduced here is the only point where inter-level dependencies are
        enforced, making its effect directly measurable.

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

        # Level 0: free prediction
        predictions[:, 0] = logits[0].argmax(dim=-1)

        # Levels 1+: mask invalid children based on parent prediction
        for level in range(1, self.num_levels):
            level_logits = logits[level].clone()

            for sample_idx in range(batch_size):
                parent_idx = predictions[sample_idx, level - 1].item()
                valid_children = self.hierarchy.get_valid_children(
                    level=level - 1,
                    parent_idx=parent_idx,
                )
                mask = torch.full(
                    (self.classes_per_level[level],),
                    float("-inf"),
                    device=input_ids.device,
                )
                mask[valid_children] = 0.0
                level_logits[sample_idx] += mask

            predictions[:, level] = level_logits.argmax(dim=-1)

        return predictions
