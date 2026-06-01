import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager
from src.models.backbone import HierarchicalClassifier


class ClassifierChainingModel(HierarchicalClassifier):
    """
    Global classifier with Classifier Chaining for hierarchical text classification.

    Extends the Global Multi-Head approach by explicitly conditioning each
    classification head on the output of the previous level. The predicted
    probability vector of level n is concatenated to the shared CLS
    representation before being passed to the classification head of level n+1.

    Training uses Teacher Forcing: the ground-truth label of the previous level
    (as a one-hot vector) is used as conditioning input instead of the model's
    own prediction. This stabilises training and avoids compounding errors
    during gradient updates.

    Inference uses the model's own predictions sequentially (autoregressive),
    enabling direct measurement of cascading error rates — a key research
    question of this thesis.

    Args:
        config: Dataset-specific config.
        hierarchy: Fully built HierarchyManager for masking during inference.
    """

    def __init__(self, config: BaseConfig, token: str, hierarchy: HierarchyManager, freeze_n_layers: int):
        super().__init__(config, token, freeze_n_layers)
        self.hierarchy = hierarchy

        # Classification heads with chaining:
        # - Level 0 head: input is CLS output (hidden_size)
        # - Level n head (n > 0): input is CLS output + prob vector of level n-1
        #                          = hidden_size + classes_per_level[n-1]
        self.heads = nn.ModuleList()
        for level in range(config.num_levels):
            if level == 0:
                input_size = self.hidden_size
            else:
                input_size = self.hidden_size + config.classes_per_level[level - 1]
            self.heads.append(
                nn.Linear(input_size, config.classes_per_level[level])
            )

        # Uniform loss weights
        level_weights = torch.ones(config.num_levels)
        self.register_buffer("level_weights", level_weights)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with Teacher Forcing during training.

        When labels are provided (training), the ground-truth label of level n
        is converted to a one-hot vector and used as conditioning input for
        the head of level n+1. This is Teacher Forcing: the model sees the
        correct previous label regardless of its own prediction.

        When labels are not provided (eval without loss), the model's own
        softmax probabilities are used as conditioning — identical to inference.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).
            labels: Ground truth indices, shape (batch_size, num_levels).
                    If provided, Teacher Forcing is applied and loss is computed.

        Returns:
            A dict with:
                - 'logits': list of tensors, one per level,
                             each shape (batch_size, classes_per_level[i])
                - 'loss': weighted sum of per-level cross-entropy (if labels given)
        """
        cls_output = self.get_cls_output(input_ids, attention_mask)

        logits = []
        conditioning: torch.Tensor | None = None

        for level in range(self.num_levels):
            if conditioning is None:
                head_input = cls_output
            else:
                head_input = torch.cat([cls_output, conditioning], dim=-1)

            level_logits = self.heads[level](head_input)
            logits.append(level_logits)

            # Prepare conditioning vector for next level
            if level < self.num_levels - 1:
                if labels is not None:
                    # Teacher Forcing: use ground-truth one-hot as conditioning
                    conditioning = F.one_hot(
                        labels[:, level],
                        num_classes=self.classes_per_level[level],
                    ).float()
                else:
                    # Inference / eval without labels: use own softmax probs
                    conditioning = F.softmax(level_logits.detach(), dim=-1)

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
        Computes the weighted sum of per-level cross-entropy losses.

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
            level_loss = F.cross_entropy(level_logits, labels[:, level])
            total_loss = total_loss + weight * level_loss
        return total_loss

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Autoregressive inference with hierarchical masking.

        Uses the model's own predictions sequentially: the softmax output of
        level n conditions the head of level n+1. Hierarchical masking ensures
        taxonomy-consistent predictions, applied after conditioning.

        This autoregressive behaviour (as opposed to Teacher Forcing in training)
        enables direct measurement of cascading error rates: an error at level n
        propagates through the conditioning vector to level n+1.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            Predicted label indices, shape (batch_size, num_levels).
        """
        with torch.no_grad():
            cls_output = self.get_cls_output(input_ids, attention_mask)

        batch_size = input_ids.shape[0]
        predictions = torch.zeros(
            batch_size, self.num_levels, dtype=torch.long, device=input_ids.device
        )
        conditioning: torch.Tensor | None = None

        for level in range(self.num_levels):
            if conditioning is None:
                head_input = cls_output
            else:
                head_input = torch.cat([cls_output, conditioning], dim=-1)

            level_logits = self.heads[level](head_input).clone()

            # Apply hierarchical masking for levels beyond root
            if level > 0:
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

            # Conditioning for next level: softmax over masked logits
            if level < self.num_levels - 1:
                conditioning = F.softmax(level_logits, dim=-1)

        return predictions
