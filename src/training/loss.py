import torch
import torch.nn.functional as F

from src.config import BaseConfig


def compute_loss(
    logits: list[torch.Tensor],
    labels: torch.Tensor,
    config: BaseConfig,
) -> torch.Tensor:
    """
    Computes the weighted sum of per-level cross-entropy losses.

    Serves as a centralised loss function for all hierarchical classifiers
    that produce one logit tensor per level (FlatClassifier excluded, as it
    computes its own loss internally due to its single-head structure).

    Loss weights are uniform (1.0 per level) by default, as defined in
    BaseConfig.level_weights. This ensures that performance differences
    between architectures are attributable solely to the classification
    strategy rather than to loss weighting choices.

    Args:
        logits: List of logit tensors, one per level. Each tensor has shape
                (batch_size, classes_per_level[i]).
        labels: Ground truth label indices, shape (batch_size, num_levels).
        config: Config carrying level_weights (one float per level).

    Returns:
        Scalar loss tensor — weighted sum of per-level cross-entropy losses.
    """
    total_loss = torch.tensor(0.0, device=labels.device)

    for level, level_logits in enumerate(logits):
        weight = config.level_weights[level]
        level_labels = labels[:, level]
        level_loss = F.cross_entropy(level_logits, level_labels)
        total_loss = total_loss + weight * level_loss

    return total_loss
