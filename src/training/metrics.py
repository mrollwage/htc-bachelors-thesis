import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.data.hierarchy import HierarchyManager


def compute_accuracy_per_level(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> list[float]:
    """
    Computes accuracy separately for each hierarchy level.

    Args:
        predictions: Predicted label indices, shape (n_samples, num_levels).
        labels: Ground truth label indices, shape (n_samples, num_levels).

    Returns:
        List of accuracy scores, one per level.
    """
    num_levels = predictions.shape[1]
    return [
        accuracy_score(labels[:, level], predictions[:, level])
        for level in range(num_levels)
    ]


def compute_f1_per_level(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> list[float]:
    """
    Computes macro-averaged F1 score separately for each hierarchy level.

    Macro averaging treats all classes equally regardless of frequency,
    which is appropriate for the imbalanced class distributions in WOS
    and DBpedia.

    Args:
        predictions: Predicted label indices, shape (n_samples, num_levels).
        labels: Ground truth label indices, shape (n_samples, num_levels).

    Returns:
        List of macro F1 scores, one per level.
    """
    num_levels = predictions.shape[1]
    return [
        f1_score(
            labels[:, level],
            predictions[:, level],
            average="macro",
            zero_division=0,
        )
        for level in range(num_levels)
    ]


def compute_true_hierarchical_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Computes the True Hierarchical Accuracy (A).

    A prediction is counted as correct only if all hierarchy levels are
    predicted correctly. A single wrong level marks the entire sample
    as incorrect. This is the primary evaluation metric of this thesis.

    Args:
        predictions: Predicted label indices, shape (n_samples, num_levels).
        labels: Ground truth label indices, shape (n_samples, num_levels).

    Returns:
        Fraction of samples where all levels are predicted correctly.
    """
    # All levels must match simultaneously
    all_correct = np.all(predictions == labels, axis=1)
    return float(all_correct.mean())


def compute_hierarchical_precision_recall(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    """
    Computes Hierarchical Precision (hP), Recall (hR) and F1 (hF)
    following Kiritchenko et al. (2004).

    For each sample, the predicted and true ancestor sets are compared.
    In a tree-structured hierarchy with single labels per level, the
    ancestor set of a prediction is the full predicted path (all levels),
    since each level is an ancestor of the levels below it.

        hP = |predicted_path ∩ true_path| / |predicted_path|
        hR = |predicted_path ∩ true_path| / |true_path|
        hF = 2 * hP * hR / (hP + hR)

    Both paths have the same length (num_levels), so hP and hR only
    differ when predictions partially overlap with the true path.

    Args:
        predictions: Predicted label indices, shape (n_samples, num_levels).
        labels: Ground truth label indices, shape (n_samples, num_levels).

    Returns:
        Tuple of (hP, hR, hF) averaged across all samples.
    """
    n_samples = predictions.shape[0]
    hp_scores = np.zeros(n_samples)
    hr_scores = np.zeros(n_samples)

    for i in range(n_samples):
        # Represent each level as a (level, class_idx) tuple to avoid
        # collisions between identical indices at different levels
        pred_set = {(level, predictions[i, level])
                    for level in range(predictions.shape[1])}
        true_set = {(level, labels[i, level])
                    for level in range(labels.shape[1])}

        intersection = len(pred_set & true_set)
        hp_scores[i] = intersection / len(pred_set) if pred_set else 0.0
        hr_scores[i] = intersection / len(true_set) if true_set else 0.0

    hp = float(hp_scores.mean())
    hr = float(hr_scores.mean())
    hf = (2 * hp * hr / (hp + hr)) if (hp + hr) > 0 else 0.0

    return hp, hr, hf


def compute_cascading_error_rate(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> list[float]:
    """
    Computes the cascading error rate for each level beyond the root.

    The cascading error rate at level n is the fraction of errors at level n
    that are directly caused by an error at level n-1, i.e. cases where
    both level n-1 and level n are incorrect.

    Formally, for level n (n > 0):
        cascading_rate[n] = |errors at n AND errors at n-1| / |errors at n|

    A high cascading rate indicates that errors propagate downward through
    the hierarchy, which is a key quality dimension for hierarchical
    classifiers.

    Args:
        predictions: Predicted label indices, shape (n_samples, num_levels).
        labels: Ground truth label indices, shape (n_samples, num_levels).

    Returns:
        List of cascading error rates for levels 1..num_levels-1.
        (Level 0 is excluded as it has no parent to cascade from.)
        Returns 0.0 for a level with no errors.
    """
    num_levels = predictions.shape[1]
    cascading_rates = []

    for level in range(1, num_levels):
        errors_at_level = predictions[:, level] != labels[:, level]
        errors_at_parent = predictions[:, level - 1] != labels[:, level - 1]

        n_errors_at_level = errors_at_level.sum()
        if n_errors_at_level == 0:
            cascading_rates.append(0.0)
            continue

        # Errors at this level that are preceded by an error at the parent
        cascading = (errors_at_level & errors_at_parent).sum()
        cascading_rates.append(float(cascading / n_errors_at_level))

    return cascading_rates


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float | list[float]]:
    """
    Convenience wrapper that computes and returns all metrics as a flat dict.

    Intended for logging and result serialization. Keys follow the pattern:
        acc_l{n}, f1_l{n}       — per-level accuracy and macro F1
        hierarchical_accuracy   — True Hierarchical Accuracy (primary metric)
        hP, hR, hF              — hierarchical precision, recall, F1
        cascade_l{n}            — cascading error rate at level n

    Args:
        predictions: Predicted label indices, shape (n_samples, num_levels).
        labels: Ground truth label indices, shape (n_samples, num_levels).

    Returns:
        Dictionary of metric names to scalar or list values.
    """
    acc = compute_accuracy_per_level(predictions, labels)
    f1  = compute_f1_per_level(predictions, labels)
    ha  = compute_true_hierarchical_accuracy(predictions, labels)
    hp, hr, hf = compute_hierarchical_precision_recall(predictions, labels)
    cascade = compute_cascading_error_rate(predictions, labels)

    metrics: dict[str, float | list[float]] = {}

    for level, (a, f) in enumerate(zip(acc, f1)):
        metrics[f"acc_l{level + 1}"] = a
        metrics[f"f1_l{level + 1}"] = f

    metrics["hierarchical_accuracy"] = ha
    metrics["hP"] = hp
    metrics["hR"] = hr
    metrics["hF"] = hf

    for level, rate in enumerate(cascade):
        metrics[f"cascade_l{level + 2}"] = rate

    return metrics
