import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from src.config import BaseConfig
from src.data.hierarchy import HierarchyManager
from src.models.backbone import HierarchicalClassifier


class LocalClassifierPerParentNode(HierarchicalClassifier):
    """
    Local Classifier per Parent Node (LCPN) for hierarchical text classification.

    Trains one independent classification head for each internal node of the
    hierarchy. Each node classifier distinguishes only between the direct
    children of that node, reducing each sub-problem to a smaller, focused
    classification task.

    During training, each node classifier is updated only on samples that
    pass through that node (i.e. whose ground-truth path includes the node
    as a parent). This is the defining characteristic of LCPN and differs
    fundamentally from LCL and Global Multi-Head, where all heads receive
    the full batch.

    During inference, predictions are made sequentially top-down: the root
    classifier predicts the L1 class, which determines which L2 node
    classifier is invoked next. No masking is needed since each node
    classifier only outputs valid children by construction.

    Args:
        config: Dataset-specific config.
        hierarchy: Fully built HierarchyManager.
    """

    def __init__(self, config: BaseConfig, token: str, hierarchy: HierarchyManager, freeze_n_layers: int):
        super().__init__(config, token, freeze_n_layers)
        self.hierarchy = hierarchy

        # Build one classification head per internal node.
        # The root node covers all L1 classes.
        # Each L1 node covers its direct L2 children (and so on for deeper hierarchies).
        #
        # heads is a nn.ModuleDict with string keys of the form:
        #   "root"        -> classifies among all L1 classes
        #   "l1_{idx}"    -> classifies among children of L1 node idx
        #   "l2_{idx}"    -> classifies among children of L2 node idx (DBpedia only)
        self.heads = nn.ModuleDict()
        self.node_to_children: dict[str, list[int]] = {}

        self._build_node_classifiers()

    def _build_node_classifiers(self) -> None:
        """
        Constructs one classification head per internal node.

        The root classifier covers all classes at level 0. For each subsequent
        level, one classifier is built per parent node, covering only that
        node's direct children.
        """
        # Root classifier: all L1 classes
        num_l1 = self.classes_per_level[0]
        self.heads["root"] = self._build_classification_head(num_l1)
        self.node_to_children["root"] = list(range(num_l1))

        # Node classifiers for each subsequent level
        for level in range(self.num_levels - 1):
            num_parents = self.classes_per_level[level]
            prefix = f"l{level + 1}"

            for parent_idx in range(num_parents):
                children = self.hierarchy.get_valid_children(
                    level=level,
                    parent_idx=parent_idx,
                )
                if not children:
                    continue

                key = f"{prefix}_{parent_idx}"
                self.heads[key] = self._build_classification_head(len(children))
                # Store the mapping from local output index -> global class index
                self.node_to_children[key] = children

    def _get_node_key(self, level: int, parent_idx: int) -> str:
        """Returns the ModuleDict key for a node classifier."""
        if level == 0:
            return "root"
        return f"l{level}_{parent_idx}"

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with per-node local training.

        Each node classifier is applied only to the subset of samples in the
        batch whose ground-truth path passes through that node. Loss is
        averaged across all active node classifiers that received at least
        one sample.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).
            labels: Ground truth indices, shape (batch_size, num_levels).
                    Required for loss computation and for routing samples
                    to the correct node classifiers.

        Returns:
            A dict with:
                - 'logits': dict mapping node keys to local logit tensors.
                             Each tensor has shape (n_relevant_samples, n_children).
                - 'loss': mean of per-node cross-entropy losses (if labels given).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_hidden = outputs.last_hidden_state[:, 0, :]
        cls_hidden = self.dropout(cls_hidden)

        node_logits: dict[str, torch.Tensor] = {}
        node_losses: list[torch.Tensor] = []

        # --- Root classifier: all samples ---
        root_logits = self.heads["root"](cls_hidden)
        node_logits["root"] = root_logits

        if labels is not None:
            root_loss = F.cross_entropy(root_logits, labels[:, 0])
            node_losses.append(root_loss)

        # --- Node classifiers for deeper levels ---
        for level in range(self.num_levels - 1):
            num_parents = self.classes_per_level[level]
            prefix = f"l{level + 1}"

            for parent_idx in range(num_parents):
                key = f"{prefix}_{parent_idx}"
                if key not in self.heads:
                    continue

                # Select only samples whose ground-truth parent is this node
                if labels is not None:
                    mask = labels[:, level] == parent_idx
                    relevant_indices = mask.nonzero(as_tuple=True)[0]

                    if len(relevant_indices) == 0:
                        continue

                    relevant_cls = cls_hidden[relevant_indices]
                    local_logits = self.heads[key](relevant_cls)
                    node_logits[key] = local_logits

                    # Map global child labels to local indices for this node
                    global_children = self.node_to_children[key]
                    global_to_local = {
                        g: l for l, g in enumerate(global_children)
                    }
                    global_labels = labels[relevant_indices, level + 1]
                    local_labels = torch.tensor(
                        [global_to_local[g.item()] for g in global_labels],
                        dtype=torch.long,
                        device=labels.device,
                    )
                    node_loss = F.cross_entropy(local_logits, local_labels)
                    node_losses.append(node_loss)

        result = {"logits": node_logits}

        if labels is not None and node_losses:
            # Average across all active node classifiers
            result["loss"] = torch.stack(node_losses).mean()

        return result

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sequential top-down inference through node classifiers.

        No hierarchical masking is needed: each node classifier only outputs
        valid children by construction, so predictions are always
        taxonomy-consistent.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            Predicted label indices, shape (batch_size, num_levels).
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            cls_hidden = outputs.last_hidden_state[:, 0, :]
            cls_hidden = self.dropout(cls_hidden)

        batch_size = input_ids.shape[0]
        predictions = torch.zeros(
            batch_size, self.num_levels, dtype=torch.long, device=input_ids.device
        )

        # Level 0: root classifier
        root_logits = self.heads["root"](cls_hidden)
        predictions[:, 0] = root_logits.argmax(dim=-1)

        # Levels 1+: route each sample to its parent's node classifier
        for level in range(1, self.num_levels):
            prefix = f"l{level}"

            # Group samples by their predicted parent
            parent_predictions = predictions[:, level - 1]
            unique_parents = parent_predictions.unique()

            for parent_idx in unique_parents:
                parent_idx = parent_idx.item()
                key = f"{prefix}_{parent_idx}"

                if key not in self.heads:
                    continue

                sample_mask = parent_predictions == parent_idx
                relevant_indices = sample_mask.nonzero(as_tuple=True)[0]
                relevant_cls = cls_hidden[relevant_indices]

                local_logits = self.heads[key](relevant_cls)
                local_predictions = local_logits.argmax(dim=-1)

                # Map local predictions back to global class indices
                global_children = self.node_to_children[key]
                global_predictions = torch.tensor(
                    [global_children[l.item()] for l in local_predictions],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                predictions[relevant_indices, level] = global_predictions

        return predictions
