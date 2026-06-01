from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from transformers import AutoModel
import logging

from src.config import BaseConfig

logger = logging.getLogger(__name__)


class HierarchicalClassifier(ABC, nn.Module):
    """
    Abstract base class for all hierarchical text classification models.

    Provides a shared DistilBERT encoder and a CLS token extraction helper.
    All subclasses must implement forward() and predict().

    Args:
        config: Dataset-specific config carrying backbone name, dropout,
                num_levels and classes_per_level.
    """

    def __init__(self, config: BaseConfig, token: str, freeze_n_layers: int):
        super().__init__()
        self.config = config
        self.num_levels = config.num_levels
        self.classes_per_level = config.classes_per_level

        # Shared encoder: all classification heads operate on the same
        # contextualized representation
        self.encoder = AutoModel.from_pretrained(
            config.backbone_name,
            token=token
        )
        self.hidden_size = self.encoder.config.hidden_size  # 768 for DistilBERT
        self.dropout = nn.Dropout(config.dropout)

        # Freeze encoder layers if configured
        if freeze_n_layers != 0:
            self._freeze_encoder_layers(freeze_n_layers)
        else:
            # Only enable gradient checkpointing when encoder is being fine-tuned
            self.encoder.gradient_checkpointing_enable()

    def get_cls_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs the encoder and extracts the CLS token representation.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            CLS token representation, shape (batch_size, hidden_size).
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # DistilBERT returns last_hidden_state; CLS token is at position 0
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.dropout(cls_output)

    def _build_classification_head(self, num_classes: int) -> nn.Linear:
        """
        Builds a single linear classification head.

        Args:
            num_classes: Number of output classes for this head.

        Returns:
            A Linear layer mapping hidden_size -> num_classes.
        """
        return nn.Linear(self.hidden_size, num_classes)

    def _freeze_encoder_layers(self, n_layers: int) -> None:
        """
        Freezes encoder layers to reduce backbone dominance.

        Freezing the embeddings and lower transformer layers forces the
        classification heads to do more of the discriminative work,
        making architectural differences between strategies more visible.

        Args:
            n_layers: Number of transformer layers to freeze from the bottom.
                    -1 freezes all layers including embeddings.
                    n freezes embeddings + first n transformer layers.
        """

        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        if n_layers == -1:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Frozen: entire encoder (embeddings + all layers)")
            return

        for i in range(min(n_layers, 6)):
          for param in self.encoder.transformer.layer[i].parameters():
              param.requires_grad = False

        n_frozen = sum(
            1 for p in self.encoder.parameters() if not p.requires_grad
        )
        n_total = sum(
            1 for p in self.encoder.parameters()
        )
        logger.info(
            f"Frozen: embeddings + first {n_layers} transformer layers | "
            f"{n_frozen}/{n_total} parameter tensors frozen"
        )


    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).
            labels: Ground truth label indices, shape (batch_size, num_levels).
                    If provided, the returned dict must include a 'loss' key.

        Returns:
            A dict containing at minimum:
                - 'logits': list of tensors, one per level,
                            each shape (batch_size, classes_per_level[i])
                - 'loss': scalar tensor (only if labels are provided)
        """
        pass

    @abstractmethod
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Runs inference and returns the predicted label indices.

        Subclasses implement hierarchy-specific decoding here, e.g.
        hierarchical masking for Global Multi-Head or sequential
        decoding for LCL/LCN.

        Args:
            input_ids: Token ids, shape (batch_size, seq_length).
            attention_mask: Attention mask, shape (batch_size, seq_length).

        Returns:
            Predicted label indices, shape (batch_size, num_levels).
        """
        pass
