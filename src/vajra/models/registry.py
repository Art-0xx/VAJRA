# Model factory for VAJRA.
from __future__ import annotations

import torch.nn as nn

from vajra.config import Config
from vajra.models.transformer import TransformerNIDS


def get_model(config: Config, n_features: int, store_attention: bool = False) -> nn.Module:
    architecture = config.get("model.architecture", "transformer")
    if architecture == "transformer":
        return TransformerNIDS(
            n_features=n_features,
            d_model=config.get("model.d_model", 128),
            n_heads=config.get("model.n_heads", 8),
            n_layers=config.get("model.n_layers", 4),
            n_classes=config.get("model.n_classes", 2),
            dropout=config.get("model.dropout", 0.1),
            store_attention=store_attention,
        )
    raise ValueError(f"Unknown architecture: {architecture}")