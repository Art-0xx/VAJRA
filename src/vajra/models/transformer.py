# Transformer-based NIDS with feature tokenization and attention rollout support.
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TransformerNIDS(nn.Module):
    """Feature-token Transformer for tabular network flow classification.

    Each of the ``n_features`` input values becomes one token via a linear
    projection to ``d_model``. A learnable CLS token is prepended, and the
    Transformer encoder processes the resulting sequence. The CLS token's
    final representation is used for classification.

    When ``store_attention=True``, the attention weights from every layer are
    saved in ``self.attention_maps`` after each forward pass. This is the
    input to the attention-rollout justification module (Phase 4).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        n_classes: int = 2,
        dropout: float = 0.1,
        store_attention: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.store_attention = store_attention

        self.feature_tokenizer = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        self.register_buffer(
            "pos_encoding",
            self._sinusoidal_encoding(n_features + 1, d_model),
            persistent=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        self.attention_maps: list[torch.Tensor] = []

    @staticmethod
    def _sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.feature_tokenizer(x.unsqueeze(-1))

        cls = self.cls_token.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_encoding[:, : tokens.size(1), :]

        self.attention_maps = []

        if self.store_attention:
            # Recompute per-layer attention by mirroring the layer's internal
            # pre-attention normalization. This is used by the rollout module
            # in Phase 4; training uses the fast built-in encoder path.
            for layer in self.encoder.layers:
                with torch.no_grad():
                    pre = layer.norm1(tokens) if layer.norm_first else tokens
                    attn = layer.self_attn(
                        pre, pre, pre,
                        need_weights=True,
                        average_attn_weights=False,
                    )[1]
                self.attention_maps.append(attn.detach())
                tokens = layer(tokens)
        else:
            tokens = self.encoder(tokens)

        cls_out = self.norm(tokens[:, 0, :])
        return self.classifier(cls_out)

    def get_attention_maps(self) -> list[torch.Tensor]:
        if not self.attention_maps:
            raise RuntimeError(
                "No attention maps stored. Run forward with store_attention=True."
            )
        return self.attention_maps