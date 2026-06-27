"""Temporal Encoder — Point Process Transformer for inter-event sequences.

Inputs: sequence of inter-event intervals (seconds) per account
Outputs: 128-dimensional temporal embedding

Architecture:
- Learnable positional encoding over interval sequences
- Transformer encoder (4 heads, 2 layers)
- Pooling → 128-d projection head

Training: Self-supervised via masked interval prediction.
"""

import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, intervals: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            intervals: (batch, seq_len) — inter-event intervals in seconds
            mask: (batch, seq_len) — padding mask (True = ignore)
        Returns:
            embeddings: (batch, output_dim)
        """
        x = intervals.unsqueeze(-1)       # (B, S, 1)
        x = self.input_proj(x)            # (B, S, d_model)
        x = self.transformer(x, src_key_padding_mask=mask)  # (B, S, d_model)
        x = x.permute(0, 2, 1)           # (B, d_model, S)
        x = self.pool(x).squeeze(-1)     # (B, d_model)
        return self.output_proj(x)        # (B, output_dim)


if __name__ == "__main__":
    model = TemporalEncoder()
    dummy = torch.rand(8, 64)  # batch of 8 accounts, 64 intervals each
    out = model(dummy)
    print(f"Temporal embedding shape: {out.shape}")  # (8, 128)
