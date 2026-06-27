"""Fusion Encoder — combines temporal + entropy (+ device echo) embeddings.

Inputs: temporal_emb (128-d), entropy_emb (64-d), [device_emb (32-d optional)]
Outputs: 256-dimensional fused behavioral fingerprint embedding

Architecture: Cross-modal Transformer fusion
"""

import torch
import torch.nn as nn


class FusionEncoder(nn.Module):
    def __init__(
        self,
        temporal_dim: int = 128,
        entropy_dim: int = 64,
        device_dim: int = 32,
        output_dim: int = 256,
    ):
        super().__init__()
        # Project all modalities to same dim
        self.temporal_proj = nn.Linear(temporal_dim, 128)
        self.entropy_proj = nn.Linear(entropy_dim, 128)
        self.device_proj = nn.Linear(device_dim, 128)

        # Cross-modal attention
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(128)
        self.output_proj = nn.Linear(128 * 3, output_dim)

    def forward(
        self,
        temporal_emb: torch.Tensor,
        entropy_emb: torch.Tensor,
        device_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        t = self.temporal_proj(temporal_emb).unsqueeze(1)   # (B, 1, 128)
        e = self.entropy_proj(entropy_emb).unsqueeze(1)     # (B, 1, 128)

        if device_emb is not None:
            d = self.device_proj(device_emb).unsqueeze(1)   # (B, 1, 128)
        else:
            d = torch.zeros_like(t)                          # zero-fill if no device signal

        # Stack as sequence for attention
        seq = torch.cat([t, e, d], dim=1)                   # (B, 3, 128)
        attn_out, _ = self.attn(seq, seq, seq)
        seq = self.norm(seq + attn_out)                     # residual

        flat = seq.reshape(seq.size(0), -1)                 # (B, 384)
        return self.output_proj(flat)                       # (B, 256)


if __name__ == "__main__":
    model = FusionEncoder()
    t_emb = torch.rand(8, 128)
    e_emb = torch.rand(8, 64)
    out = model(t_emb, e_emb)
    print(f"Fusion embedding shape: {out.shape}")  # (8, 256)
