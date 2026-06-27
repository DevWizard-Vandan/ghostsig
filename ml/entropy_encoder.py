"""Entropy Encoder — encodes linguistic entropy profiles into 64-d embeddings.

Inputs: char_entropy_mean, char_entropy_std, word_entropy_mean, word_entropy_std,
        sample_count (normalized), + future extension: n-gram frequency vectors
Outputs: 64-dimensional entropy embedding

Architecture: Simple MLP for now; can extend to Transformer over token distributions.
"""

import torch
import torch.nn as nn


class EntropyEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — entropy feature vector per account
        Returns:
            embeddings: (batch, output_dim)
        """
        return self.net(x)


if __name__ == "__main__":
    model = EntropyEncoder()
    dummy = torch.rand(8, 8)
    out = model(dummy)
    print(f"Entropy embedding shape: {out.shape}")  # (8, 64)
