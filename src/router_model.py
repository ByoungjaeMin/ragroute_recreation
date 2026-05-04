from __future__ import annotations

import torch
import torch.nn as nn


class CorpusRoutingNN(nn.Module):
    """Lightweight 3-layer classifier that predicts corpus relevance.

    Architecture (from original repo, NOT the paper description):
      fc1:    Linear(input_dim, 128) -> LayerNorm(128) -> ReLU -> Dropout(0.4)
      fc2:    Linear(128, 64)        -> LayerNorm(64)  -> ReLU -> Dropout(0.4)
      fc3:    Linear(64, 32)         -> LayerNorm(32)  -> ReLU -> Dropout(0.4)
      fc_out: Linear(32, 1)          -> raw logit  (NO sigmoid here)

    The caller (loss fn = BCEWithLogitsLoss) applies sigmoid internally.
    """

    def __init__(self, input_dim: int):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.4),
        )
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, 1) raw logit."""
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.fc_out(x)
