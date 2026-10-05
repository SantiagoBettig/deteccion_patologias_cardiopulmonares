"""
gate_model/model_periodicity.py
===================================
Gate v6: misma CNN de gate_model/model.py (rama de timbre, sobre
log-Mel/MFCC) más una rama chica (MLP) sobre el vector de periodicidad
rítmica (gate_model/periodicity.py). Los embeddings de ambas ramas se
concatenan antes de la capa de clasificación final.

Ver avances/8_gate_v6_periodicity.md.
"""

import torch
import torch.nn as nn

from gate_model.model import _conv_block
from gate_model.periodicity import N_PERIODICITY_FEATURES


class GateCNNPeriodicity(nn.Module):
    def __init__(self, n_periodicity_features: int = N_PERIODICITY_FEATURES,
                 periodicity_hidden: int = 16):
        super().__init__()
        self.spec_features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
        )
        self.spec_pool = nn.AdaptiveAvgPool2d(1)

        self.periodicity_branch = nn.Sequential(
            nn.Linear(n_periodicity_features, periodicity_hidden),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(64 + periodicity_hidden, 1)

    def forward(self, x_spec, x_period):
        spec_emb = self.spec_pool(self.spec_features(x_spec)).flatten(1)  # (B, 64)
        period_emb = self.periodicity_branch(x_period)                     # (B, periodicity_hidden)
        combined = torch.cat([spec_emb, period_emb], dim=1)
        return self.classifier(combined).squeeze(1)  # logit, usar con BCEWithLogitsLoss
