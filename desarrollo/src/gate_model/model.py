"""
gate_model/model.py
=====================
CNN 2D pequeña para el modelo gate (clasificación binaria cardíaco/pulmonar
sobre espectrogramas log-Mel). Usa global average pooling antes de la capa
final para no depender de un tamaño de entrada exacto.
"""

import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class GateCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)  # logit, usar con BCEWithLogitsLoss
