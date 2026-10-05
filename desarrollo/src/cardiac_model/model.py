"""
cardiac_model/model.py
=========================
CNN 2D pequeña para el modelo de patología cardíaca (3 clases, sobre
espectrogramas MFCC). Mismo bloque convolucional que gate_model/model.py
(GateCNN) — se reutiliza el diseño que ya funcionó ahí, solo cambia la capa
final: n_classes logits en vez de 1, para CrossEntropyLoss en vez de
BCEWithLogitsLoss.
"""

import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CardiacCNN(nn.Module):
    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)  # logits, usar con CrossEntropyLoss
