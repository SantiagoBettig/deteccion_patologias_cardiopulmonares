"""
cardiac_outcome_model/model.py
=================================
CNN 2D para el modelo de outcome clínico cardíaco (binario: normal/
abnormal), sobre espectrogramas MFCC. Arrancó con el mismo bloque de 3
capas que cardiac_murmur_model/ y el gate (baseline: accuracy 0.578 en
test, apenas sobre el azar — ver avances/10_cardiaco_split_dos_tareas.md,
sección 4). Se agrega un 4to bloque (64→128 canales) para probar si más
profundidad/capacidad ayuda a extraer una señal más sutil, antes de asumir
que el problema es de información disponible y no de capacidad.
"""

import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class OutcomeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x).squeeze(1)  # logit, usar con BCEWithLogitsLoss
