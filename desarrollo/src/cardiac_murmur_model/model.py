"""
cardiac_murmur_model/model.py
================================
CNN 2D para el modelo de detección de murmullo cardíaco (binario: presente/
ausente), sobre espectrogramas MFCC. Arranca con el mismo bloque
convolucional que ya funcionó en el gate y en el intento de 3 clases
(cardiac_model/), como punto de partida común para comparar — este archivo
es independiente del de gate_model/cardiac_model, así que su capacidad se
puede ajustar libremente sin afectar al modelo de outcome
(cardiac_outcome_model/) ni al gate.
"""

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class MurmurCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x).squeeze(1)  # logit, usar con BCEWithLogitsLoss


class MurmurCNNMultimodal(nn.Module):
    """
    Igual rama de audio que MurmurCNN (v1) — se agrega una rama chica para
    la metadata del paciente (sexo, edad, altura, peso; ver
    cardiac_murmur_model/dataset.py:encode_metadata), fusionada por
    concatenación antes de la capa final ("late fusion"), en vez de
    mezclarla con el audio antes (ver avances/10_cardiaco_split_dos_tareas.md,
    sección 8, para la motivación y el chequeo de que no sea un atajo
    demográfico).
    """

    def __init__(self, meta_dim: int, dropout: float = 0.3, meta_embed_dim: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, meta_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(64 + meta_embed_dim, 1)

    def forward(self, x_audio, x_meta):
        a = self.features(x_audio)
        a = self.pool(a).flatten(1)
        a = self.dropout(a)
        m = self.meta_fc(x_meta)
        combined = torch.cat([a, m], dim=1)
        return self.classifier(combined).squeeze(1)  # logit, usar con BCEWithLogitsLoss
