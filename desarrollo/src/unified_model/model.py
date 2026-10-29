"""
unified_model/model.py
=========================
CNN 2D del modelo único sin gate (avances/22): el mismo bloque
convolucional que gate y murmullo (MurmurCNN), con 6 salidas softmax en vez
de un logit. El ancho es configurable para comparar a igual presupuesto de
parámetros:

    base  (width=16): canales 16-32-64,  ~24 k parámetros — igual a cada rama chica
    ancha (width=32): canales 32-64-128, ~93 k parámetros — del orden de la
                      suma gate + murmullo + pulmonar chica, los tres modelos
                      que reemplaza

La entrada es la onda (B, n) a 4 kHz: las features (dataset.SpectralFrontend,
MFCC o log-Mel, sin parámetros entrenables) van adentro del modelo, así el
checkpoint se usa igual al entrenar y al predecir.

Pooling final (parámetro `pooling`):
- "avg" (piloto 1): promedio global sobre frecuencia y tiempo, como gate y
  murmullo. Diluye los eventos cortos (crackles): mismo problema que la v1
  pulmonar (avances/13).
- "avg_max": promedio sobre frecuencia y después, sobre el tiempo, el
  promedio y el máximo concatenados (2 × canales). El promedio sirve al
  murmullo, que ocupa buena parte del ciclo; el máximo, a los eventos
  cortos (lo que hizo "time_max" en la v2 pulmonar).

Resolución temporal (ajuste 1, avances/22): un crackle dura 10-20 ms y el
frontend de gate/murmullo (n_fft 512, hop 160 a 4 kHz) da frames de 40 ms
sobre ventanas de 128 ms; con tres MaxPool 2×2, cada paso final resume
~320 ms. n_fft/hop configurables y `time_keep_blocks`: los últimos N
bloques reducen solo frecuencia (MaxPool 2×1), sin perder tiempo. Ninguno
de los dos cambia la cantidad de parámetros.

SpecAugment (ajuste 3): en train, `specaug` máscaras de hasta 8 bandas y
de hasta 10% de los frames, en 0 (la media tras el z-score).

Ablación por dominio (avances/22): `active_classes` restringe la softmax a
las clases de un dominio (logits del otro en -inf), así la misma red se
entrena solo como rama cardíaca (2 clases) o pulmonar (4) sin cambiar el
resto del código. Las filas sobrantes del clasificador no se usan (516
parámetros de más en la cardíaca, 258 en la pulmonar).
"""

import torch
import torch.nn as nn

from unified_model.dataset import SpectralFrontend

N_CLASSES = 6
WIDTHS = {"base": 16, "wide": 32}
POOLING_TYPES = ("avg", "avg_max")


def _conv_block(in_channels: int, out_channels: int, keep_time: bool = False) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d((2, 1) if keep_time else 2),
    )


def spec_augment(x: torch.Tensor, n_masks: int, max_bands: int = 8, max_time_frac: float = 0.1) -> torch.Tensor:
    """x: (B, 1, bandas, frames). n_masks máscaras de frecuencia y n_masks de
    tiempo por ejemplo, de ancho uniforme en [0, máximo]."""
    b, _, n_f, n_t = x.shape
    x = x.clone()
    max_t = max(1, int(max_time_frac * n_t))
    for _ in range(n_masks):
        for size, maximum, axis in ((n_f, max_bands, 2), (n_t, max_t, 3)):
            width = torch.randint(0, maximum + 1, (b,))
            start = (torch.rand(b) * (size - width + 1)).long()
            idx = torch.arange(size)
            mask = (idx[None] >= start[:, None]) & (idx[None] < (start + width)[:, None])   # (B, size)
            shape = (b, 1, size, 1) if axis == 2 else (b, 1, 1, size)
            x = x.masked_fill(mask.view(shape), 0.0)
    return x


class UnifiedCNN(nn.Module):
    def __init__(self, width: int = 16, n_classes: int = N_CLASSES, dropout: float = 0.3,
                 feature: str = "mfcc", pooling: str = "avg", n_fft: int = 512, hop_length: int = 160,
                 time_keep_blocks: int = 0, specaug: int = 0, active_classes=None):
        super().__init__()
        if pooling not in POOLING_TYPES:
            raise ValueError(f"pooling debe ser uno de {POOLING_TYPES}, recibido: {pooling!r}")
        self.pooling = pooling
        self.specaug = specaug
        self.frontend = SpectralFrontend(feature, n_fft=n_fft, hop_length=hop_length)
        chans = [1, width, 2 * width, 4 * width]
        self.features = nn.Sequential(*[
            _conv_block(chans[k], chans[k + 1], keep_time=k >= 3 - time_keep_blocks) for k in range(3)
        ])
        self.dropout = nn.Dropout(dropout)
        n_emb = 4 * width if pooling == "avg" else 8 * width
        self.classifier = nn.Linear(n_emb, n_classes)
        mask = torch.zeros(n_classes)
        if active_classes is not None:
            mask.fill_(float("-inf"))
            mask[list(active_classes)] = 0.0
        self.register_buffer("class_mask", mask, persistent=False)

    def forward(self, y):
        x = self.frontend(y)
        if self.training and self.specaug:
            x = spec_augment(x, self.specaug)
        x = self.features(x)                        # (batch, canales, frecuencia, tiempo)
        if self.pooling == "avg":
            x = x.mean(dim=(2, 3))
        else:
            x = x.mean(dim=2)
            x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)
        x = self.dropout(x)
        return self.classifier(x) + self.class_mask  # logits, usar con CrossEntropyLoss
