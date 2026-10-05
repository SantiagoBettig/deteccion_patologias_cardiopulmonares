"""
pulmonary_adventitious_model/model.py
========================================
CNN 2D para detección de sonidos adventicios por ciclo respiratorio.
Mismo bloque convolucional de partida que el gate y el modelo de murmullo
(para que la comparación entre ramas sea limpia), con **dos salidas
independientes** (logit de crackle, logit de wheeze) en vez de una softmax
de 4 clases:

- "ambos" (crackle + wheeze) no es una clase aparte sino las dos salidas
  activas a la vez — el modelo no tiene que aprender una cuarta categoría
  con solo ~500 ejemplos.
- cada salida tiene su propio umbral calibrado sobre validación.
- las 4 clases del challenge ICBHI (normal / crackle / wheeze / both) se
  reconstruyen después para calcular el score oficial (ver metrics.py).

Pooling final (parámetro `pooling`):
- "avg" (v1): promedio global sobre frecuencia y tiempo, igual que el gate
  y el murmullo. Descartado: diluye eventos cortos — un crackle ocupa pocos
  frames de ~250 y el promedio lo borra. Ni siquiera logra memorizar 600
  ciclos de train (ver avances/13).
- "time_max" (v2): promedio sobre frecuencia y **máximo sobre tiempo** —
  alcanza con que el evento aparezca una vez en el ciclo para activar la
  salida.
"""

import torch.nn as nn

POOLING_TYPES = ("avg", "time_max")


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class AdventitiousCNN(nn.Module):
    N_OUTPUTS = 2  # [crackle, wheeze]

    def __init__(self, dropout: float = 0.3, pooling: str = "time_max"):
        super().__init__()
        if pooling not in POOLING_TYPES:
            raise ValueError(f"pooling debe ser uno de {POOLING_TYPES}, recibido: {pooling!r}")
        self.pooling = pooling
        self.features = nn.Sequential(
            _conv_block(1, 16),
            _conv_block(16, 32),
            _conv_block(32, 64),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(64, self.N_OUTPUTS)

    def forward(self, x):
        x = self.features(x)
        # x: (batch, canales, frecuencia, tiempo)
        if self.pooling == "avg":
            x = x.mean(dim=(2, 3))
        else:
            x = x.mean(dim=2).amax(dim=2)
        x = self.dropout(x)
        return self.classifier(x)  # (batch, 2) logits, usar con BCEWithLogitsLoss
