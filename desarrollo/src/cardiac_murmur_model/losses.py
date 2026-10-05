"""
cardiac_murmur_model/losses.py
=================================
Focal loss binaria (Lin et al., 2017) — alternativa a BCEWithLogitsLoss +
pos_weight para el desbalance de clases. En vez de escalar uniformemente
el gradiente de la clase positiva (lo que puede desestabilizar el
entrenamiento, ver avances/10_cardiaco_split_dos_tareas.md, sección 5-7:
val_loss con picos fuertes en las 4 versiones probadas), baja el peso de
los ejemplos "fáciles" (donde el modelo ya está seguro) y concentra el
gradiente en los difíciles — alpha sigue controlando el balance
positivo/negativo, gamma controla cuánto se ignoran los ejemplos fáciles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - p_t).pow(self.gamma) * bce
        return loss.mean()
