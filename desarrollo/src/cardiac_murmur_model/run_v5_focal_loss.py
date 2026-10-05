"""
cardiac_murmur_model/run_v5_focal_loss.py
=============================================
Murmullo v5: MFCC (igual que v1) + FocalLoss en vez de BCE+pos_weight —
ver cardiac_murmur_model/losses.py y avances/10_cardiaco_split_dos_tareas.md,
sección 10. Motivación: v1/v2/v3/v4 mostraron val_loss con picos fuertes
en varias épocas; FocalLoss concentra el gradiente en los ejemplos
difíciles en vez de escalar uniformemente los positivos (lo que hace
pos_weight), podría estabilizar el entrenamiento.

Guarda checkpoint y reportes en carpetas separadas de v1-v4:
    cardiac_murmur_model/checkpoints_v5_focal_loss/best.pt
    reports/cardiac_murmur/runs/v5_focal_loss/

Uso:
    python cardiac_murmur_model/run_v5_focal_loss.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_MURMUR_V5_CHECKPOINTS_DIR, CARDIAC_MURMUR_V5_REPORTS_DIR
from cardiac_murmur_model.train import train
from cardiac_murmur_model.evaluate import evaluate


def main() -> None:
    print("=" * 55)
    print("  MURMULLO v5 — MFCC + FocalLoss")
    print("=" * 55)
    train(checkpoint_dir=CARDIAC_MURMUR_V5_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V5_REPORTS_DIR,
          feature_type="mfcc", loss_type="focal")
    evaluate(checkpoint_dir=CARDIAC_MURMUR_V5_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V5_REPORTS_DIR,
              feature_type="mfcc")


if __name__ == "__main__":
    main()
