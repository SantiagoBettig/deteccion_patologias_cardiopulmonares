"""
cardiac_murmur_model/run_v3_lr_scheduler.py
==============================================
Murmullo v3: MFCC (igual que v1 — v2 con log-Mel perdió la comparación,
ver avances/10_cardiaco_split_dos_tareas.md sección 6) + ReduceLROnPlateau
sobre val_f2, para amortiguar los picos de val_loss vistos en v1/v2.

Guarda checkpoint y reportes en carpetas separadas de v1/v2, para no
pisarlas:
    cardiac_murmur_model/checkpoints_v3_lr_scheduler/best.pt
    reports/cardiac_murmur/runs/v3_lr_scheduler/

Uso:
    python cardiac_murmur_model/run_v3_lr_scheduler.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_MURMUR_V3_CHECKPOINTS_DIR, CARDIAC_MURMUR_V3_REPORTS_DIR
from cardiac_murmur_model.train import train
from cardiac_murmur_model.evaluate import evaluate


def main() -> None:
    print("=" * 55)
    print("  MURMULLO v3 — MFCC + ReduceLROnPlateau sobre val_f2")
    print("=" * 55)
    train(checkpoint_dir=CARDIAC_MURMUR_V3_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V3_REPORTS_DIR,
          feature_type="mfcc", use_scheduler=True)
    evaluate(checkpoint_dir=CARDIAC_MURMUR_V3_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V3_REPORTS_DIR,
              feature_type="mfcc")


if __name__ == "__main__":
    main()
