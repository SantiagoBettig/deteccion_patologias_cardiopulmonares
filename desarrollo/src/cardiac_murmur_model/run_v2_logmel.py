"""
cardiac_murmur_model/run_v2_logmel.py
========================================
Murmullo v2: espectrograma log-Mel en vez de MFCC, resto de la receta
idéntica a v1 (pos_weight x1.4, checkpoint por F2, dropout, umbral
calibrado sobre validación) — ver avances/10_cardiaco_split_dos_tareas.md,
sección 6, para la motivación (a diferencia del gate, acá no hay problema
de dominio cruzado que justifique la compresión de MFCC, y un murmullo
tiene contenido de alta frecuencia que MFCC podría estar recortando).

Guarda checkpoint y reportes en carpetas separadas de v1 (MFCC), para no
pisarlas:
    cardiac_murmur_model/checkpoints_v2_logmel/best.pt
    reports/cardiac_murmur/runs/v2_logmel/

Uso:
    python cardiac_murmur_model/run_v2_logmel.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_MURMUR_V2_CHECKPOINTS_DIR, CARDIAC_MURMUR_V2_REPORTS_DIR
from cardiac_murmur_model.train import train
from cardiac_murmur_model.evaluate import evaluate


def main() -> None:
    print("=" * 55)
    print("  MURMULLO v2 — log-Mel en vez de MFCC")
    print("=" * 55)
    train(checkpoint_dir=CARDIAC_MURMUR_V2_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V2_REPORTS_DIR,
          feature_type="logmel")
    evaluate(checkpoint_dir=CARDIAC_MURMUR_V2_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V2_REPORTS_DIR,
              feature_type="logmel")


if __name__ == "__main__":
    main()
