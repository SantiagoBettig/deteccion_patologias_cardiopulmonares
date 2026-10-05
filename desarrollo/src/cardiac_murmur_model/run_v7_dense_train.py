"""
cardiac_murmur_model/run_v7_dense_train.py
==============================================
Murmullo v7: sobre la mejor receta encontrada hasta acá (a definir según
resultados de v5/v6), pero usando las ventanas de train más densas de
preprocessing/make_cardiac_cycle_windows_dense_train.py (hop=1 ciclo en
train, en vez de 2) — más ejemplos de entrenamiento por grabación, aunque
correlacionados entre sí (ver avances/10_cardiaco_split_dos_tareas.md,
sección 12).

val/test siguen viniendo de CARDIAC_DENSE_WINDOWS_DIR pero son idénticas
a las de CARDIAC_WINDOWS_DIR (mismo hop=2) — la comparación contra
v1/v5/v6 sigue siendo directa.

Uso:
    python cardiac_murmur_model/run_v7_dense_train.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    CARDIAC_DENSE_WINDOWS_DIR, CARDIAC_DENSE_WINDOWS_INDEX_CSV,
    CARDIAC_MURMUR_V7_CHECKPOINTS_DIR, CARDIAC_MURMUR_V7_REPORTS_DIR,
)
from cardiac_murmur_model.train import train
from cardiac_murmur_model.evaluate import evaluate


def main() -> None:
    print("=" * 55)
    print("  MURMULLO v7 — ventanas de train más densas (hop=1 ciclo)")
    print("=" * 55)
    train(index_csv=CARDIAC_DENSE_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_DENSE_WINDOWS_DIR,
          checkpoint_dir=CARDIAC_MURMUR_V7_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V7_REPORTS_DIR,
          feature_type="mfcc")
    evaluate(index_csv=CARDIAC_DENSE_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_DENSE_WINDOWS_DIR,
             checkpoint_dir=CARDIAC_MURMUR_V7_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V7_REPORTS_DIR,
             feature_type="mfcc")


if __name__ == "__main__":
    main()
