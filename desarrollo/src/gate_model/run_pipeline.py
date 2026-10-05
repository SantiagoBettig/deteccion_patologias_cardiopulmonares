"""
gate_model/run_pipeline.py
=============================
Corre entrenamiento + evaluación (test) + validación externa (HLS-CMDS) del
gate en una sola llamada, para poder lanzar todo y dejar la PC corriendo sin
tener que estar pendiente de cuándo termina cada paso para lanzar el
siguiente a mano.

Con los parámetros por default (feature_type="logmel"), reemplaza correr a
mano los pasos 5-7 del README (train.py, evaluate.py,
evaluate_external_hls.py) del gate vigente.

Uso:
    python gate_model/run_pipeline.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, GATE_CHECKPOINTS_DIR, GATE_REPORTS_DIR
from gate_model.train import train
from gate_model.evaluate import evaluate
from gate_model.evaluate_external_hls import evaluate_external_hls


def run_pipeline(index_csv=GATE_WINDOWS_INDEX_CSV, windows_dir=GATE_WINDOWS_DIR,
                  label_column: str = "sound_type", positive_value: str = "cardiac",
                  checkpoint_dir=GATE_CHECKPOINTS_DIR, reports_dir=GATE_REPORTS_DIR,
                  feature_type: str = "logmel", n_mfcc: int = 20) -> None:
    print("=" * 55)
    print("  1/3 — Entrenamiento")
    print("=" * 55)
    train(
        index_csv=index_csv, windows_dir=windows_dir,
        label_column=label_column, positive_value=positive_value,
        checkpoint_dir=checkpoint_dir, reports_dir=reports_dir,
        feature_type=feature_type, n_mfcc=n_mfcc,
    )

    print("\n" + "=" * 55)
    print("  2/3 — Evaluación (test)")
    print("=" * 55)
    evaluate(
        index_csv=index_csv, windows_dir=windows_dir,
        label_column=label_column, positive_value=positive_value,
        checkpoint_dir=checkpoint_dir, reports_dir=reports_dir,
        feature_type=feature_type, n_mfcc=n_mfcc,
    )

    print("\n" + "=" * 55)
    print("  3/3 — Validación externa (HLS-CMDS)")
    print("=" * 55)
    evaluate_external_hls(
        checkpoint_dir=checkpoint_dir, reports_dir=reports_dir,
        feature_type=feature_type, n_mfcc=n_mfcc,
    )

    print("\n✓ Pipeline completo.")


if __name__ == "__main__":
    run_pipeline()
