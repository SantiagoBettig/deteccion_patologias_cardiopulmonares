"""
gate_model/run_v5_mfcc.py
============================
Gate v5: MFCC en vez de espectrograma log-Mel, sobre las mitigaciones de v4
(z-score, augmentation de audio, class weighting) — ver
avances/6_gate_v5_mfcc.md para la motivación y los resultados.

Guarda checkpoint y reportes en carpetas separadas de las del gate vigente
(v4), para no pisarlas:
    gate_model/checkpoints_v5_mfcc/best.pt
    reports/gate/runs/v5_mfcc/

Uso:
    python gate_model/run_v5_mfcc.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_V5_CHECKPOINTS_DIR, GATE_V5_REPORTS_DIR
from gate_model.run_pipeline import run_pipeline

N_MFCC = 20


def main() -> None:
    print("=" * 55)
    print(f"  GATE v5 — MFCC (n_mfcc={N_MFCC}) en vez de log-Mel")
    print("=" * 55)
    run_pipeline(
        checkpoint_dir=GATE_V5_CHECKPOINTS_DIR, reports_dir=GATE_V5_REPORTS_DIR,
        feature_type="mfcc", n_mfcc=N_MFCC,
    )


if __name__ == "__main__":
    main()
