"""
gate_model/diagnostics_source_db.py
=====================================
Diagnóstico cuantitativo del sesgo detectado en el gate v1 (ver
avances/4_gate_model_baseline_y_sesgo.md).

Entrena la MISMA arquitectura (GateCNN), sobre las MISMAS ventanas de audio
del gate, pero prediciendo `source_db` (CirCor vs. ICBHI) en vez de
`sound_type` (cardíaco vs. pulmonar). Como en el dataset de entrenamiento
sound_type está perfectamente correlacionado con source_db, si este
clasificador "trivial" también alcanza una accuracy muy alta, confirma
cuantitativamente que la red puede separar el origen del dataset con
facilidad — evidencia de que el gate real probablemente esté explotando
ese mismo atajo en vez de aprender la diferencia acústica real.

No usa HLS-CMDS (no tiene columna source_db comparable — es un tercer
origen que no participa de este chequeo).

Guarda checkpoint y reportes en una carpeta separada de los del gate real,
para no pisarlos:
    gate_model/checkpoints_source_diag/best.pt
    reports/gate_diagnostics/source_db/

Uso:
    python gate_model/diagnostics_source_db.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR,
    GATE_DIAG_CHECKPOINTS_DIR, GATE_DIAG_REPORTS_DIR,
)
from gate_model.train import train
from gate_model.evaluate import evaluate

LABEL_COLUMN = "source_db"
POSITIVE_VALUE = "CIR"  # CIR=1 (CirCor), ICB=0 (ICBHI)


def main() -> None:
    print("=" * 55)
    print("  DIAGNÓSTICO: ¿se puede predecir source_db trivialmente?")
    print("=" * 55)

    train(
        index_csv=GATE_WINDOWS_INDEX_CSV, windows_dir=GATE_WINDOWS_DIR,
        label_column=LABEL_COLUMN, positive_value=POSITIVE_VALUE,
        checkpoint_dir=GATE_DIAG_CHECKPOINTS_DIR, reports_dir=GATE_DIAG_REPORTS_DIR,
    )

    evaluate(
        index_csv=GATE_WINDOWS_INDEX_CSV, windows_dir=GATE_WINDOWS_DIR,
        label_column=LABEL_COLUMN, positive_value=POSITIVE_VALUE,
        checkpoint_dir=GATE_DIAG_CHECKPOINTS_DIR, reports_dir=GATE_DIAG_REPORTS_DIR,
        display_labels=("ICBHI", "CirCor"),
        title="Diagnóstico — ¿se distingue source_db trivialmente? (test)",
    )


if __name__ == "__main__":
    main()
