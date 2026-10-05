"""
gate_model/evaluate_external_hls.py
=====================================
Evalúa el modelo gate sobre HLS-CMDS, usado como set de VALIDACIÓN EXTERNA
(nunca visto en entrenamiento). Ver preprocessing/make_hls_windows.py para
la justificación completa: en CirCor+ICBHI, `sound_type` está perfectamente
correlacionado con `source_db`, así que este chequeo ayuda a detectar si el
modelo aprendió la diferencia acústica real entre sonidos cardíacos y
pulmonares, en vez de una firma del dataset/equipo de origen.

Requiere haber corrido antes:
    preprocessing/make_hls_windows.py
    gate_model/train.py (para tener un checkpoint entrenado)

Guarda:
    - reports/gate/external_hls_metrics.csv
    - reports/gate/external_hls_confusion_matrix.png

Uso:
    python gate_model/evaluate_external_hls.py
"""

import torch
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    GATE_WINDOWS_DIR, GATE_HLS_INDEX_CSV, GATE_REPORTS_DIR, GATE_CHECKPOINTS_DIR,
)
from preprocessing.make_hls_windows import EXTERNAL_SPLIT_NAME
from gate_model.dataset import GateDataset
from gate_model.evaluate import load_best_model, run_inference, report

BATCH_SIZE = 32


def evaluate_external_hls(checkpoint_dir=GATE_CHECKPOINTS_DIR, reports_dir=GATE_REPORTS_DIR,
                           feature_type: str = "logmel", n_mfcc: int = 20) -> None:
    """
    checkpoint_dir/reports_dir/feature_type/n_mfcc permiten apuntar la
    validación externa a un checkpoint/reportes distintos de los del gate
    vigente (ver gate_model/run_pipeline.py y gate_model/run_v5_mfcc.py).
    feature_type/n_mfcc deben coincidir con los usados al entrenar ese
    checkpoint.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = GateDataset(GATE_HLS_INDEX_CSV, GATE_WINDOWS_DIR, split=EXTERNAL_SPLIT_NAME,
                      feature_type=feature_type, n_mfcc=n_mfcc)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Ventanas HLS-CMDS (validación externa): {len(ds)}")

    model = load_best_model(device, checkpoint_dir=checkpoint_dir)
    all_labels, all_preds = run_inference(model, loader, device)

    report(
        all_labels, all_preds,
        metrics_csv=reports_dir / "external_hls_metrics.csv",
        confusion_png=reports_dir / "external_hls_confusion_matrix.png",
        title="Modelo gate — validación externa (HLS-CMDS)",
        display_labels=("pulmonary", "cardiac"),
    )


if __name__ == "__main__":
    evaluate_external_hls()
