"""
gate_model/evaluate.py
========================
Evalúa el modelo gate entrenado sobre el split de test.

Guarda:
    - reports/gate/test_metrics.csv (accuracy, precision, recall, F1)
    - reports/gate/confusion_matrix.png

Las funciones _run_inference y _report se reutilizan en
gate_model/evaluate_external_hls.py para la validación externa con HLS-CMDS.

Uso:
    python gate_model/evaluate.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, GATE_CHECKPOINTS_DIR, GATE_REPORTS_DIR
from gate_model.dataset import GateDataset
from gate_model.model import GateCNN

BATCH_SIZE = 32


def load_best_model(device, checkpoint_dir=GATE_CHECKPOINTS_DIR) -> GateCNN:
    model = GateCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device))
    model.eval()
    return model


def run_inference(model, loader, device) -> tuple[list, list]:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_preds


def report(all_labels, all_preds, metrics_csv: Path, confusion_png: Path, title: str,
           display_labels=("negative", "positive")) -> dict:
    """Calcula métricas, las imprime, y guarda CSV + matriz de confusión."""
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
    }

    print(f"\n── Métricas — {title} ──")
    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}")

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(display_labels))
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(confusion_png, dpi=120)
    plt.close(fig)

    print(f"\n✓ Métricas guardadas en: {metrics_csv}")
    print(f"✓ Matriz de confusión guardada en: {confusion_png}\n")
    return metrics


def evaluate(index_csv=GATE_WINDOWS_INDEX_CSV, windows_dir=GATE_WINDOWS_DIR,
             label_column: str = "sound_type", positive_value: str = "cardiac",
             checkpoint_dir=GATE_CHECKPOINTS_DIR, reports_dir=GATE_REPORTS_DIR,
             display_labels=("pulmonary", "cardiac"),
             title: str = "Modelo gate — matriz de confusión (test)",
             feature_type: str = "logmel", n_mfcc: int = 20) -> None:
    """
    Parametrizado igual que train() para reutilizarse en el diagnóstico de
    sesgo por source_db (ver gate_model/diagnostics_source_db.py).

    feature_type/n_mfcc deben coincidir con los usados al entrenar el
    checkpoint que se está evaluando (ver gate_model/run_v5_mfcc.py).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = GateDataset(index_csv, windows_dir, split="test",
                           label_column=label_column, positive_value=positive_value,
                           feature_type=feature_type, n_mfcc=n_mfcc)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Ventanas de test: {len(test_ds)}")

    model = load_best_model(device, checkpoint_dir=checkpoint_dir)
    all_labels, all_preds = run_inference(model, test_loader, device)

    report(
        all_labels, all_preds,
        metrics_csv=reports_dir / "test_metrics.csv",
        confusion_png=reports_dir / "confusion_matrix.png",
        title=title,
        display_labels=display_labels,
    )


if __name__ == "__main__":
    evaluate()
