"""
cardiac_outcome_model/evaluate.py
====================================
Evalúa el modelo de outcome clínico cardíaco entrenado sobre el split de test.

Guarda:
    - reports/cardiac_outcome/test_metrics.csv (accuracy, precision, recall, F1)
    - reports/cardiac_outcome/confusion_matrix.png

Uso:
    python cardiac_outcome_model/evaluate.py
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
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_OUTCOME_CHECKPOINTS_DIR, CARDIAC_OUTCOME_REPORTS_DIR
from cardiac_outcome_model.dataset import OutcomeDataset, collate_pad
from cardiac_outcome_model.model import OutcomeCNN

BATCH_SIZE = 32


def load_best_model(device, checkpoint_dir=CARDIAC_OUTCOME_CHECKPOINTS_DIR) -> OutcomeCNN:
    model = OutcomeCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device))
    model.eval()
    return model


def run_inference(model, loader, device) -> tuple[list, list]:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_preds


def report(all_labels, all_preds, metrics_csv: Path, confusion_png: Path,
           title: str = "Modelo de outcome — matriz de confusión (test)") -> dict:
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
    disp = ConfusionMatrixDisplay(cm, display_labels=["normal", "abnormal"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(confusion_png, dpi=120)
    plt.close(fig)

    print(f"\n✓ Métricas guardadas en: {metrics_csv}")
    print(f"✓ Matriz de confusión guardada en: {confusion_png}\n")
    return metrics


def evaluate(index_csv=CARDIAC_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_WINDOWS_DIR,
             checkpoint_dir=CARDIAC_OUTCOME_CHECKPOINTS_DIR, reports_dir=CARDIAC_OUTCOME_REPORTS_DIR,
             feature_type: str = "mfcc", n_mfcc: int = 20) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = OutcomeDataset(index_csv, windows_dir, split="test", feature_type=feature_type, n_mfcc=n_mfcc)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)
    print(f"Ventanas de test: {len(test_ds)}")

    model = load_best_model(device, checkpoint_dir=checkpoint_dir)
    all_labels, all_preds = run_inference(model, test_loader, device)

    report(
        all_labels, all_preds,
        metrics_csv=reports_dir / "test_metrics.csv",
        confusion_png=reports_dir / "confusion_matrix.png",
    )


if __name__ == "__main__":
    evaluate()
