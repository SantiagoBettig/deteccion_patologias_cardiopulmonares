"""
cardiac_model/evaluate.py
============================
Evalúa el modelo de patología cardíaca entrenado sobre el split de test.

Guarda:
    - reports/cardiac/test_metrics.csv (accuracy, F1 macro, precision/recall/F1 por clase)
    - reports/cardiac/confusion_matrix.png

Uso:
    python cardiac_model/evaluate.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_CHECKPOINTS_DIR, CARDIAC_REPORTS_DIR
from cardiac_model.dataset import CardiacDataset, collate_pad, CLASSES
from cardiac_model.model import CardiacCNN

BATCH_SIZE = 32


def load_best_model(device, checkpoint_dir=CARDIAC_CHECKPOINTS_DIR) -> CardiacCNN:
    model = CardiacCNN(n_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device))
    model.eval()
    return model


def run_inference(model, loader, device) -> tuple[list, list]:
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_preds


def report(all_labels, all_preds, metrics_csv: Path, confusion_png: Path,
           title: str = "Modelo cardíaco — matriz de confusión (test)") -> pd.DataFrame:
    """Calcula métricas por clase (foco en recall, ver estrategia acordada:
    un falso negativo de heart_murmur/abnormal importa más que uno de
    normal_heart), las imprime, y guarda CSV + matriz de confusión."""
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=range(len(CLASSES)), zero_division=0)

    metrics_df = pd.DataFrame({
        "class": CLASSES, "precision": precision, "recall": recall,
        "f1": f1, "support": support,
    })
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")

    print(f"\n── Métricas — {title} ──")
    print(f"  accuracy   : {accuracy:.4f}")
    print(f"  f1_macro   : {f1_macro:.4f}")
    print(metrics_df.to_string(index=False))

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_csv, index=False)

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(CLASSES)))
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, colorbar=False, xticks_rotation=25)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(confusion_png, dpi=120)
    plt.close(fig)

    print(f"\n✓ Métricas guardadas en: {metrics_csv}")
    print(f"✓ Matriz de confusión guardada en: {confusion_png}\n")
    return metrics_df


def evaluate(index_csv=CARDIAC_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_WINDOWS_DIR,
             checkpoint_dir=CARDIAC_CHECKPOINTS_DIR, reports_dir=CARDIAC_REPORTS_DIR,
             feature_type: str = "mfcc", n_mfcc: int = 20) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ds = CardiacDataset(index_csv, windows_dir, split="test",
                              feature_type=feature_type, n_mfcc=n_mfcc)
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
