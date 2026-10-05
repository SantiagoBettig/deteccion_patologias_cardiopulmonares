"""
cardiac_murmur_model/evaluate_multimodal.py
===============================================
Evalúa la variante multimodal del modelo de murmullo (v4) sobre el split
de test — mismo procedimiento que evaluate.py (calibración de umbral sobre
validación, maximizando F2), adaptado para pasar también la metadata al
modelo.

Uso:
    python cardiac_murmur_model/evaluate_multimodal.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_MURMUR_V4_CHECKPOINTS_DIR, CARDIAC_MURMUR_V4_REPORTS_DIR
from cardiac_murmur_model.dataset import MurmurDataset, collate_pad_multimodal, compute_meta_stats, META_DIM
from cardiac_murmur_model.model import MurmurCNNMultimodal

BATCH_SIZE = 32


def load_best_model(device, checkpoint_dir=CARDIAC_MURMUR_V4_CHECKPOINTS_DIR) -> MurmurCNNMultimodal:
    model = MurmurCNNMultimodal(meta_dim=META_DIM).to(device)
    model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device))
    model.eval()
    return model


def run_inference_probs(model, loader, device) -> tuple[list, list]:
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, meta, y in loader:
            x, meta = x.to(device), meta.to(device)
            probs = torch.sigmoid(model(x, meta))
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_probs


def best_threshold_by_f2(labels, probs) -> float:
    best_t, best_f2 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        preds = [1.0 if p > t else 0.0 for p in probs]
        f2 = fbeta_score(labels, preds, beta=2.0, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t = f2, t
    return best_t


def report(all_labels, all_preds, metrics_csv: Path, confusion_png: Path, threshold: float,
           title: str = "Modelo de murmullo multimodal — matriz de confusión (test)") -> dict:
    metrics = {
        "threshold": threshold,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds),
        "recall": recall_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds),
        "f2": fbeta_score(all_labels, all_preds, beta=2.0),
    }

    print(f"\n── Métricas — {title} (umbral={threshold:.2f}) ──")
    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}" if name != "threshold" else f"  {name:<10}: {value:.2f}")

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["sin_murmullo", "murmullo"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{title}\numbral={threshold:.2f}")
    fig.tight_layout()
    fig.savefig(confusion_png, dpi=120)
    plt.close(fig)

    print(f"\n✓ Métricas guardadas en: {metrics_csv}")
    print(f"✓ Matriz de confusión guardada en: {confusion_png}\n")
    return metrics


def evaluate(index_csv=CARDIAC_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_WINDOWS_DIR,
             checkpoint_dir=CARDIAC_MURMUR_V4_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V4_REPORTS_DIR,
             feature_type: str = "mfcc", n_mfcc: int = 20) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(device, checkpoint_dir=checkpoint_dir)

    full_df = pd.read_csv(index_csv)
    meta_stats = compute_meta_stats(full_df[full_df["split"] == "train"])

    val_ds = MurmurDataset(index_csv, windows_dir, split="val", feature_type=feature_type, n_mfcc=n_mfcc,
                            include_metadata=True, meta_stats=meta_stats)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad_multimodal)
    val_labels, val_probs = run_inference_probs(model, val_loader, device)
    threshold = best_threshold_by_f2(val_labels, val_probs)
    print(f"Umbral calibrado sobre validación (máximo F2): {threshold:.2f}")

    test_ds = MurmurDataset(index_csv, windows_dir, split="test", feature_type=feature_type, n_mfcc=n_mfcc,
                             include_metadata=True, meta_stats=meta_stats)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad_multimodal)
    print(f"Ventanas de test: {len(test_ds)}")
    test_labels, test_probs = run_inference_probs(model, test_loader, device)

    report(
        test_labels, [1.0 if p > 0.5 else 0.0 for p in test_probs],
        metrics_csv=reports_dir / "test_metrics_threshold_0.5.csv",
        confusion_png=reports_dir / "confusion_matrix_threshold_0.5.png",
        threshold=0.5, title="Modelo de murmullo multimodal — matriz de confusión (test, umbral 0.5)",
    )
    report(
        test_labels, [1.0 if p > threshold else 0.0 for p in test_probs],
        metrics_csv=reports_dir / "test_metrics.csv",
        confusion_png=reports_dir / "confusion_matrix.png",
        threshold=threshold, title="Modelo de murmullo multimodal — matriz de confusión (test, umbral calibrado)",
    )


if __name__ == "__main__":
    evaluate()
