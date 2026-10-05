"""
cardiac_murmur_model/evaluate_subject_level.py
==================================================
Evalúa un checkpoint agregando las predicciones de TODAS las ventanas de un
mismo sujeto (todas sus ubicaciones de auscultación y ciclos) antes de
decidir, en vez de evaluar ventana por ventana — más fiel a cómo se usaría
esto en la práctica (un clínico ausculta varios puntos, no un clip de 2
segundos) y reduce la varianza de una ventana individual ruidosa (ver
avances/10_cardiaco_split_dos_tareas.md, sección 9).

Agregación: máximo de las probabilidades del sujeto (si CUALQUIER ventana
sugiere murmullo, se marca al sujeto — coherente con priorizar recall) y
media, para comparar. Ground truth por sujeto: 1 si el sujeto tiene AL
MENOS una ventana con heart_murmur (mismo criterio que
splits/split_utils.py:pick_subject_pathology).

Reusa el umbral ya calibrado a nivel ventana (ver evaluate.py) — funciona
sobre las probabilidades agregadas de la misma forma.

Uso:
    python cardiac_murmur_model/evaluate_subject_level.py
"""

import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_MURMUR_CHECKPOINTS_DIR, CARDIAC_MURMUR_REPORTS_DIR
from cardiac_murmur_model.dataset import MurmurDataset, collate_pad
from cardiac_murmur_model.evaluate import load_best_model, best_threshold_by_f2

BATCH_SIZE = 32


def _probs_with_ids(model, ds, loader, device) -> pd.DataFrame:
    all_probs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            probs = torch.sigmoid(model(x))
            all_probs.extend(probs.cpu().tolist())
    out = ds.df[["subject_id", "pathology_label"]].copy()
    out["prob"] = all_probs
    return out


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("subject_id").agg(
        prob_max=("prob", "max"),
        prob_mean=("prob", "mean"),
        label=("pathology_label", lambda s: int((s == "heart_murmur").any())),
        n_windows=("prob", "count"),
    ).reset_index()
    return agg


def _report(labels, preds, agg_name: str, threshold: float) -> dict:
    metrics = {
        "agregacion": agg_name,
        "threshold": threshold,
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "f2": fbeta_score(labels, preds, beta=2.0, zero_division=0),
    }
    print(f"  [{agg_name}] accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
          f"recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} f2={metrics['f2']:.4f}")
    return metrics


def evaluate_subject_level(index_csv=CARDIAC_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_WINDOWS_DIR,
                            checkpoint_dir=CARDIAC_MURMUR_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_REPORTS_DIR,
                            feature_type: str = "mfcc", n_mfcc: int = 20) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_best_model(device, checkpoint_dir=checkpoint_dir)

    val_ds = MurmurDataset(index_csv, windows_dir, split="val", feature_type=feature_type, n_mfcc=n_mfcc)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)
    val_window_probs = _probs_with_ids(model, val_ds, val_loader, device)
    val_agg = _aggregate(val_window_probs)

    test_ds = MurmurDataset(index_csv, windows_dir, split="test", feature_type=feature_type, n_mfcc=n_mfcc)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)
    test_window_probs = _probs_with_ids(model, test_ds, test_loader, device)
    test_agg = _aggregate(test_window_probs)

    print(f"Sujetos — val: {len(val_agg)}, test: {len(test_agg)}")

    results = []
    for agg_col, agg_name in [("prob_max", "max"), ("prob_mean", "mean")]:
        threshold = best_threshold_by_f2(val_agg["label"].tolist(), val_agg[agg_col].tolist())
        preds = (test_agg[agg_col] > threshold).astype(int).tolist()
        print(f"\nAgregación por sujeto — {agg_name} (umbral calibrado sobre validación: {threshold:.2f})")
        results.append(_report(test_agg["label"].tolist(), preds, agg_name, threshold))

    results_df = pd.DataFrame(results)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(reports_dir / "test_metrics_subject_level.csv", index=False)
    print(f"\n✓ Métricas guardadas en: {reports_dir / 'test_metrics_subject_level.csv'}\n")
    return results_df


if __name__ == "__main__":
    evaluate_subject_level()
