"""
cardiac_murmur_model/run_kfold.py
=====================================
Validación k-fold (K=5) de la versión ganadora del modelo de murmullo (v1:
MFCC + pos_weight x1.4 + checkpoint por F2 + dropout + umbral calibrado),
evaluada tanto a nivel ventana como a nivel sujeto (agregación por
máximo) — ver avances/10_cardiaco_split_dos_tareas.md, sección 14.

Motivación: con un solo split, val/test tienen ~27-36 sujetos con
murmullo — pocos como para confiar en una sola medición. Con k-fold, cada
uno de los ~179 sujetos con murmullo pasa por el rol de test exactamente
una vez a lo largo de los 5 folds, y se reporta media ± desvío en vez de
un solo número.

Requiere haber corrido antes:
    splits/make_cardiac_murmur_kfold.py (genera los folds)
    preprocessing/make_cardiac_cycle_windows.py (genera windows_index.csv)

Guarda:
    reports/cardiac_murmur/kfold/summary.csv (media ± desvío)
    reports/cardiac_murmur/kfold/per_fold.csv (resultado de cada fold)

Uso:
    python cardiac_murmur_model/run_kfold.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_REPORTS_DIR, UNIFIED_V2_DIR
from cardiac_murmur_model.dataset_kfold import KFoldMurmurDataset, collate_pad, build_window_path_index
from cardiac_murmur_model.model import MurmurCNN
from cardiac_murmur_model.train import _run_epoch, POS_WEIGHT_MULTIPLIER, LEARNING_RATE, EPOCHS, BATCH_SIZE
from cardiac_murmur_model.evaluate import run_inference_probs, best_threshold_by_f2

K = 5
KFOLD_DIR = UNIFIED_V2_DIR / "cardiac_murmur_kfold"
KFOLD_REPORTS_DIR = CARDIAC_REPORTS_DIR.parent / "cardiac_murmur" / "kfold"
KFOLD_CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints_kfold"


def _aggregate_subject_level(window_probs_df: pd.DataFrame) -> pd.DataFrame:
    return window_probs_df.groupby("subject_id").agg(
        prob_max=("prob", "max"),
        label=("pathology_label", lambda s: int((s == "heart_murmur").any())),
    ).reset_index()


def _metrics(labels, preds) -> dict:
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "f2": fbeta_score(labels, preds, beta=2.0, zero_division=0),
    }


def run_one_fold(fold_idx: int, windows_df: pd.DataFrame, path_index: dict, device) -> dict:
    fold_csv = KFOLD_DIR / f"fold{fold_idx}.csv"
    fold_splits = pd.read_csv(fold_csv)[["subject_id", "split"]]

    fold_df = windows_df.drop(columns=["split"]).merge(fold_splits, on="subject_id", how="inner")

    train_ds = KFoldMurmurDataset(fold_df, path_index, split="train", augment=True)
    val_ds = KFoldMurmurDataset(fold_df, path_index, split="val", augment=False)
    test_ds = KFoldMurmurDataset(fold_df, path_index, split="test", augment=False)
    print(f"  Ventanas — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

    n_pos = int((train_ds.df["pathology_label"] == "heart_murmur").sum())
    n_neg = len(train_ds.df) - n_pos
    pos_weight = torch.tensor(POS_WEIGHT_MULTIPLIER * n_neg / n_pos, dtype=torch.float32, device=device)

    model = MurmurCNN().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    KFOLD_CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = KFOLD_CHECKPOINTS_DIR / f"fold{fold_idx}_best.pt"
    best_val_f2 = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1, train_f2, train_recall = _run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1, val_f2, val_recall = _run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)
        print(f"    [{epoch:02d}/{EPOCHS}] train_loss={train_loss:.4f} | "
              f"val_f2={val_f2:.4f} val_recall={val_recall:.4f}")
        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    val_labels, val_probs = run_inference_probs(model, val_loader, device)
    threshold = best_threshold_by_f2(val_labels, val_probs)

    test_labels, test_probs = run_inference_probs(model, test_loader, device)
    window_preds = [1.0 if p > threshold else 0.0 for p in test_probs]
    window_metrics = _metrics(test_labels, window_preds)
    window_metrics["threshold"] = threshold

    subj_probs_df = test_ds.df[["subject_id", "pathology_label"]].copy()
    subj_probs_df["prob"] = test_probs
    subj_agg = _aggregate_subject_level(subj_probs_df)
    subj_preds = (subj_agg["prob_max"] > threshold).astype(int).tolist()
    subject_metrics = _metrics(subj_agg["label"].tolist(), subj_preds)
    subject_metrics["threshold"] = threshold

    print(f"  Fold {fold_idx} — ventana: f2={window_metrics['f2']:.4f} recall={window_metrics['recall']:.4f} | "
          f"sujeto: f1={subject_metrics['f1']:.4f} f2={subject_metrics['f2']:.4f}")

    return {"window": window_metrics, "subject": subject_metrics}


def run_kfold() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    windows_df = pd.read_csv(CARDIAC_WINDOWS_INDEX_CSV)
    path_index = build_window_path_index(CARDIAC_WINDOWS_DIR)
    print(f"Ventanas indexadas: {len(path_index)}")

    all_results = []
    for i in range(K):
        print(f"\n{'=' * 55}\n  FOLD {i}/{K - 1}\n{'=' * 55}")
        result = run_one_fold(i, windows_df, path_index, device)
        all_results.append(result)

    window_df = pd.DataFrame([r["window"] for r in all_results])
    subject_df = pd.DataFrame([r["subject"] for r in all_results])

    per_fold = pd.concat([
        window_df.add_prefix("window_"),
        subject_df.add_prefix("subject_"),
    ], axis=1)
    per_fold.insert(0, "fold", range(K))

    summary_rows = []
    for prefix, df in [("window", window_df), ("subject", subject_df)]:
        for metric in ["accuracy", "precision", "recall", "f1", "f2"]:
            summary_rows.append({
                "nivel": prefix, "metrica": metric,
                "media": df[metric].mean(), "std": df[metric].std(),
            })
    summary_df = pd.DataFrame(summary_rows)

    print(f"\n{'=' * 55}\n  RESUMEN K-FOLD (K={K})\n{'=' * 55}")
    print(summary_df.to_string(index=False))

    KFOLD_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(KFOLD_REPORTS_DIR / "per_fold.csv", index=False)
    summary_df.to_csv(KFOLD_REPORTS_DIR / "summary.csv", index=False)
    print(f"\n✓ Detalle por fold: {KFOLD_REPORTS_DIR / 'per_fold.csv'}")
    print(f"✓ Resumen: {KFOLD_REPORTS_DIR / 'summary.csv'}\n")


if __name__ == "__main__":
    run_kfold()
