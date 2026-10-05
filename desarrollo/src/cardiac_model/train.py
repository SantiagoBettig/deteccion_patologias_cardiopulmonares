"""
cardiac_model/train.py
=========================
Entrena el modelo de patología cardíaca (3 clases: normal_heart,
abnormal_heart_unspecified, heart_murmur) sobre las ventanas por ciclo
generadas por preprocessing/make_cardiac_cycle_windows.py.

Arranca directo con la receta que ganó en el gate (v5, ver
avances/6_gate_v5_mfcc.md): MFCC + normalización z-score + augmentation de
audio (gain+ruido) + class weighting, sin SpecAugment (empeoró en el gate
v3, avances/4_gate_model_baseline_y_sesgo.md sección 8 — se deja
parametrizado en CardiacDataset pero apagado por default).

Guarda:
    - El checkpoint con mejor F1 macro de validación (cardiac_model/checkpoints/best.pt)
    - Un log de métricas por época (reports/cardiac/training_log.csv)
    - Un gráfico de la evolución de esas métricas (reports/cardiac/training_curves.png)

Uso:
    python cardiac_model/train.py
"""

import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_CHECKPOINTS_DIR, CARDIAC_REPORTS_DIR
from cardiac_model.dataset import CardiacDataset, collate_pad, CLASSES
from cardiac_model.model import CardiacCNN

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0 if platform.system() == "Windows" else 2  # 0 en Windows evita overhead de multiprocessing
N_MFCC = 20


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    recall_per_class = recall_score(all_labels, all_preds, average=None, labels=range(len(CLASSES)))
    return avg_loss, f1_macro, recall_per_class


def train(index_csv=CARDIAC_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_WINDOWS_DIR,
          checkpoint_dir=CARDIAC_CHECKPOINTS_DIR, reports_dir=CARDIAC_REPORTS_DIR,
          feature_type: str = "mfcc", n_mfcc: int = N_MFCC) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_ds = CardiacDataset(index_csv, windows_dir, split="train",
                               feature_type=feature_type, n_mfcc=n_mfcc, augment=True)
    val_ds = CardiacDataset(index_csv, windows_dir, split="val",
                             feature_type=feature_type, n_mfcc=n_mfcc, augment=False)
    print(f"Ventanas — train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_pad)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_pad)

    # Class weighting sobre el desbalance de train (normal_heart es ~2.7x
    # heart_murmur) — inversamente proporcional a la frecuencia de cada clase.
    counts = train_ds.df["pathology_label"].value_counts()
    class_counts = np.array([counts.get(c, 0) for c in CLASSES], dtype=np.float32)
    class_weights = torch.tensor(class_counts.sum() / class_counts, dtype=torch.float32, device=device)
    print("Balance de train:")
    for cls, n, w in zip(CLASSES, class_counts, class_weights.cpu().tolist()):
        print(f"  {cls:<28}: {int(n):>6} ventanas (weight={w:.3f})")

    model = CardiacCNN(n_classes=len(CLASSES)).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_f1 = -1.0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1, train_recall = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_f1, val_recall = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        recall_str = ", ".join(f"{c}={r:.3f}" for c, r in zip(CLASSES, val_recall))
        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} | val_recall: {recall_str}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_f1": train_f1,
            "val_loss": val_loss, "val_f1": val_f1,
            **{f"val_recall_{c}": r for c, r in zip(CLASSES, val_recall)},
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  ✓ Nuevo mejor checkpoint (val_f1_macro={val_f1:.4f})")

    # ── Log y gráfico de métricas ────────────────────────────────────────────
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(reports_dir / "training_log.csv", index=False)

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history_df["epoch"], history_df["train_loss"], label="train")
    ax_loss.plot(history_df["epoch"], history_df["val_loss"], label="val")
    ax_loss.set_title("Loss por época")
    ax_loss.set_xlabel("Época")
    ax_loss.legend()

    ax_metric.plot(history_df["epoch"], history_df["train_f1"], label="train F1 macro")
    ax_metric.plot(history_df["epoch"], history_df["val_f1"], label="val F1 macro")
    for c in CLASSES:
        ax_metric.plot(history_df["epoch"], history_df[f"val_recall_{c}"], label=f"val recall {c}", linestyle="--")
    ax_metric.set_title("F1 macro / recall por clase")
    ax_metric.set_xlabel("Época")
    ax_metric.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(reports_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    print(f"\n✓ Mejor val_f1_macro: {best_val_f1:.4f}")
    print(f"✓ Log de entrenamiento: {reports_dir / 'training_log.csv'}")
    print(f"✓ Curvas de entrenamiento: {reports_dir / 'training_curves.png'}\n")


if __name__ == "__main__":
    train()
