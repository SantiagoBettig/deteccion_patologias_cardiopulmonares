"""
cardiac_murmur_model/train_multimodal.py
============================================
Entrena la variante multimodal del modelo de murmullo (v4): mismo audio +
receta que v1 (MFCC, pos_weight x1.4, checkpoint por F2, dropout), más la
metadata del paciente (sexo, edad, altura, peso) fusionada con el embedding
de audio antes de la capa final — ver
avances/10_cardiaco_split_dos_tareas.md, sección 8, y
cardiac_murmur_model/model.py:MurmurCNNMultimodal.

Archivo separado de train.py (no un flag ahí) para no complicar el
entrenamiento de audio-only que ya está congelado como v1/v3.

Uso:
    python cardiac_murmur_model/train_multimodal.py
"""

import platform

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, fbeta_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV, CARDIAC_WINDOWS_DIR, CARDIAC_MURMUR_V4_CHECKPOINTS_DIR, CARDIAC_MURMUR_V4_REPORTS_DIR
from cardiac_murmur_model.dataset import MurmurDataset, collate_pad_multimodal, compute_meta_stats, META_DIM
from cardiac_murmur_model.model import MurmurCNNMultimodal

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0 if platform.system() == "Windows" else 2
N_MFCC = 20
POS_WEIGHT_MULTIPLIER = 1.4


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, meta, y in loader:
            x, meta, y = x.to(device), meta.to(device), y.to(device)
            logits = model(x, meta)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    f2 = fbeta_score(all_labels, all_preds, beta=2.0)
    recall = recall_score(all_labels, all_preds)
    return avg_loss, acc, f1, f2, recall


def train(index_csv=CARDIAC_WINDOWS_INDEX_CSV, windows_dir=CARDIAC_WINDOWS_DIR,
          checkpoint_dir=CARDIAC_MURMUR_V4_CHECKPOINTS_DIR, reports_dir=CARDIAC_MURMUR_V4_REPORTS_DIR,
          feature_type: str = "mfcc", n_mfcc: int = N_MFCC) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # meta_stats se calcula SOLO sobre train, y se reusa igual en val/test —
    # para no filtrar estadísticas de val/test al entrenamiento.
    full_df = pd.read_csv(index_csv)
    meta_stats = compute_meta_stats(full_df[full_df["split"] == "train"])
    print(f"Meta stats (de train) — altura: {meta_stats['height_mean']:.1f}±{meta_stats['height_std']:.1f} cm, "
          f"peso: {meta_stats['weight_mean']:.1f}±{meta_stats['weight_std']:.1f} kg")

    train_ds = MurmurDataset(index_csv, windows_dir, split="train",
                              feature_type=feature_type, n_mfcc=n_mfcc, augment=True,
                              include_metadata=True, meta_stats=meta_stats)
    val_ds = MurmurDataset(index_csv, windows_dir, split="val",
                            feature_type=feature_type, n_mfcc=n_mfcc, augment=False,
                            include_metadata=True, meta_stats=meta_stats)
    print(f"Ventanas — train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, collate_fn=collate_pad_multimodal)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_pad_multimodal)

    n_pos = int((train_ds.df["pathology_label"] == "heart_murmur").sum())
    n_neg = len(train_ds.df) - n_pos
    pos_weight = torch.tensor(POS_WEIGHT_MULTIPLIER * n_neg / n_pos, dtype=torch.float32, device=device)
    print(f"Balance de train — murmullo: {n_pos}, sin murmullo: {n_neg}, "
          f"pos_weight: {pos_weight.item():.3f} (x{POS_WEIGHT_MULTIPLIER} sobre el natural)")

    model = MurmurCNNMultimodal(meta_dim=META_DIM).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_f2 = -1.0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1, train_f2, train_recall = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1, val_f2, val_recall = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} val_f2={val_f2:.4f} val_recall={val_recall:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1, "train_f2": train_f2, "train_recall": train_recall,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1, "val_f2": val_f2, "val_recall": val_recall,
        })

        if val_f2 > best_val_f2:
            best_val_f2 = val_f2
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  ✓ Nuevo mejor checkpoint (val_f2={val_f2:.4f})")

    reports_dir.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(reports_dir / "training_log.csv", index=False)

    fig, (ax_loss, ax_metric) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history_df["epoch"], history_df["train_loss"], label="train")
    ax_loss.plot(history_df["epoch"], history_df["val_loss"], label="val")
    ax_loss.set_title("Loss por época")
    ax_loss.set_xlabel("Época")
    ax_loss.legend()

    ax_metric.plot(history_df["epoch"], history_df["val_f1"], label="val F1")
    ax_metric.plot(history_df["epoch"], history_df["val_f2"], label="val F2")
    ax_metric.plot(history_df["epoch"], history_df["val_recall"], label="val recall", linestyle="--")
    ax_metric.set_title("F1 / F2 / recall por época (val)")
    ax_metric.set_xlabel("Época")
    ax_metric.legend()

    fig.tight_layout()
    fig.savefig(reports_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    print(f"\n✓ Mejor val_f2: {best_val_f2:.4f}")
    print(f"✓ Log de entrenamiento: {reports_dir / 'training_log.csv'}")
    print(f"✓ Curvas de entrenamiento: {reports_dir / 'training_curves.png'}\n")

    return meta_stats


if __name__ == "__main__":
    train()
