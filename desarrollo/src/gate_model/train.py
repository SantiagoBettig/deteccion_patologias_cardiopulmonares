"""
gate_model/train.py
=====================
Entrena el modelo gate (cardíaco vs. pulmonar) sobre las ventanas generadas
por preprocessing/make_gate_windows.py.

Guarda:
    - El checkpoint con mejor F1 de validación (gate_model/checkpoints/best.pt)
    - Un log de métricas por época (reports/gate/training_log.csv)
    - Un gráfico de la evolución de esas métricas (reports/gate/training_curves.png)

Uso:
    python gate_model/train.py
"""

import platform

import matplotlib.pyplot as plt
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, GATE_CHECKPOINTS_DIR, GATE_REPORTS_DIR
from gate_model.dataset import GateDataset
from gate_model.model import GateCNN

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0 if platform.system() == "Windows" else 2  # 0 en Windows evita overhead de multiprocessing


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
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


def train(index_csv=GATE_WINDOWS_INDEX_CSV, windows_dir=GATE_WINDOWS_DIR,
          label_column: str = "sound_type", positive_value: str = "cardiac",
          checkpoint_dir=GATE_CHECKPOINTS_DIR, reports_dir=GATE_REPORTS_DIR,
          feature_type: str = "logmel", n_mfcc: int = 20) -> None:
    """
    Parametrizado por label_column/positive_value para poder reutilizar este
    mismo loop de entrenamiento en el diagnóstico de sesgo por source_db
    (ver gate_model/diagnostics_source_db.py), sin duplicar el loop.

    feature_type/n_mfcc se pasan directo a GateDataset (ver su docstring) —
    permiten reutilizar este mismo loop para el gate v5 (MFCC en vez de
    espectrograma log-Mel, ver gate_model/run_v5_mfcc.py).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_ds = GateDataset(index_csv, windows_dir, split="train",
                            label_column=label_column, positive_value=positive_value,
                            feature_type=feature_type, n_mfcc=n_mfcc,
                            augment=True)
    val_ds = GateDataset(index_csv, windows_dir, split="val",
                          label_column=label_column, positive_value=positive_value,
                          feature_type=feature_type, n_mfcc=n_mfcc,
                          augment=False)
    print(f"Ventanas — train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # pos_weight corrige el desbalance de clases del split de train (ver
    # avances/4_gate_model_baseline_y_sesgo.md, sección 7): downweightea la
    # clase positiva cuando es la mayoritaria (ej. cardiac ~3.5x pulmonary),
    # para que la loss no favorezca a la clase más representada.
    n_pos = int((train_ds.df[label_column] == positive_value).sum())
    n_neg = len(train_ds.df) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    print(f"Balance de train — positivos: {n_pos}, negativos: {n_neg}, pos_weight: {pos_weight.item():.3f}")

    model = GateCNN().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_f1 = -1.0

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, val_f1 = _run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"[{epoch:02d}/{EPOCHS}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            print(f"  ✓ Nuevo mejor checkpoint (val_f1={val_f1:.4f})")

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

    ax_metric.plot(history_df["epoch"], history_df["train_acc"], label="train acc")
    ax_metric.plot(history_df["epoch"], history_df["val_acc"], label="val acc")
    ax_metric.plot(history_df["epoch"], history_df["train_f1"], label="train F1", linestyle="--")
    ax_metric.plot(history_df["epoch"], history_df["val_f1"], label="val F1", linestyle="--")
    ax_metric.set_title("Accuracy / F1 por época")
    ax_metric.set_xlabel("Época")
    ax_metric.legend()

    fig.tight_layout()
    fig.savefig(reports_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    print(f"\n✓ Mejor val_f1: {best_val_f1:.4f}")
    print(f"✓ Log de entrenamiento: {reports_dir / 'training_log.csv'}")
    print(f"✓ Curvas de entrenamiento: {reports_dir / 'training_curves.png'}\n")


if __name__ == "__main__":
    train()
