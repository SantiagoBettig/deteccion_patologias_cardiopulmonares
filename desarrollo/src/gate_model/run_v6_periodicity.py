"""
gate_model/run_v6_periodicity.py
====================================
Gate v6: sobre v5 (MFCC, versión vigente), suma una rama de periodicidad
rítmica (gate_model/periodicity.py, gate_model/model_periodicity.py) como
hipótesis alternativa a los features de timbre para atacar la debilidad
persistente en cardíaco de la validación externa (HLS-CMDS). Ver
avances/7_investigacion_debilidad_cardiaca.md (motivación) y
avances/8_gate_v6_periodicity.md (resultados).

Self-contenido (no reutiliza train.py/evaluate.py de v1-v5): el modelo toma
dos inputs (espectrograma + vector de periodicidad) en vez de uno solo, así
que el loop de entrenamiento/evaluación es distinto.

Guarda checkpoint y reportes en carpetas separadas del gate vigente (v5),
para no pisarlas:
    gate_model/checkpoints_v6_periodicity/best.pt
    reports/gate/runs/v6_periodicity/

Uso:
    python gate_model/run_v6_periodicity.py
"""

import platform
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, GATE_HLS_INDEX_CSV, REPORTS_DIR,
)
from preprocessing.make_hls_windows import EXTERNAL_SPLIT_NAME
from gate_model.dataset_periodicity import GatePeriodicityDataset
from gate_model.model_periodicity import GateCNNPeriodicity

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_WORKERS = 0 if platform.system() == "Windows" else 2
FEATURE_TYPE = "mfcc"
N_MFCC = 20

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints_v6_periodicity"
REPORTS_DIR_V6 = REPORTS_DIR / "gate" / "runs" / "v6_periodicity"


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x_spec, x_period, y in loader:
            x_spec, x_period, y = x_spec.to(device), x_period.to(device), y.to(device)
            logits = model(x_spec, x_period)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x_spec.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    train_ds = GatePeriodicityDataset(GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, split="train",
                                       feature_type=FEATURE_TYPE, n_mfcc=N_MFCC, augment=True)
    val_ds = GatePeriodicityDataset(GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, split="val",
                                     feature_type=FEATURE_TYPE, n_mfcc=N_MFCC, augment=False)
    print(f"Ventanas — train: {len(train_ds)}, val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    n_pos = int((train_ds.df["sound_type"] == "cardiac").sum())
    n_neg = len(train_ds.df) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    print(f"Balance de train — positivos: {n_pos}, negativos: {n_neg}, pos_weight: {pos_weight.item():.3f}")

    model = GateCNNPeriodicity().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_f1 = -1.0
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

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
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best.pt")
            print(f"  ✓ Nuevo mejor checkpoint (val_f1={val_f1:.4f})")

    REPORTS_DIR_V6.mkdir(parents=True, exist_ok=True)
    history_df = pd.DataFrame(history)
    history_df.to_csv(REPORTS_DIR_V6 / "training_log.csv", index=False)

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
    fig.savefig(REPORTS_DIR_V6 / "training_curves.png", dpi=120)
    plt.close(fig)

    print(f"\n✓ Mejor val_f1: {best_val_f1:.4f}\n")


def _load_best_model(device) -> GateCNNPeriodicity:
    model = GateCNNPeriodicity().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_DIR / "best.pt", map_location=device))
    model.eval()
    return model


def _run_inference(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_spec, x_period, y in loader:
            x_spec, x_period = x_spec.to(device), x_period.to(device)
            logits = model(x_spec, x_period)
            preds = (torch.sigmoid(logits) > 0.5).float()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.tolist())
    return all_labels, all_preds


def _report(all_labels, all_preds, metrics_csv: Path, confusion_png: Path, title: str) -> dict:
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
    disp = ConfusionMatrixDisplay(cm, display_labels=["pulmonary", "cardiac"])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(confusion_png, dpi=120)
    plt.close(fig)

    print(f"\n✓ Métricas guardadas en: {metrics_csv}")
    print(f"✓ Matriz de confusión guardada en: {confusion_png}\n")
    return metrics


def evaluate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = GatePeriodicityDataset(GATE_WINDOWS_INDEX_CSV, GATE_WINDOWS_DIR, split="test",
                                      feature_type=FEATURE_TYPE, n_mfcc=N_MFCC)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Ventanas de test: {len(test_ds)}")

    model = _load_best_model(device)
    all_labels, all_preds = _run_inference(model, test_loader, device)
    _report(
        all_labels, all_preds,
        metrics_csv=REPORTS_DIR_V6 / "test_metrics.csv",
        confusion_png=REPORTS_DIR_V6 / "confusion_matrix.png",
        title="Gate v6 (MFCC + periodicidad) — matriz de confusión (test)",
    )


def evaluate_external_hls() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = GatePeriodicityDataset(GATE_HLS_INDEX_CSV, GATE_WINDOWS_DIR, split=EXTERNAL_SPLIT_NAME,
                                 feature_type=FEATURE_TYPE, n_mfcc=N_MFCC)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Ventanas HLS-CMDS (validación externa): {len(ds)}")

    model = _load_best_model(device)
    all_labels, all_preds = _run_inference(model, loader, device)
    _report(
        all_labels, all_preds,
        metrics_csv=REPORTS_DIR_V6 / "external_hls_metrics.csv",
        confusion_png=REPORTS_DIR_V6 / "external_hls_confusion_matrix.png",
        title="Gate v6 (MFCC + periodicidad) — validación externa (HLS-CMDS)",
    )


def main() -> None:
    print("=" * 55)
    print("  1/3 — Entrenamiento (gate v6: MFCC + periodicidad)")
    print("=" * 55)
    train()

    print("\n" + "=" * 55)
    print("  2/3 — Evaluación (test)")
    print("=" * 55)
    evaluate()

    print("\n" + "=" * 55)
    print("  3/3 — Validación externa (HLS-CMDS)")
    print("=" * 55)
    evaluate_external_hls()

    print("\n✓ Pipeline completo.")


if __name__ == "__main__":
    main()
