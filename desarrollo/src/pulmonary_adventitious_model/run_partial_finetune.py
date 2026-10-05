"""
pulmonary_adventitious_model/run_partial_finetune.py
=======================================================
Fine-tuning PARCIAL de la CNN6 de PANNs en CPU (ver avances/17). Pregunta
que responde: con ~85 pacientes de train por fold, ¿adaptar aunque sea la
última capa convolucional de la red preentrenada mejora sobre usarla
congelada (avances/15: 0.559 ± 0.024), o solo sobreajusta?

Qué se entrena y qué no:
    congelado  : bn0 + bloques 1-3 — su salida ya está calculada una vez
                 (extract_cnn6_block3.py), no se recalcula por época.
    entrenable : bloque 4 (conv 256->512, pesos preentrenados) + fc1
                 (512->512, preentrenada) + cabeza nueva (512->2).

Decisiones para pocos datos:
- learning rate bajo para lo preentrenado (1e-4) y normal para la cabeza
  nueva (1e-3): los filtros preentrenados se mueven poco.
- BatchNorm del bloque 4 con estadísticas congeladas (las de AudioSet);
  solo se entrenan su escala y desplazamiento. Con batches chicos y
  padding, recalcular esas estadísticas las distorsionaría.
- dropout de PANNs (0.2 después del bloque, 0.5 antes de fc1 y de la
  cabeza) y enmascarado temporal sobre la salida del bloque 3 como
  augmentation (no se puede augmentar el audio: los bloques 1-3 ya están
  calculados).
- pooling enmascarado: los ciclos tienen duración distinta y se rellenan
  con ceros dentro del batch; el máximo y el promedio temporal se calculan
  solo sobre los pasos reales.

Mismo protocolo que v4 y la etapa 1: 5 folds por paciente, pos_weight
natural, mejor época por AUC medio de val, umbrales calibrados en val,
predicciones OOF y desglose por equipo y edad.

Uso:
    python pulmonary_adventitious_model/run_partial_finetune.py
    python pulmonary_adventitious_model/run_partial_finetune.py --folds 0 --epochs 2   # prueba rápida
    python pulmonary_adventitious_model/run_partial_finetune.py --resume   # retoma una corrida cortada

--resume: los folds que ya terminaron (checkpoint + training_log_fold{i}.csv
con todas las épocas) no se reentrenan; se recalculan sus predicciones de
val/test desde el checkpoint (la evaluación es determinística, así que da
exactamente lo mismo). Un fold cortado a mitad retoma desde la última época
terminada: al final de cada época se guarda el estado completo (modelo,
optimizador, historial, generadores aleatorios) en fold{i}_last.pt.
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    PANNS_CNN6_CKPT, PULMONARY_KFOLD_DIR, PULMONARY_ADV_CHECKPOINTS_DIR, PULMONARY_ADV_REPORTS_DIR,
    pulmonary_cycles_paths,
)
from pulmonary_adventitious_model.dataset import LABEL_COLUMNS
from pulmonary_adventitious_model.extract_cnn6_block3 import block3_paths
from pulmonary_adventitious_model.extract_cnn6_embeddings import CYCLES_VARIANT
from pulmonary_adventitious_model.metrics import CLASS_NAMES, calibrate_thresholds, full_metrics, predict_4class, to_4class
from pulmonary_adventitious_model.panns_cnn6 import load_pretrained_cnn6
from pulmonary_adventitious_model.run_frozen_probe import mean_auc
from pulmonary_adventitious_model.run_kfold import K, SUMMARY_METRICS, _age_group, _group_metrics

RUN_NAME = "v7_cnn6_partial_ft"
EPOCHS = 30
BATCH_SIZE = 64
LR_PRETRAINED = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
MAX_STEPS = 50        # pasos de tiempo del bloque 3 (~12.5/s) -> 4 s, igual que v4
TIME_MASK_MAX = 5     # augmentation: hasta 5 pasos (~0.4 s) puestos a 0, solo en train


class Block3Dataset(Dataset):
    def __init__(self, feats: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
                 labels: np.ndarray, augment: bool):
        self.feats, self.starts, self.lengths = feats, starts, lengths
        self.labels, self.augment = labels, augment

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i):
        start, length = int(self.starts[i]), int(self.lengths[i])
        if length > MAX_STEPS:
            offset = (np.random.randint(0, length - MAX_STEPS + 1) if self.augment
                      else (length - MAX_STEPS) // 2)
            start, length = start + offset, MAX_STEPS
        x = torch.from_numpy(self.feats[start:start + length].astype(np.float32)).permute(1, 0, 2)  # (256, t, 8)
        if self.augment and length > 2 * TIME_MASK_MAX:
            width = np.random.randint(0, TIME_MASK_MAX + 1)
            t0 = np.random.randint(0, length - width + 1)
            x[:, t0:t0 + width, :] = 0.0
        return x, length, torch.from_numpy(self.labels[i])


def collate(batch):
    xs, lengths, ys = zip(*batch)
    max_t = max(lengths)
    x = torch.stack([F.pad(xi, (0, 0, 0, max_t - xi.shape[1])) for xi in xs])
    return x, torch.tensor(lengths), torch.stack(ys)


class PartialCnn6(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        self.conv_block4 = pretrained.conv_block4
        self.fc1 = pretrained.fc1
        self.head = nn.Linear(512, 2)

    def train(self, mode: bool = True):
        super().train(mode)
        self.conv_block4.bn1.eval()   # estadísticas de AudioSet congeladas (ver docstring)
        return self

    def forward(self, x, lengths):
        x = F.dropout(self.conv_block4(x), p=0.2, training=self.training)   # (B, 512, t/2, 4)
        x = x.mean(dim=3)                                                   # (B, 512, t/2)
        valid = torch.clamp(lengths // 2, min=1)
        mask = torch.arange(x.shape[2])[None, :] < valid[:, None]           # (B, t/2)
        x_max = x.masked_fill(~mask[:, None, :], float("-inf")).amax(dim=2)
        x_mean = (x * mask[:, None, :]).sum(dim=2) / valid[:, None]
        x = F.dropout(x_max + x_mean, p=0.5, training=self.training)
        x = F.relu(self.fc1(x))
        return self.head(F.dropout(x, p=0.5, training=self.training))


def predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            probs.append(torch.sigmoid(model(x, lengths)).numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(probs)


def run(folds: list[int], epochs: int, resume: bool = False) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_flush_denormal(True)

    cycles = pd.read_csv(pulmonary_cycles_paths(CYCLES_VARIANT)[1])
    feats_npy, index_csv = block3_paths()
    feats = np.load(feats_npy)
    index = pd.read_csv(index_csv)
    assert (index["cycle_id"].to_numpy() == cycles["cycle_id"].to_numpy()).all(), "orden distinto al índice"
    starts, lengths = index["start"].to_numpy(), index["length"].to_numpy()
    y_all = cycles[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    print(f"Salida del bloque 3: {feats.shape} ({feats.nbytes / 1e9:.2f} GB en memoria)")

    reports_dir = PULMONARY_ADV_REPORTS_DIR / RUN_NAME
    checkpoints_dir = PULMONARY_ADV_CHECKPOINTS_DIR / RUN_NAME
    reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    per_fold, oof_parts, histories = [], [], []
    for i in folds:
        split = cycles[["subject_id"]].merge(
            pd.read_csv(PULMONARY_KFOLD_DIR / f"fold{i}.csv"), on="subject_id", how="left")["split"].to_numpy()
        masks = {s: split == s for s in ("train", "val", "test")}
        loaders = {
            s: DataLoader(Block3Dataset(feats, starts[m], lengths[m], y_all[m], augment=(s == "train")),
                          batch_size=BATCH_SIZE, shuffle=(s == "train"), collate_fn=collate)
            for s, m in masks.items()
        }

        model = PartialCnn6(load_pretrained_cnn6(PANNS_CNN6_CKPT))
        y_tr = y_all[masks["train"]]
        pos_weight = torch.tensor((len(y_tr) - y_tr.sum(0)) / y_tr.sum(0), dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam([
            {"params": list(model.conv_block4.parameters()) + list(model.fc1.parameters()), "lr": LR_PRETRAINED},
            {"params": model.head.parameters(), "lr": LR_HEAD},
        ], weight_decay=WEIGHT_DECAY)

        print(f"\n{'=' * 60}\n  FOLD {i} ({RUN_NAME}) — ciclos train {masks['train'].sum()}, "
              f"val {masks['val'].sum()}, test {masks['test'].sum()}\n{'=' * 60}")
        ckpt = checkpoints_dir / f"fold{i}_best.pt"
        log_csv = reports_dir / f"training_log_fold{i}.csv"
        done = resume and ckpt.exists() and log_csv.exists() and len(pd.read_csv(log_csv)) == epochs
        last_pt = checkpoints_dir / f"fold{i}_last.pt"
        best_auc, history, start_epoch = -1.0, [], 1
        if resume and not done and last_pt.exists():
            state = torch.load(last_pt, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            best_auc, history, start_epoch = state["best_auc"], state["history"], state["epoch"] + 1
            np.random.set_state(state["np_rng"])
            torch.set_rng_state(state["torch_rng"])
            print(f"  [fold {i}] retomando desde la época {start_epoch}")
        if done:
            history = pd.read_csv(log_csv).to_dict("records")
            best_auc = max(r["val_mean_auc"] for r in history)
            print(f"  [fold {i}] completo en una corrida anterior — se evalúa desde el checkpoint")
        for epoch in range(start_epoch, (0 if done else epochs) + 1):
            t0 = time.perf_counter()
            model.train()
            total = 0.0
            for x, lens, y in loaders["train"]:
                loss = criterion(model(x, lens), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item() * len(y)
            train_loss = total / masks["train"].sum()
            y_va, p_va = predict(model, loaders["val"])
            val_auc = mean_auc(y_va, p_va)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_mean_auc": val_auc,
                            "seconds": time.perf_counter() - t0})
            mark = ""
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), ckpt)
                mark = "  ✓ mejor"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "best_auc": best_auc, "history": history, "np_rng": np.random.get_state(),
                        "torch_rng": torch.get_rng_state()}, last_pt)
            print(f"  [fold {i}] [{epoch:02d}/{epochs}] train_loss={train_loss:.4f} val_AUC={val_auc:.4f} "
                  f"({history[-1]['seconds']:.0f} s){mark}", flush=True)
        h = pd.DataFrame(history)
        h.to_csv(log_csv, index=False)
        last_pt.unlink(missing_ok=True)   # fold terminado: el estado intermedio ya no hace falta
        histories.append((i, h))

        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        y_va, p_va = predict(model, loaders["val"])
        thresholds, val_score = calibrate_thresholds(y_va, p_va)
        y_te, p_te = predict(model, loaders["test"])
        m = full_metrics(y_te, p_te, thresholds)
        m.update({"fold": i, "best_epoch": int(h.loc[h.val_mean_auc.idxmax(), "epoch"]),
                  "val_mean_auc": best_auc, "val_icbhi_score": val_score})
        per_fold.append(m)
        print(f"  Fold {i} — TEST score={m['icbhi_score']:.4f} (Se={m['se']:.3f} Sp={m['sp']:.3f}) | "
              f"AUC crackle={m['crackle_auc']:.3f} wheeze={m['wheeze_auc']:.3f} | mejor época {m['best_epoch']}")

        part = cycles.loc[masks["test"], ["cycle_id", "subject_id", "pathology_label", "equipment",
                                          "age_years", "crackle", "wheeze"]].copy()
        part["fold"] = i
        part["p_crackle"], part["p_wheeze"] = p_te[:, 0], p_te[:, 1]
        part["pred_4class"] = predict_4class(p_te, thresholds)
        oof_parts.append(part)

    # ── Reportes ──────────────────────────────────────────────────────────────
    per_fold_df = pd.DataFrame(per_fold)
    per_fold_df.to_csv(reports_dir / "per_fold.csv", index=False)
    summary = pd.DataFrame([{"metrica": k, "media": per_fold_df[k].mean(), "std": per_fold_df[k].std()}
                            for k in SUMMARY_METRICS])
    summary.to_csv(reports_dir / "summary.csv", index=False)
    oof = pd.concat(oof_parts, ignore_index=True)
    oof["true_4class"] = to_4class(oof["crackle"].to_numpy(), oof["wheeze"].to_numpy())
    oof["age_group"] = oof["age_years"].map(_age_group)
    oof.to_csv(reports_dir / "oof_predictions.csv", index=False)
    by_equipment, by_age = _group_metrics(oof, "equipment"), _group_metrics(oof, "age_group")
    by_equipment.to_csv(reports_dir / "metrics_by_equipment.csv", index=False)
    by_age.to_csv(reports_dir / "metrics_by_age_group.csv", index=False)

    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4))
    for i, h in histories:
        ax_loss.plot(h["epoch"], h["train_loss"], color=f"C{i}", label=f"fold {i}")
        ax_auc.plot(h["epoch"], h["val_mean_auc"], color=f"C{i}", label=f"fold {i}")
    ax_loss.set_title("Loss de train por época")
    ax_auc.set_title("AUC medio de validación por época")
    for ax in (ax_loss, ax_auc):
        ax.set_xlabel("Época")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    pd.set_option("display.float_format", "{:.3f}".format)
    cm = pd.crosstab(oof["true_4class"].map(dict(enumerate(CLASS_NAMES))),
                     oof["pred_4class"].map(dict(enumerate(CLASS_NAMES))))
    print(f"\n{'=' * 60}\n  RESUMEN {RUN_NAME} (folds={folds})\n{'=' * 60}")
    print(summary.to_string(index=False))
    print(cm.to_string())
    print(by_equipment[["equipment", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
    print(by_age[["age_group", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
    print(f"\n✓ Reportes en: {reports_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--resume", action="store_true",
                        help="No reentrena los folds ya completos (ver docstring)")
    args = parser.parse_args()
    run(args.folds, args.epochs, args.resume)
