"""
pulmonary_adventitious_model/run_finetune_gpu.py
===================================================
Fine-tuning de la CNN6 de PANNs pensado para correr en GPU (Google Colab,
notebook colab_pulmonary_finetune.ipynb), con la red completa en el loop —
a diferencia de run_partial_finetune.py, que precalcula los bloques
congelados para poder correr en CPU. Ver avances/17.

Lee el paquete generado por pack_for_colab.py (espectrogramas log-Mel de
PANNs ya calculados + índice + folds + pesos), no los WAV: así no hay
cuello de botella de lectura (avances/5) y toda la augmentation se hace en
la GPU sobre el espectrograma.

Variantes (--variant), de menos a más ambiciosa:

    A: entrena bloque 4 + fc1, 60 épocas, solo enmascarado temporal.
       Equivale al fine-tuning parcial de CPU (v7) con el doble de épocas:
       mide cuánto le faltaba a esa corrida (su mejor época caía en 27-30).
    B: entrena bloques 3-4 + fc1, 40 épocas, augmentation completa
       (ganancia + SpecAugment en tiempo y frecuencia).
    C: entrena toda la red (bloques 1-4 + fc1), 30 épocas, learning rate
       más bajo, augmentation completa. Es el techo, con el mayor riesgo de
       sobreajuste.

Común a todas (mismas decisiones que v7, run_partial_finetune.py):
- cabeza nueva 512 -> 2; lr 1e-3 para la cabeza, más bajo para lo
  preentrenado;
- BatchNorm con estadísticas de AudioSet congeladas en toda la red (solo se
  entrenan escala y desplazamiento): batches chicos + padding las
  distorsionarían;
- ciclos recortados a 4 s (al azar en train, centrado en eval) y rellenados
  con silencio (-100 dB); pooling final enmascarado sobre los frames reales;
- dropout de PANNs solo en los bloques que se entrenan;
- mismo protocolo que todo el modelo pulmonar: 5 folds por paciente,
  pos_weight natural, mejor época por AUC medio de val, umbrales calibrados
  en val, predicciones OOF y desglose por equipo y edad.

Pensado para Colab gratuito, que corta sesiones: guarda el estado completo
(modelo + optimizador + época) al final de cada época en --out-dir (Drive),
y los resultados de cada fold apenas termina. Si la sesión se corta, volver
a correr el mismo comando retoma desde la última época guardada.

Uso:
    python pulmonary_adventitious_model/run_finetune_gpu.py --package-dir <carpeta> --out-dir <carpeta> --variant A
    # prueba rápida en CPU, con un subconjunto de ciclos:
    python pulmonary_adventitious_model/run_finetune_gpu.py --package-dir ... --out-dir ... --variant C --folds 0 --epochs 1 --limit 300
"""

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulmonary_adventitious_model.metrics import (
    CLASS_NAMES, calibrate_thresholds, full_metrics, predict_4class, to_4class,
)
from pulmonary_adventitious_model.panns_cnn6 import load_pretrained_cnn6
from pulmonary_adventitious_model.run_frozen_probe import mean_auc
from pulmonary_adventitious_model.run_kfold import K, SUMMARY_METRICS, _age_group, _group_metrics

LABEL_COLUMNS = ["crackle", "wheeze"]
BLOCKS = ["conv_block1", "conv_block2", "conv_block3", "conv_block4"]
VARIANTS = {
    "A": {"first_trainable": 3, "epochs": 60, "lr_pretrained": 1e-4, "full_augment": False},
    "B": {"first_trainable": 2, "epochs": 40, "lr_pretrained": 1e-4, "full_augment": True},
    "C": {"first_trainable": 0, "epochs": 30, "lr_pretrained": 3e-5, "full_augment": True},
}
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 64
MAX_FRAMES = 400        # 4 s a 100 frames/s, igual que v4 y v7
PAD_DB = -100.0         # silencio, el mínimo del log-Mel (10·log10(1e-10))
TIME_MASK_MAX = 40      # ~0.4 s (equivale a los 5 pasos del bloque 3 de v7)
FREQ_MASK_MAX = 6       # bandas Mel, solo dentro de las 30 bandas con contenido (< 2 kHz)
N_BANDS_WITH_CONTENT = 30
GAIN_DB = 6.0           # ganancia aleatoria ± 6 dB (corrimiento aditivo en dB)


# ── Datos ─────────────────────────────────────────────────────────────────────
class LogmelBank:
    """Todos los espectrogramas en memoria (CPU); arma batches recortados y
    rellenados, y los manda al dispositivo."""

    def __init__(self, feats: np.ndarray, starts: np.ndarray, lengths: np.ndarray, device):
        self.feats, self.starts, self.lengths, self.device = feats, starts, lengths, device

    def batch(self, idx: np.ndarray, train: bool):
        crops, lens = [], []
        for i in idx:
            start, length = int(self.starts[i]), int(self.lengths[i])
            if length > MAX_FRAMES:
                offset = np.random.randint(0, length - MAX_FRAMES + 1) if train else (length - MAX_FRAMES) // 2
                start, length = start + offset, MAX_FRAMES
            crops.append(self.feats[start:start + length])
            lens.append(length)
        max_t = max(lens)
        x = np.full((len(idx), max_t, 64), PAD_DB, dtype=np.float32)
        for j, c in enumerate(crops):
            x[j, :len(c)] = c
        return (torch.from_numpy(x).to(self.device, non_blocking=True)[:, None],   # (B, 1, T, 64)
                torch.tensor(lens, device=self.device))


def augment(x: torch.Tensor, lengths: torch.Tensor, full: bool) -> torch.Tensor:
    """Augmentation en GPU sobre el log-Mel. x: (B, 1, T, 64) en dB."""
    x = x.clone()
    B, _, T, _ = x.shape
    for b in range(B):
        L = int(lengths[b])
        w = int(np.random.randint(0, TIME_MASK_MAX + 1))
        if 0 < w < L // 2:
            t0 = int(np.random.randint(0, L - w + 1))
            x[b, :, t0:t0 + w, :] = PAD_DB
        if full:
            f = int(np.random.randint(0, FREQ_MASK_MAX + 1))
            if f > 0:
                f0 = int(np.random.randint(0, N_BANDS_WITH_CONTENT - f + 1))
                x[b, :, :L, f0:f0 + f] = PAD_DB
    if full:
        gain = (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * GAIN_DB
        x = torch.where(x > PAD_DB, x + gain, x)   # el silencio de relleno queda en -100 dB
    return x


# ── Modelo ────────────────────────────────────────────────────────────────────
class FineTuneCnn6(nn.Module):
    def __init__(self, pretrained, first_trainable: int):
        super().__init__()
        self.cnn = pretrained
        self.cnn.fc_audioset = nn.Identity()   # la cabeza de AudioSet no se usa
        self.head = nn.Linear(512, 2)
        self.first_trainable = first_trainable
        for p in self.cnn.bn0.parameters():
            p.requires_grad = False
        for k, name in enumerate(BLOCKS):
            for p in getattr(self.cnn, name).parameters():
                p.requires_grad = k >= first_trainable

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():   # estadísticas de BatchNorm congeladas en toda la red
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def pretrained_params(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("head.")]

    def forward(self, x, lengths):
        x = x.transpose(1, 3)
        x = self.cnn.bn0(x).transpose(1, 3)
        for k, name in enumerate(BLOCKS):
            x = getattr(self.cnn, name)(x)
            x = F.dropout(x, p=0.2, training=self.training and k >= self.first_trainable)
        x = x.mean(dim=3)                                         # (B, 512, T/16)
        valid = torch.clamp(lengths // 16, min=1)
        mask = torch.arange(x.shape[2], device=x.device)[None, :] < valid[:, None]
        x_max = x.masked_fill(~mask[:, None, :], float("-inf")).amax(dim=2)
        x_mean = (x * mask[:, None, :]).sum(dim=2) / valid[:, None]
        x = F.dropout(x_max + x_mean, p=0.5, training=self.training)
        x = F.relu(self.cnn.fc1(x))
        return self.head(F.dropout(x, p=0.5, training=self.training))


def predict(model, bank: LogmelBank, idx: np.ndarray, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=bank.device.type, enabled=amp):
        for s in range(0, len(idx), BATCH_SIZE):
            x, lens = bank.batch(idx[s:s + BATCH_SIZE], train=False)
            out.append(torch.sigmoid(model(x, lens).float()).cpu().numpy())
    return np.concatenate(out)


# ── Un fold, con guardado/retome por época ───────────────────────────────────
def run_fold(i, cfg, epochs, cycles, bank, split, out_dir, device, amp) -> None:
    result_json = out_dir / f"fold{i}_result.json"
    if result_json.exists():
        print(f"[fold {i}] ya terminado en una sesión anterior — se omite")
        return

    idx = {s: np.flatnonzero(split == s) for s in ("train", "val", "test")}
    y_all = cycles[LABEL_COLUMNS].to_numpy(dtype=np.float32)
    y_tr = y_all[idx["train"]]

    model = FineTuneCnn6(load_pretrained_cnn6(cfg["weights"]), cfg["first_trainable"]).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.pretrained_params(), "lr": cfg["lr_pretrained"]},
        {"params": model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=amp)
    pos_weight = torch.tensor((len(y_tr) - y_tr.sum(0)) / y_tr.sum(0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    last_pt, best_pt = out_dir / f"fold{i}_last.pt", out_dir / f"fold{i}_best.pt"
    start_epoch, best_auc, history = 1, -1.0, []
    if last_pt.exists():
        state = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_auc, history = state["epoch"] + 1, state["best_auc"], state["history"]
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        print(f"[fold {i}] retomando desde la época {start_epoch}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fold {i}] ciclos train {len(idx['train'])}, val {len(idx['val'])}, test {len(idx['test'])} | "
          f"parámetros entrenables {n_trainable / 1e6:.2f} M")

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perm = np.random.permutation(idx["train"])
        total = 0.0
        for s in range(0, len(perm), BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            x, lens = bank.batch(b, train=True)
            x = augment(x, lens, cfg["full_augment"])
            y = torch.from_numpy(y_all[b]).to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(x, lens).float(), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(b)
        val_auc = mean_auc(y_all[idx["val"]], predict(model, bank, idx["val"], amp))
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_mean_auc": val_auc,
                        "seconds": time.perf_counter() - t0})
        mark = ""
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_pt)
            mark = "  ✓ mejor"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best_auc": best_auc,
                    "history": history, "np_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state()}, last_pt)
        print(f"[fold {i}] [{epoch:02d}/{epochs}] train_loss={history[-1]['train_loss']:.4f} "
              f"val_AUC={val_auc:.4f} ({history[-1]['seconds']:.0f} s){mark}", flush=True)

    pd.DataFrame(history).to_csv(out_dir / f"training_log_fold{i}.csv", index=False)
    model.load_state_dict(torch.load(best_pt, map_location=device))
    p_va = predict(model, bank, idx["val"], amp)
    thresholds, val_score = calibrate_thresholds(y_all[idx["val"]], p_va)
    # Probabilidades de validación: permiten estudiar otras formas de calibrar
    # los umbrales sin reentrenar (ver avances/17, fold 4 de la variante A).
    val_part = cycles.iloc[idx["val"]][["cycle_id", "subject_id", "crackle", "wheeze"]].copy()
    val_part["fold"] = i
    val_part["p_crackle"], val_part["p_wheeze"] = p_va[:, 0], p_va[:, 1]
    val_part.to_csv(out_dir / f"val_probs_fold{i}.csv", index=False)
    p_te = predict(model, bank, idx["test"], amp)
    m = full_metrics(y_all[idx["test"]], p_te, thresholds)
    h = pd.DataFrame(history)
    m.update({"fold": i, "best_epoch": int(h.loc[h.val_mean_auc.idxmax(), "epoch"]),
              "val_mean_auc": best_auc, "val_icbhi_score": val_score})

    part = cycles.iloc[idx["test"]][["cycle_id", "subject_id", "pathology_label", "equipment",
                                     "age_years", "crackle", "wheeze"]].copy()
    part["fold"] = i
    part["p_crackle"], part["p_wheeze"] = p_te[:, 0], p_te[:, 1]
    part["pred_4class"] = predict_4class(p_te, thresholds)
    part.to_csv(out_dir / f"oof_fold{i}.csv", index=False)
    result_json.write_text(json.dumps({k: (float(v) if isinstance(v, (np.floating, float)) else v)
                                       for k, v in m.items()}, indent=2))
    last_pt.unlink(missing_ok=True)   # el fold terminó: el estado intermedio ya no hace falta
    print(f"[fold {i}] TEST score={m['icbhi_score']:.4f} (Se={m['se']:.3f} Sp={m['sp']:.3f}) | "
          f"AUC crackle={m['crackle_auc']:.3f} wheeze={m['wheeze_auc']:.3f} | mejor época {m['best_epoch']}")


def aggregate(out_dir: Path, run_name: str) -> None:
    results = sorted(out_dir.glob("fold*_result.json"))
    if len(results) < K:
        print(f"Folds terminados: {len(results)}/{K} — el resumen se arma cuando estén los {K}.")
        return
    per_fold = pd.DataFrame([json.loads(f.read_text()) for f in results]).sort_values("fold")
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)
    summary = pd.DataFrame([{"metrica": k, "media": per_fold[k].mean(), "std": per_fold[k].std()}
                            for k in SUMMARY_METRICS])
    summary.to_csv(out_dir / "summary.csv", index=False)
    oof = pd.concat([pd.read_csv(out_dir / f"oof_fold{i}.csv") for i in range(K)], ignore_index=True)
    oof["true_4class"] = to_4class(oof["crackle"].to_numpy(), oof["wheeze"].to_numpy())
    oof["age_group"] = oof["age_years"].map(_age_group)
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    by_equipment, by_age = _group_metrics(oof, "equipment"), _group_metrics(oof, "age_group")
    by_equipment.to_csv(out_dir / "metrics_by_equipment.csv", index=False)
    by_age.to_csv(out_dir / "metrics_by_age_group.csv", index=False)

    cm = confusion_matrix(oof["true_4class"], oof["pred_4class"], labels=range(4))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    ax.set_title(f"Sonidos adventicios — OOF ({run_name})")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_4class.png", dpi=120)
    plt.close(fig)
    fig, (ax_loss, ax_auc) = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(K):
        h = pd.read_csv(out_dir / f"training_log_fold{i}.csv")
        ax_loss.plot(h["epoch"], h["train_loss"], color=f"C{i}", label=f"fold {i}")
        ax_auc.plot(h["epoch"], h["val_mean_auc"], color=f"C{i}", label=f"fold {i}")
    ax_loss.set_title("Loss de train por época")
    ax_auc.set_title("AUC medio de validación por época")
    for ax in (ax_loss, ax_auc):
        ax.set_xlabel("Época")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=120)
    plt.close(fig)

    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"\n{'=' * 60}\n  RESUMEN {run_name}\n{'=' * 60}")
    print(summary.to_string(index=False))
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    print(by_equipment[["equipment", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
    print(by_age[["age_group", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
    print(f"\n✓ Reportes en: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True,
                        help="Carpeta con el contenido de colab_pulmonary_package.zip")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Dónde guardar checkpoints y reportes (en Colab: una carpeta de Drive)")
    parser.add_argument("--variant", choices=list(VARIANTS), required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--epochs", type=int, default=None, help="Por defecto, el de la variante")
    parser.add_argument("--limit", type=int, default=None,
                        help="Solo para pruebas: usa un subconjunto aleatorio de N ciclos")
    args = parser.parse_args()

    cfg = dict(VARIANTS[args.variant])
    epochs = args.epochs or cfg["epochs"]
    run_name = f"v8_cnn6_ft_{args.variant}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    torch.manual_seed(42)
    np.random.seed(42)
    if device.type == "cpu":
        torch.set_flush_denormal(True)
    print(f"Variante {args.variant}: {cfg} | épocas {epochs} | dispositivo {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if amp else ''} | AMP {amp}")

    pkg = args.package_dir
    cfg["weights"] = pkg / "Cnn6_mAP=0.343.pth"
    cycles = pd.read_csv(pkg / "cycles_index.csv")
    feats = np.load(pkg / "logmel_panns32k.npy")
    if args.limit:
        cycles = cycles.sample(args.limit, random_state=0).sort_index().reset_index(drop=True)
    bank = LogmelBank(feats, cycles["logmel_start"].to_numpy(), cycles["logmel_length"].to_numpy(), device)
    print(f"Ciclos: {len(cycles)} | log-Mel en memoria: {feats.nbytes / 1e6:.0f} MB")

    for i in args.folds:
        split = cycles[["subject_id"]].merge(
            pd.read_csv(pkg / "folds" / f"fold{i}.csv"), on="subject_id", how="left")["split"].to_numpy()
        run_fold(i, cfg, epochs, cycles, bank, split, out_dir, device, amp)
    aggregate(out_dir, run_name)


if __name__ == "__main__":
    main()
