"""
gate_model/run_kfold_circor_spr.py
=====================================
Gate k-fold (cardíaco vs. pulmonar) con CirCor + SPRSound (avances/25),
para el pipeline separado que se compara con el modelo de 6 clases
(unified_model/run_kfold.py --data circor_spr, avances/24).

Misma receta que el gate v5 (gate_model/run_v5_mfcc.py): GateCNN sobre MFCC
(20 coef., n_mels 64, n_fft 512, hop 160) con z-score por ventana, ventanas
de 5 s con paso 2.5 s cortadas como preprocessing/make_gate_windows.py (la
última se rellena con ceros si tiene al menos 2.5 s), augmentation de
ganancia ±3 dB + ruido (std 0.005), BCE con pos_weight = negativos /
positivos del train, Adam lr 1e-3, batch 32, 15 épocas y checkpoint por F1
de val por ventana.

Cambian los datos y el esquema de splits:
- CirCor (grabaciones cardíacas del índice unificado, folds unificados) y
  SPRSound (data/sprsound/recordings.csv, sus folds): los mismos folds que
  el modelo de 6 clases, así la capa 1 se compara apareada sobre todos los
  pacientes.
- El audio sale de las caches preprocesadas (make_unified_audio_cache.py,
  el mismo audio_ops.preprocess a 4 kHz que el gate v5) y el MFCC se calcula
  por batch con unified_model.dataset.SpectralFrontend (igual a librosa).

Al terminar cada fold pasa por el mejor checkpoint todas las ventanas de
las grabaciones de val y test del fold (mismos roles que run_kfold.py).

Salidas (run = --run-name):
    gate_model/checkpoints_<run>/fold{i}_best.pt
    reports/gate/runs/<run>/history_fold{i}.csv
    reports/gate/runs/<run>/window_probs_fold{i}.csv
    reports/gate/runs/<run>/per_fold.csv

Uso:
    python gate_model/run_kfold_circor_spr.py --run-name kfold_cs
    python gate_model/run_kfold_circor_spr.py --run-name smoke --folds 0 --epochs 1
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (GATE_REPORTS_DIR, SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV,
                          SPRSOUND_KFOLD_DIR, SPRSOUND_RECORDINGS_CSV, SRC_ROOT, UNIFIED_AUDIO_CACHE_NPY,
                          UNIFIED_AUDIO_OFFSETS_CSV, UNIFIED_KFOLD_DIR, unified_windows_index)
from gate_model.model import GateCNN
from preprocessing.make_gate_windows import HOP_SAMPLES, MIN_TAIL_SAMPLES, TARGET_SR, WINDOW_SAMPLES
from unified_model.dataset import SpectralFrontend

K = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SEED = 42


def gate_window_starts(n: int) -> list[int]:
    """Inicios de las ventanas de make_gate_windows._slice_windows: paso
    2.5 s; la última, si no llega a 5 s, entra rellenada con ceros cuando
    tiene al menos 2.5 s. Grabaciones de menos de 2.5 s: ninguna."""
    starts = []
    for s in range(0, n, HOP_SAMPLES):
        if n - s >= WINDOW_SAMPLES:
            starts.append(s)
            continue
        if n - s >= MIN_TAIL_SAMPLES:
            starts.append(s)
        break
    return starts


class GateWindowDataset(Dataset):
    """Todas las ventanas del gate de las grabaciones dadas, leídas de la
    cache (int16) y rellenadas con ceros hasta 5 s. Etiqueta 1 = cardíaco."""

    def __init__(self, recs: pd.DataFrame, audio: np.ndarray, offsets: pd.DataFrame,
                 augment: bool = False, noise_std: float = 0.005, gain_db_range: tuple = (-3.0, 3.0)):
        rows = []
        for r in recs.itertuples():
            off, n = offsets.loc[r.file_id, ["offset", "n_samples"]]
            for w, s in enumerate(gate_window_starts(int(n))):
                rows.append({"file_id": r.file_id, "subject_id": r.subject_id, "domain": r.domain,
                             "window_idx": w, "start_sec": s / TARGET_SR, "start": int(off) + s,
                             "length": min(WINDOW_SAMPLES, int(n) - s), "label": int(r.domain == "cardiac")})
        self.df = pd.DataFrame(rows)
        self.audio = audio
        self.augment, self.noise_std, self.gain_db_range = augment, noise_std, gain_db_range

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        s, n = self.df.at[idx, "start"], self.df.at[idx, "length"]
        y = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        y[:n] = self.audio[s:s + n].astype(np.float32) / 32768.0
        if self.augment:   # como GateDataset._augment_waveform, sobre la ventana ya rellenada
            y = y * (10.0 ** (np.random.uniform(*self.gain_db_range) / 20.0))
            y = np.clip(y + np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32), -1.0, 1.0)
        return torch.from_numpy(y), torch.tensor(float(self.df.at[idx, "label"]))


def load_sources() -> dict[str, tuple]:
    """nombre -> (grabaciones file_id/subject_id/domain, audio, offsets, carpeta de folds)."""
    circor = (pd.read_csv(unified_windows_index(4.0)).query("domain == 'cardiac'")
              [["file_id", "subject_id", "domain"]].drop_duplicates())
    spr = pd.read_csv(SPRSOUND_RECORDINGS_CSV)[["file_id", "subject_id"]].assign(domain="pulmonary")
    out = {}
    for name, recs, cache_npy, offsets_csv, fold_dir in (
            ("circor", circor, UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV, UNIFIED_KFOLD_DIR),
            ("sprsound", spr, SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV, SPRSOUND_KFOLD_DIR)):
        out[name] = (recs, np.load(cache_npy, mmap_mode="r"), pd.read_csv(offsets_csv).set_index("file_id"), fold_dir)
    return out


def predict(model, frontend, loader) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.sigmoid(model(frontend(x))).numpy())
    return np.concatenate(out)


def run_fold(i: int, sources: dict, args, ckpt_dir: Path, out_dir: Path) -> dict:
    torch.manual_seed(SEED + i)
    np.random.seed(SEED + i)
    ds = {s: [] for s in ("train", "val", "test")}
    for name, (recs, audio, offsets, fold_dir) in sources.items():
        fold = pd.read_csv(fold_dir / f"fold{i}.csv")[["subject_id", "split"]]
        r = recs.merge(fold, on="subject_id")
        for s in ds:
            ds[s].append(GateWindowDataset(r[r["split"] == s], audio, offsets, augment=s == "train"))
    train_df = pd.concat([d.df for d in ds["train"]])
    val_df = pd.concat([d.df for d in ds["val"]])
    n_pos = int(train_df["label"].sum())
    pos_weight = torch.tensor((len(train_df) - n_pos) / n_pos, dtype=torch.float32)
    print(f"Fold {i}: train {len(train_df)} | val {len(val_df)} | test {sum(len(d) for d in ds['test'])} ventanas "
          f"| cardíacas en train {n_pos} | pos_weight {pos_weight.item():.3f}", flush=True)

    train_loader = DataLoader(ConcatDataset(ds["train"]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ConcatDataset(ds["val"]), batch_size=256, shuffle=False)
    frontend = SpectralFrontend("mfcc")
    model = GateCNN()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(frontend(x)), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        p = predict(model, frontend, val_loader)
        y_val = val_df["label"].to_numpy()
        m = {"epoch": epoch, "train_loss": total / n, "val_acc": accuracy_score(y_val, p > 0.5),
             "val_f1": f1_score(y_val, p > 0.5), "sec": time.perf_counter() - t0}
        history.append(m)
        improved = m["val_f1"] > best
        print(f"  [{epoch:02d}] loss {m['train_loss']:.4f} | val acc {m['val_acc']:.4f} f1 {m['val_f1']:.4f} "
              f"| {m['sec']:.0f} s{' *' if improved else ''}", flush=True)
        if improved:
            best = m["val_f1"]
            torch.save(model.state_dict(), ckpt_dir / f"fold{i}_best.pt")
    pd.DataFrame(history).to_csv(out_dir / f"history_fold{i}.csv", index=False)

    # Probabilidades del mejor checkpoint en todas las ventanas de val y test
    model.load_state_dict(torch.load(ckpt_dir / f"fold{i}_best.pt", map_location="cpu"))
    rows, res = [], {"fold": i, "best_epoch": max(history, key=lambda h: h["val_f1"])["epoch"]}
    for role in ("val", "test"):
        p = predict(model, frontend, DataLoader(ConcatDataset(ds[role]), batch_size=256, shuffle=False))
        part = pd.concat([d.df for d in ds[role]], ignore_index=True)
        part = part[["file_id", "subject_id", "domain", "window_idx", "start_sec", "label"]].assign(
            fold=i, role=role, prob_cardiac=p)
        rows.append(part)
        rec = part.groupby("file_id").agg(label=("label", "first"), prob=("prob_cardiac", "mean"))
        res.update({f"{role}_window_acc": accuracy_score(part["label"], part["prob_cardiac"] > 0.5),
                    f"{role}_window_f1": f1_score(part["label"], part["prob_cardiac"] > 0.5),
                    f"{role}_recording_acc": accuracy_score(rec["label"], rec["prob"] > 0.5),
                    f"{role}_recording_errors": int(((rec["prob"] > 0.5) != rec["label"]).sum())})
    pd.concat(rows).to_csv(out_dir / f"window_probs_fold{i}.csv", index=False)
    print(f"  Fold {i} — mejor época {res['best_epoch']} | test: ventana acc {res['test_window_acc']:.4f} | "
          f"grabación acc {res['test_recording_acc']:.4f} ({res['test_recording_errors']} errores)", flush=True)
    (out_dir / f"result_fold{i}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    out_dir = GATE_REPORTS_DIR / "runs" / args.run_name
    ckpt_dir = SRC_ROOT / "gate_model" / f"checkpoints_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({k: v for k, v in vars(args).items() if k != "folds"}, indent=2))
    sources = load_sources()
    print(f"Gate k-fold {args.run_name}: CirCor + SPRSound, {args.epochs} épocas, folds {args.folds}", flush=True)

    for i in args.folds:
        run_fold(i, sources, args, ckpt_dir, out_dir)
    per_fold = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out_dir.glob("result_fold*.json"))])
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)
    print(f"\nFolds terminados: {per_fold['fold'].tolist()}")
    cols = [c for c in per_fold.columns if c.startswith("test_")]
    print(f"\n{per_fold[cols].agg(['mean', 'std', 'sum']).T.round(4).to_string()}")


if __name__ == "__main__":
    main()
