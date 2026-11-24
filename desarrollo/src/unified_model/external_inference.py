"""
unified_model/external_inference.py
======================================
Inferencia para la validación externa (avances/27 §8.1, avances/28): pasa
TODAS las grabaciones de una base externa por los 5 modelos de fold de cada
modelo y guarda las probabilidades por ventana. No calcula métricas
(eso es external_validation.py), así que se corre una sola vez por modelo y
base.

Bases (audio de las caches preprocesadas, audio_ops.preprocess a 4 kHz sin
pasa-altos, igual que al entrenar):
- ICBHI: las 920 grabaciones pulmonares (cache del modelo único, que se armó
  cuando ICBHI era base de entrenamiento). Ningún modelo de esta etapa la vio.
- HLS: los 535 audios de HLS-CMDS (make_unified_audio_cache.py --data hls).
- KAUH (pulmonar) y PASCAL B (cardíaco): grabadas con el equipo de CirCor
  (Littmann 3200 / DigiScope), avances/28 §11 (make_external_recordings.py y
  make_unified_audio_cache.py --data kauh|pascal).

Modelos (criterio B, fold{i}_best.pt):
- cs_wide, abl_card, spr_inv: UnifiedCNN de run_kfold.build_model (config.json
  de la corrida). Las ablaciones son las ramas del pipeline separado.
- cnn6_ours64: Cnn6Classifier de run_cnn6_gpu.py (frontend ours64), con los
  pesos de PANNs como base y el state_dict del fold encima.
- gate_kfold_cs: el gate k-fold del pipeline separado (GateCNN sobre MFCC,
  ventanas de 5 s con paso 2.5 s, como gate_model/run_kfold_circor_spr.py).

Ventanas de los modelos de 6 clases y las ramas: 4 s con paso 2 s, la última
alineada al final (integration.window_inference.fixed_window_starts); las
grabaciones de menos de 4 s se rellenan con ceros (como
evaluate_full_recordings.py).

Salida: reports/external_validation/window_probs/<modelo>_<base>.csv, con
fold, file_id, window_idx, start_sec y las probabilidades (las de
evaluate_circor_spr.FULL_COLS, o prob_cardiac para el gate). Cada fold se
guarda apenas termina (<modelo>_<base>_fold{i}.csv) y no se recalcula: si se
corta, volver a correr retoma.

Uso:
    python unified_model/external_inference.py --models cs_wide cnn6_ours64 --data ICBHI HLS
    python unified_model/external_inference.py --models gate_kfold_cs abl_card spr_inv --data ICBHI HLS
    python unified_model/external_inference.py --models cnn6_ours64 --data HLS --limit 5   # prueba rápida
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (EXTERNAL_REPORTS_DIR, external_paths, HLS_AUDIO_CACHE_NPY, HLS_AUDIO_OFFSETS_CSV, HLS_RECORDINGS_CSV,
                          PANNS_CNN6_CKPT, SPRSOUND_REPORTS_DIR, SRC_ROOT, UNIFIED_AUDIO_CACHE_NPY,
                          UNIFIED_AUDIO_OFFSETS_CSV, UNIFIED_MODEL_REPORTS_DIR, UNIFIED_MODEL_SRC_DIR)
from gate_model.model import GateCNN
from gate_model.run_kfold_circor_spr import gate_window_starts
from integration.evaluate_integration import load_recordings
from integration.window_inference import fixed_window_starts
from preprocessing.make_gate_windows import WINDOW_SAMPLES as GATE_WIN
from unified_model.dataset import SpectralFrontend, derived_probs
from unified_model.evaluate_circor_spr import FULL_COLS
from unified_model.run_cnn6_gpu import AudioBank, Cnn6Classifier, predict as predict_wave
from unified_model.run_kfold import K, build_model

SR = 4000
WIN, HOP = 4 * SR, 2 * SR
OUT_DIR = EXTERNAL_REPORTS_DIR / "window_probs"
MODELS = {   # nombre -> (tipo, carpeta de la corrida con config.json)
    "cs_wide": ("unified", UNIFIED_MODEL_REPORTS_DIR / "runs" / "cs_wide"),
    "cnn6_ours64": ("cnn6", UNIFIED_MODEL_REPORTS_DIR / "runs" / "cnn6_ours64"),
    "abl_card": ("unified", UNIFIED_MODEL_REPORTS_DIR / "runs" / "abl_card"),
    "spr_inv": ("unified", SPRSOUND_REPORTS_DIR / "runs" / "spr_inv"),
    "gate_kfold_cs": ("gate", None),
}
DATASETS = ("ICBHI", "HLS", "KAUH", "PASCAL")


# ── Datos ─────────────────────────────────────────────────────────────────────
def load_dataset(name: str, limit: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """(grabaciones con file_id/offset/n_samples en el audio compacto, audio
    int16 compacto). Solo se copian a memoria las grabaciones de la base
    (ICBHI ocupa ~160 MB de los ~720 MB de la cache del modelo único)."""
    if name == "ICBHI":
        recs = load_recordings().query("source_db == 'ICB'")[["file_id", "subject_id"]]
        cache_npy, offsets_csv = UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV
    else:   # HLS, KAUH, PASCAL: tabla y cache propias (common.paths.external_paths)
        recs_csv, cache_npy, offsets_csv = external_paths(name)
        recs = pd.read_csv(recs_csv)
        recs = recs[["file_id"]].assign(subject_id=recs["subject_id"] if "subject_id" in recs else recs["file_id"])
    if limit:
        recs = recs.head(limit)
    audio = np.load(cache_npy, mmap_mode="r")
    offsets = pd.read_csv(offsets_csv).set_index("file_id")
    recs = recs.merge(offsets, left_on="file_id", right_index=True, how="left")
    assert recs["offset"].notna().all(), "hay grabaciones que no están en la cache"
    chunks, new_off, pos = [], [], 0
    for r in recs.itertuples():
        chunks.append(np.asarray(audio[int(r.offset):int(r.offset) + int(r.n_samples)]))
        new_off.append(pos)
        pos += int(r.n_samples)
    return recs.assign(offset=new_off).reset_index(drop=True), np.concatenate(chunks)


def window_table(recs: pd.DataFrame, gate: bool) -> pd.DataFrame:
    """Una fila por ventana: inicio absoluto en el audio compacto y largo
    (lo que falte hasta la ventana completa se rellena con ceros)."""
    win = GATE_WIN if gate else WIN
    rows = []
    for r in recs.itertuples():
        n = int(r.n_samples)
        # el gate no corta ventanas en grabaciones de menos de 2.5 s (make_gate_windows): afuera se les da
        # una sola ventana rellenada con ceros, como a las de menos de 4 s en los modelos de 6 clases
        starts = (gate_window_starts(n) or [0]) if gate else fixed_window_starts(n, WIN, HOP)
        for w, s in enumerate(starts):
            rows.append({"file_id": r.file_id, "window_idx": w, "start_sec": s / SR,
                         "start": int(r.offset) + s, "length": min(win, n - s)})
    return pd.DataFrame(rows)


# ── Modelos ───────────────────────────────────────────────────────────────────
def load_model(name: str, fold: int) -> torch.nn.Module:
    kind, run_dir = MODELS[name]
    if kind == "gate":
        model = GateCNN()
        ckpt = SRC_ROOT / "gate_model" / "checkpoints_kfold_cs" / f"fold{fold}_best.pt"
    elif kind == "cnn6":
        model = Cnn6Classifier(PANNS_CNN6_CKPT, json.loads((run_dir / "config.json").read_text())["frontend"])
        ckpt = UNIFIED_MODEL_SRC_DIR / f"checkpoints_{name}" / f"fold{fold}_best.pt"
    else:
        model = build_model(argparse.Namespace(**json.loads((run_dir / "config.json").read_text())))
        ckpt = UNIFIED_MODEL_SRC_DIR / f"checkpoints_{name}" / f"fold{fold}_best.pt"
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return model.eval()


def gate_predict(model: GateCNN, bank: AudioBank, starts: np.ndarray, lengths: np.ndarray,
                 batch: int = 256) -> np.ndarray:
    """Como gate_model/run_kfold_circor_spr.predict, con ventanas de 5 s."""
    frontend = SpectralFrontend("mfcc")
    arange = torch.arange(GATE_WIN)
    out = []
    with torch.no_grad():
        for s in range(0, len(starts), batch):
            st = torch.tensor(starts[s:s + batch])[:, None]
            n = torch.tensor(lengths[s:s + batch])[:, None]
            idx = torch.clamp(st + arange, max=len(bank.audio) - 1)
            y = bank.audio[idx].float() / 32768.0 * (arange < n)
            out.append(torch.sigmoid(model(frontend(y))).numpy())
    return np.concatenate(out)


def run(name: str, data: str, limit: int | None) -> Path:
    tag = f"{name}_{data}" + (f"_limit{limit}" if limit else "")
    path = OUT_DIR / f"{tag}.csv"
    if path.exists():
        print(f"  {tag}: ya está ({path.name})", flush=True)
        return path
    gate = MODELS[name][0] == "gate"
    recs, audio = load_dataset(data, limit)
    win = window_table(recs, gate)
    bank = AudioBank(audio, "cpu")
    starts, lengths = win["start"].to_numpy(), win["length"].to_numpy()
    parts = []
    for i in range(K):
        fold_path = OUT_DIR / f"{tag}_fold{i}.csv"
        if fold_path.exists():
            parts.append(pd.read_csv(fold_path))
            continue
        t0 = time.perf_counter()
        model = load_model(name, i)
        part = win[["file_id", "window_idx", "start_sec"]].assign(fold=i)
        if gate:
            part["prob_cardiac"] = gate_predict(model, bank, starts, lengths)
        else:
            d = derived_probs(predict_wave(model, bank, starts, lengths, amp=False))
            for c in FULL_COLS:
                part[c] = d[c]
        part.to_csv(fold_path, index=False)
        parts.append(part)
        print(f"  {tag}: fold {i} — {len(part)} ventanas, {time.perf_counter() - t0:.0f} s", flush=True)
    w = pd.concat(parts, ignore_index=True)
    w.to_csv(path, index=False)
    for i in range(K):
        (OUT_DIR / f"{tag}_fold{i}.csv").unlink(missing_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--data", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--limit", type=int, help="solo las primeras N grabaciones (prueba del código)")
    args = parser.parse_args()
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_flush_denormal(True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for data in args.data:
        for name in args.models:
            print(f"{name} → {data}", flush=True)
            run(name, data, args.limit)
    print(f"✓ {OUT_DIR}")


if __name__ == "__main__":
    main()
