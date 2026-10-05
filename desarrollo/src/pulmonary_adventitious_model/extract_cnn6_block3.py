"""
pulmonary_adventitious_model/extract_cnn6_block3.py
======================================================
Paso previo del fine-tuning parcial de la CNN6 (ver avances/17 y
run_partial_finetune.py): pasa cada ciclo UNA sola vez por la parte que
queda congelada (bn0 + bloques convolucionales 1-3) y guarda su salida. Así,
entrenar el bloque 4 no requiere volver a calcular los bloques 1-3 en cada
época, que es lo que hace inviable el fine-tuning completo en CPU
(avances/15 §1).

Misma entrada que la mejor variante de la etapa 1: ciclos con pasa-altos a
100 Hz, remuestreados a 32 kHz, frontend original de PANNs ("panns32k").

Cada ciclo sale como un tensor (256 canales, t pasos de tiempo, 8 bandas de
frecuencia), con t ≈ frames / 8 (unos 12.5 pasos por segundo). Se guardan
todos concatenados sobre el eje del tiempo, en float16 para que entre en
memoria (~1 GB):

    data/pulmonary_hp100/cnn6_block3_panns32k.npy      (sum(t), 256, 8) float16
    data/pulmonary_hp100/cnn6_block3_panns32k_index.csv cycle_id, start, length

Uso:
    python pulmonary_adventitious_model/extract_cnn6_block3.py
"""

import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PANNS_CNN6_CKPT, pulmonary_cycles_paths
from pulmonary_adventitious_model.panns_cnn6 import Cnn6, load_pretrained_cnn6
from pulmonary_adventitious_model.extract_cnn6_embeddings import CYCLES_VARIANT, FRONTENDS, logmel

FRONTEND = "panns32k"


def block3_paths() -> tuple[Path, Path]:
    base = pulmonary_cycles_paths(CYCLES_VARIANT)[0].parent
    return base / f"cnn6_block3_{FRONTEND}.npy", base / f"cnn6_block3_{FRONTEND}_index.csv"


def extract() -> None:
    cfg = FRONTENDS[FRONTEND]
    cycles_dir, index_csv = pulmonary_cycles_paths(CYCLES_VARIANT)
    cycles = pd.read_csv(index_csv)
    mel_basis = librosa.filters.mel(sr=cfg["sr"], n_fft=cfg["n_fft"], n_mels=Cnn6.N_MELS,
                                    fmin=cfg["fmin"], fmax=cfg["fmax"])
    model = load_pretrained_cnn6(PANNS_CNN6_CKPT)
    torch.set_flush_denormal(True)  # ver extract_cnn6_embeddings.py

    chunks, rows, start = [], [], 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, cycle_id in enumerate(cycles["cycle_id"]):
            y, sr = sf.read(cycles_dir / f"{cycle_id}.wav", dtype="float32")
            x = torch.from_numpy(logmel(y, sr, cfg, mel_basis))[None, None]   # (1, 1, frames, 64)
            out = model.frozen_trunk(x)[0]                                    # (256, t, 8)
            chunk = out.permute(1, 0, 2).numpy().astype(np.float16)           # (t, 256, 8)
            chunks.append(chunk)
            rows.append({"cycle_id": cycle_id, "start": start, "length": chunk.shape[0]})
            start += chunk.shape[0]
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(cycles)} ciclos ({time.perf_counter() - t0:.0f} s)")

    feats = np.concatenate(chunks)
    out_npy, out_csv = block3_paths()
    np.save(out_npy, feats)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    lengths = pd.DataFrame(rows)["length"]
    print(f"✓ {out_npy} — {feats.shape}, {feats.nbytes / 1e9:.2f} GB, {time.perf_counter() - t0:.0f} s | "
          f"pasos por ciclo: mediana {lengths.median():.0f}, mín {lengths.min()}, máx {lengths.max()} | "
          f"rango float16 [{feats.min():.2f}, {feats.max():.2f}]")


if __name__ == "__main__":
    extract()
