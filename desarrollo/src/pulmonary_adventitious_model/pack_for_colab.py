"""
pulmonary_adventitious_model/pack_for_colab.py
=================================================
Empaqueta en UN solo archivo .zip todo lo que necesita el fine-tuning de la
CNN6 en Google Colab (run_finetune_gpu.py, notebook
colab_pulmonary_finetune.ipynb) — ver avances/17.

Por qué empaquetar: el intento anterior de usar Colab (avances/5) falló
porque el entrenamiento leía miles de archivos chicos desde Google Drive
montado, y cada apertura tiene latencia de red: la GPU pasaba la mayor parte
del tiempo esperando datos. Con un solo archivo, el notebook lo copia una
vez al disco local de la VM (segundos) y lo carga entero en memoria.

Qué se empaqueta: los espectrogramas log-Mel YA CALCULADOS con el frontend
original de PANNs ("panns32k": ciclos con pasa-altos 100 Hz, remuestreados
a 32 kHz, STFT 1024 / hop 320, 64 bandas Mel 50-14000 Hz, dB absolutos) —
exactamente la entrada que ganó en la etapa 1 (avances/15). Así Colab no
remuestrea ni calcula STFTs, y toda la augmentation (ganancia, SpecAugment)
se hace directo en la GPU sobre el espectrograma.

Contenido del zip:
    logmel_panns32k.npy      (suma de frames, 64) float16, todos los ciclos concatenados
    cycles_index.csv         índice de ciclos (etiquetas, paciente, equipo, edad)
                             + columnas logmel_start / logmel_length
    folds/fold{0..4}.csv     los mismos 5 folds por paciente de siempre
    Cnn6_mAP=0.343.pth       pesos preentrenados de PANNs (CC-BY-4.0)

Uso:
    python pulmonary_adventitious_model/pack_for_colab.py
    -> data/colab_pulmonary_package.zip (subir ese archivo a Google Drive)
"""

import shutil
import time
import zipfile

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import DATA_DIR, PANNS_CNN6_CKPT, PULMONARY_KFOLD_DIR, pulmonary_cycles_paths
from pulmonary_adventitious_model.extract_cnn6_embeddings import CYCLES_VARIANT, FRONTENDS, logmel
from pulmonary_adventitious_model.run_kfold import K

FRONTEND = "panns32k"
PACKAGE_ZIP = DATA_DIR / "colab_pulmonary_package.zip"


def pack() -> None:
    cfg = FRONTENDS[FRONTEND]
    cycles_dir, index_csv = pulmonary_cycles_paths(CYCLES_VARIANT)
    cycles = pd.read_csv(index_csv)
    mel_basis = librosa.filters.mel(sr=cfg["sr"], n_fft=cfg["n_fft"], n_mels=64,
                                    fmin=cfg["fmin"], fmax=cfg["fmax"])

    chunks, starts, lengths, start = [], [], [], 0
    t0 = time.perf_counter()
    for i, cycle_id in enumerate(cycles["cycle_id"]):
        y, sr = sf.read(cycles_dir / f"{cycle_id}.wav", dtype="float32")
        feat = logmel(y, sr, cfg, mel_basis).astype(np.float16)   # (frames, 64)
        chunks.append(feat)
        starts.append(start)
        lengths.append(feat.shape[0])
        start += feat.shape[0]
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(cycles)} ciclos ({time.perf_counter() - t0:.0f} s)")
    feats = np.concatenate(chunks)
    cycles["logmel_start"], cycles["logmel_length"] = starts, lengths

    staging = DATA_DIR / "colab_pulmonary_package"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "folds").mkdir(parents=True)
    np.save(staging / f"logmel_{FRONTEND}.npy", feats)
    cycles.to_csv(staging / "cycles_index.csv", index=False)
    for i in range(K):
        shutil.copy2(PULMONARY_KFOLD_DIR / f"fold{i}.csv", staging / "folds" / f"fold{i}.csv")
    shutil.copy2(PANNS_CNN6_CKPT, staging / PANNS_CNN6_CKPT.name)

    # Sin compresión: el float16 casi no comprime y así se descomprime al instante.
    with zipfile.ZipFile(PACKAGE_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))
    shutil.rmtree(staging)

    print(f"✓ {PACKAGE_ZIP} — {PACKAGE_ZIP.stat().st_size / 1e6:.0f} MB | log-Mel {feats.shape} "
          f"| frames por ciclo: mediana {np.median(lengths):.0f}, mín {min(lengths)}, máx {max(lengths)} "
          f"| rango dB [{feats.min():.1f}, {feats.max():.1f}] | {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    pack()
