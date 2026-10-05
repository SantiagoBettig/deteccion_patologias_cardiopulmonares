"""
pulmonary_adventitious_model/extract_cnn6_embeddings.py
==========================================================
Etapa 1 del transfer learning (ver avances/15): pasa cada ciclo respiratorio
UNA sola vez por la CNN6 preentrenada y CONGELADA (panns_cnn6.py) y guarda
su embedding de 512 valores. Después, run_frozen_probe.py entrena un
clasificador chico sobre esos vectores.

Parte de los ciclos con pasa-altos a 100 Hz (variante "hp100", la entrada de
v4, ver avances/14). Se prueban dos formas de armar el espectrograma de
entrada, porque PANNs se entrenó con audio a 32 kHz y el nuestro es de 4 kHz
(banda recortada a 2 kHz a propósito, por el riesgo de atajo por equipo,
avances/14 §2):

- "panns32k": se remuestrea el ciclo a 32 kHz y se usa el frontend ORIGINAL
  de PANNs (STFT 1024 / hop 320, 64 bandas Mel 50-14000 Hz). Cada banda
  significa la frecuencia que la red conoce, pero 34 de las 64 bandas (las
  de más de 2 kHz) quedan vacías — una entrada que la red nunca vio.
- "ours64": espectrograma propio a 4 kHz, 64 bandas Mel 50-2000 Hz, STFT de
  256 muestras (64 ms) con paso de 10 ms (la resolución temporal de PANNs).
  Aprovecha toda la resolución, pero la banda k no corresponde a la
  frecuencia con la que la red aprendió.

En ambos casos: potencia -> Mel -> 10·log10(max(x, 1e-10)), la misma escala
absoluta que PANNs (no se normaliza por el máximo), porque la primera capa
(bn0) usa las estadísticas de AudioSet. Cada ciclo se procesa con su
duración real (sin recorte ni padding a 4 s): el pooling final de la CNN6 es
global, así que no hace falta una longitud fija.

Guarda en data/pulmonary_hp100/:
    cnn6_embeddings_<frontend>.npy   (n_ciclos, 512), float32
    cnn6_embeddings_<frontend>_ids.csv  (cycle_id en el mismo orden)

Uso:
    python pulmonary_adventitious_model/extract_cnn6_embeddings.py --frontend panns32k
    python pulmonary_adventitious_model/extract_cnn6_embeddings.py --frontend ours64
"""

import argparse
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

CYCLES_VARIANT = "hp100"
MIN_FRAMES = 32  # 4 avg-pools de 2x: por debajo quedaría < 2 frames al final

FRONTENDS = {
    "panns32k": {"sr": 32000, "n_fft": 1024, "hop_length": 320, "fmin": 50, "fmax": 14000},
    "ours64":   {"sr": 4000,  "n_fft": 256,  "hop_length": 40,  "fmin": 50, "fmax": 2000},
}


def embeddings_paths(frontend: str) -> tuple[Path, Path]:
    base = pulmonary_cycles_paths(CYCLES_VARIANT)[0].parent
    return base / f"cnn6_embeddings_{frontend}.npy", base / f"cnn6_embeddings_{frontend}_ids.csv"


def logmel(y: np.ndarray, sr: int, cfg: dict, mel_basis: np.ndarray) -> np.ndarray:
    if sr != cfg["sr"]:
        y = librosa.resample(y, orig_sr=sr, target_sr=cfg["sr"])
    spec = np.abs(librosa.stft(y, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"],
                               window="hann", center=True, pad_mode="reflect")) ** 2
    mel = mel_basis @ spec                                    # (64, frames)
    feat = 10.0 * np.log10(np.maximum(mel, 1e-10)).T          # (frames, 64)
    if feat.shape[0] < MIN_FRAMES:                            # ciclos muy cortos
        feat = np.pad(feat, ((0, MIN_FRAMES - feat.shape[0]), (0, 0)), mode="edge")
    return feat.astype(np.float32)


def extract(frontend: str) -> None:
    cfg = FRONTENDS[frontend]
    cycles_dir, index_csv = pulmonary_cycles_paths(CYCLES_VARIANT)
    cycles = pd.read_csv(index_csv)
    mel_basis = librosa.filters.mel(sr=cfg["sr"], n_fft=cfg["n_fft"], n_mels=Cnn6.N_MELS,
                                    fmin=cfg["fmin"], fmax=cfg["fmax"])
    model = load_pretrained_cnn6(PANNS_CNN6_CKPT)
    # Las bandas vacías (-100 dB en "panns32k") generan números denormales que
    # la CPU procesa muy lento; tratarlos como 0 da el mismo resultado
    # (verificado, atol=1e-5) y ~1.6x más rápido.
    torch.set_flush_denormal(True)
    print(f"Frontend: {frontend} {cfg} | ciclos: {len(cycles)} (variante {CYCLES_VARIANT})")

    embeddings = np.zeros((len(cycles), Cnn6.EMBED_DIM), dtype=np.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, cycle_id in enumerate(cycles["cycle_id"]):
            y, sr = sf.read(cycles_dir / f"{cycle_id}.wav", dtype="float32")
            x = torch.from_numpy(logmel(y, sr, cfg, mel_basis))[None, None]  # (1, 1, frames, 64)
            embeddings[i] = model.embed(x)[0].numpy()
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(cycles)} ciclos ({time.perf_counter() - t0:.0f} s)")

    emb_npy, ids_csv = embeddings_paths(frontend)
    np.save(emb_npy, embeddings)
    cycles[["cycle_id"]].to_csv(ids_csv, index=False)
    active = (embeddings > 0).mean()
    print(f"✓ {emb_npy} — {embeddings.shape}, {time.perf_counter() - t0:.0f} s, "
          f"{100 * active:.1f}% de valores > 0 (post-ReLU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=list(FRONTENDS), required=True)
    extract(parser.parse_args().frontend)
