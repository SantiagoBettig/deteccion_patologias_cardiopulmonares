"""
fig_banco_mel.py
================
Figura 3.3 de la memoria (dos subfiguras): espectrograma log-Mel de una misma
ventana de 4 s con el banco de filtros de PANNs (64 bandas de 50 Hz a 14 kHz,
pensado para 32 kHz) y con el banco adaptado (64 bandas de 50 a 2000 Hz) que
usa la CNN6 del sistema. Con audio a 4 kHz, el banco de PANNs deja vacías las
bandas por encima de 2 kHz.

Los espectrogramas se calculan con la misma clase LogMel que usó el
entrenamiento (unified_model/run_cnn6_gpu.py, frontends "panns32k" y
"ours64"), así la figura muestra exactamente la entrada de la red. Las dos
subfiguras comparten la escala de colores.

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_banco_mel.py

Salida: memoria/desarrollo/Figures/banco_panns.pdf y banco_adaptado.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV = REPO_ROOT / "desarrollo"
sys.path.insert(0, str(DEV / "src"))
from preprocessing.audio_ops import preprocess  # noqa: E402
from unified_model.run_cnn6_gpu import LogMel  # noqa: E402

OUT_DIR = REPO_ROOT / "memoria" / "desarrollo" / "Figures"
STEM = "41246720_4.2_0_p3_1969"   # la grabación de la figura 3.2
INICIO = 5.2                      # ventana V4 de la figura 3.2 (crepitantes)
SR, WIN = 4000, 4.0
RANGO_DB = 80.0                   # rango de colores por debajo del máximo


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "font.size": 8,
    })
    rec = pd.read_csv(DEV / "data" / "sprsound" / "recordings.csv")
    rec = rec[rec.file_id == STEM + ".wav"].iloc[0]
    y, _ = preprocess(DEV / "data" / "raw" / "SPRSound" / rec.audio_path, target_sr=SR)
    s = int(INICIO * SR)
    ventana = torch.tensor(y[s:s + int(WIN * SR)], dtype=torch.float32)[None]

    specs = {}
    with torch.no_grad():
        for nombre in ("panns32k", "ours64"):
            specs[nombre] = LogMel(nombre)(ventana)[0, 0].numpy().T   # (bandas, tramas)
    vmax = max(v.max() for v in specs.values())
    vmin = vmax - RANGO_DB

    for nombre, archivo in (("panns32k", "banco_panns.pdf"), ("ours64", "banco_adaptado.pdf")):
        fig, ax = plt.subplots(figsize=(3.1, 2.3))
        sp = specs[nombre]
        t = np.arange(sp.shape[1]) * 0.01
        ax.imshow(sp, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                  extent=(0, t[-1], -0.5, sp.shape[0] - 0.5), interpolation="nearest")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Banda Mel")
        ax.set_yticks([0, 16, 32, 48, 63])
        vacias = np.flatnonzero(sp.max(axis=1) <= -99)
        if len(vacias):   # bandas por encima de 2 kHz, sin contenido
            b0 = vacias[0] - 0.5
            ax.axhspan(b0, sp.shape[0] - 0.5, fc="none", ec="white", hatch="///", lw=0)
            ax.axhline(b0, color="white", lw=0.8, ls="--")
            ax.text(t[-1] / 2, (b0 + sp.shape[0]) / 2, "Bandas vacías\n(más de 2 kHz)", color="white",
                    ha="center", va="center", fontsize=9.5,
                    bbox=dict(fc="black", ec="none", pad=1.5))
        salida = OUT_DIR / archivo
        fig.savefig(salida, bbox_inches="tight")
        plt.close(fig)
        vacias = int((sp.max(axis=1) <= -99).sum())
        print(f"{salida.name}: {vacias} bandas vacías")


if __name__ == "__main__":
    main()
