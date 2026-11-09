"""
fig_ventanas.py
===============
Figura 3.2 de la memoria: segmentación en ventanas de 4 s (salto de 2 s) y
etiquetado de una grabación pulmonar de SPRSound. Arriba, la forma de onda
preprocesada con los eventos anotados; abajo, cada ventana con la clase que
recibe por la regla de superposición (>= 1 s o >= 50 % del evento), o
marcada como descartada (ventana normal de una grabación anormal).

Las etiquetas se leen del índice de ventanas (data/sprsound/windows_index_4s.csv),
así la figura muestra exactamente lo que vio el entrenamiento.

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_ventanas.py

Salida: memoria/desarrollo/Figures/ventanas.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV = REPO_ROOT / "desarrollo"
sys.path.insert(0, str(DEV / "src"))
from preprocessing.audio_ops import preprocess  # noqa: E402

OUT_PDF = REPO_ROOT / "memoria" / "desarrollo" / "Figures" / "ventanas.pdf"
STEM = "41246720_4.2_0_p3_1969"
SR, WIN, HOP = 4000, 4.0, 2.0

EVENTOS = {   # tipo de evento en SPRSound -> (rótulo, color)
    "Fine Crackle": ("Crepitante", "#c0392b"),
    "Coarse Crackle": ("Crepitante", "#c0392b"),
    "Wheeze": ("Sibilancia", "#2471a3"),
    "Rhonchi": ("Sibilancia", "#2471a3"),
    "Normal": ("Normal", "#7f8c8d"),
}
CLASES = {   # clase de la ventana -> (rótulo, color)
    "pulm_crackle": ("Crepitantes", "#e6b0aa"),
    "pulm_wheeze": ("Sibilancias", "#a9cce3"),
    "pulm_both": ("Crepitantes y sibilancias", "#d2b4de"),
    "pulm_normal": ("Normal", "#d5dbdb"),
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "font.size": 8,
    })
    rec = pd.read_csv(DEV / "data" / "sprsound" / "recordings.csv")
    rec = rec[rec.file_id == STEM + ".wav"].iloc[0]
    ev = pd.read_csv(DEV / "reports" / "sprsound" / "events.csv")
    ev = ev[ev.stem == STEM].sort_values("start_sec")
    idx = pd.read_csv(DEV / "data" / "sprsound" / "windows_index_4s.csv")
    idx = idx[idx.file_id == STEM + ".wav"].set_index("window_idx")

    y, _ = preprocess(DEV / "data" / "raw" / "SPRSound" / rec.audio_path, target_sr=SR)
    dur = len(y) / SR
    t = np.arange(len(y)) / SR
    # Mismas ventanas que fixed_window_starts: salto de 2 s y la última alineada al final
    starts = list(np.arange(0, dur - WIN + 1e-9, HOP))
    if starts[-1] + WIN < dur - 1e-9:
        starts.append(dur - WIN)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.3, 3.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.1, 1.0], "hspace": 0.12})

    # Forma de onda y eventos anotados
    ax1.plot(t, y, color="#333333", lw=0.3)
    lim = np.abs(y).max() * 1.05
    ax1.set_ylim(-lim, lim)
    vistos = set()
    for _, e in ev.iterrows():
        rot, col = EVENTOS[e.type]
        ax1.axvspan(e.start_sec, e.end_sec, color=col, alpha=0.22, lw=0,
                    label=None if rot in vistos else f"Evento: {rot.lower()}")
        vistos.add(rot)
    ax1.set_ylabel("Amplitud")
    ax1.set_yticks([])
    ax1.legend(loc="upper right", fontsize=7, ncol=3, frameon=False, bbox_to_anchor=(1.0, 1.22))

    # Ventanas y su etiqueta
    for i, s in enumerate(starts):
        yb = len(starts) - 1 - i
        if i in idx.index:
            rot, col = CLASES[idx.loc[i, "label_name"]]
            ax2.add_patch(Rectangle((s, yb + 0.12), WIN, 0.76, fc=col, ec="#4d4d4d", lw=0.6))
        else:
            rot = "Descartada"
            ax2.add_patch(Rectangle((s, yb + 0.12), WIN, 0.76, fc="white", ec="#4d4d4d", lw=0.6,
                                    hatch="////"))
        ax2.text(s + WIN / 2, yb + 0.5, rot, ha="center", va="center", fontsize=7,
                 bbox=dict(fc="white", ec="none", pad=0.6) if rot == "Descartada" else None)
    for _, e in ev.iterrows():
        if e.type != "Normal":
            for x in (e.start_sec, e.end_sec):
                ax2.axvline(x, color=EVENTOS[e.type][1], lw=0.5, ls=":")
    ax2.set_ylim(0, len(starts))
    ax2.set_yticks([len(starts) - 0.5 - i for i in range(len(starts))])
    ax2.set_yticklabels([f"V{i + 1}" for i in range(len(starts))])
    ax2.set_ylabel("Ventanas")
    ax2.set_xlim(0, dur)
    ax2.set_xlabel("Tiempo (s)")
    for a in (ax1, ax2):
        a.spines[["top", "right"]].set_visible(False)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    if len(sys.argv) > 1:   # vista previa opcional en PNG
        fig.savefig(sys.argv[1], bbox_inches="tight", dpi=150)
    print(f"Figura guardada en {OUT_PDF}")


if __name__ == "__main__":
    main()
