"""
fig_alteraciones.py
===================
Figura 2.1 de la memoria: forma de onda y espectrograma log-Mel de un tramo
corto de cada alteración buscada (soplo, crepitantes y sibilancias), tomados
de CirCor y SPRSound. Cada panel muestra solo el tramo de interés: un ciclo
cardíaco completo (S1, sístole con soplo y S2) o un único evento adventicio,
con el intervalo anotado en cada base marcado sobre la forma de onda.

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_alteraciones.py

Salida: memoria/desarrollo/Figures/alteraciones.pdf
"""

import json
import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "desarrollo" / "data" / "raw"
CC_DIR    = RAW_DIR / "CirCor" / "training_data"
SPR_DIR   = RAW_DIR / "SPRSound" / "BioCAS2022"
OUT_PDF   = REPO_ROOT / "memoria" / "desarrollo" / "Figures" / "alteraciones.pdf"

# Parámetros de visualización de la figura 1.1, con un salto menor porque
# los tramos son cortos
SR       = 4000
N_FFT    = 256
HOP      = 16
N_MELS   = 64
FMAX     = 2000
MARGEN   = 0.08   # segundos que se muestran antes y después del tramo de interés
COLOR_EV = "#c0392b"

# (título, fuente, archivo, instante aproximado del tramo en segundos, rótulo, duración)
# En CirCor se toma el primer ciclo cardíaco completo a partir de ese instante;
# en SPRSound, el primer evento adventicio que empieza después de ese instante.
# Si se da una duración, se muestra ese recorte fijo (el evento de crepitantes
# de SPRSound abarca toda una fase respiratoria y en el recorte completo cada
# crepitante se ve demasiado angosto).
EJEMPLOS = [
    ("Soplo",       "circor", "84693_PV",               3.0, "Sístole",    None),
    ("Crepitantes", "spr",    "41161556_1.7_0_p2_2315", 4.6, "Crepitante", 0.6),
    ("Sibilancia",  "spr",    "40976541_2.7_1_p1_3305", 9.4, "Sibilancia", None),
]


def tramo(fuente: str, nombre: str, desde: float,
          duracion: float | None = None) -> tuple[float, float, list[tuple[str, float, float]]]:
    """Inicio y fin del recorte, y los intervalos a marcar (rótulo, inicio, fin) en tiempo absoluto."""
    if duracion is not None:  # recorte fijo, solo para SPRSound
        inicio, fin = desde, desde + duracion
        d = json.loads((SPR_DIR / "train2022_json" / f"{nombre}.json").read_text(encoding="utf-8"))
        marcas = [("ev", max(int(e["start"]) / 1000, inicio), min(int(e["end"]) / 1000, fin))
                  for e in d["event_annotation"] if e["type"] != "Normal"
                  and int(e["end"]) / 1000 > inicio and int(e["start"]) / 1000 < fin]
        return inicio, fin, marcas
    if fuente == "circor":
        seg = np.loadtxt(CC_DIR / f"{nombre}.tsv")  # 1 = S1, 2 = sístole, 3 = S2, 4 = diástole
        i = next(k for k in range(len(seg) - 2)
                 if seg[k, 0] >= desde and list(seg[k:k + 3, 2]) == [1, 2, 3])
        s1, sis, s2 = seg[i], seg[i + 1], seg[i + 2]
        marcas = [("S1", s1[0], s1[1]), ("sis", sis[0], sis[1]), ("S2", s2[0], s2[1])]
        return s1[0] - MARGEN, s2[1] + MARGEN, marcas
    d = json.loads((SPR_DIR / "train2022_json" / f"{nombre}.json").read_text(encoding="utf-8"))
    ev = sorted((int(e["start"]) / 1000, int(e["end"]) / 1000)
                for e in d["event_annotation"] if e["type"] != "Normal")  # SPRSound también anota ciclos normales
    s, f = next((s, f) for s, f in ev if s >= desde)
    return s - MARGEN, f + MARGEN, [("ev", s, f)]


def cargar(ruta: Path, inicio: float, fin: float) -> np.ndarray:
    # Se carga con contexto a ambos lados para que el filtro no deforme los bordes
    pad = 0.5
    y, _ = librosa.load(ruta, sr=SR, offset=max(inicio - pad, 0), duration=fin - inicio + 2 * pad)
    y = sosfiltfilt(butter(4, 25, btype="highpass", fs=SR, output="sos"), y)  # igual que en la figura 1.1
    a = int(round(min(pad, inicio) * SR))
    y = y[a:a + int(round((fin - inicio) * SR))]
    return y / (np.abs(y).max() + 1e-9)


def main(ejemplos=EJEMPLOS, salida=OUT_PDF):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "font.size": 8,
    })
    fig, axes = plt.subplots(
        2, len(ejemplos), figsize=(6.3, 3.3),
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.12, "wspace": 0.12},
    )
    for col, (titulo, fuente, nombre, desde, rotulo, duracion) in enumerate(ejemplos):
        inicio, fin, marcas = tramo(fuente, nombre, desde, duracion)
        ruta = CC_DIR / f"{nombre}.wav" if fuente == "circor" else SPR_DIR / "train2022_wav" / f"{nombre}.wav"
        y = cargar(ruta, inicio, fin)
        dur = len(y) / SR
        t = np.arange(len(y)) / SR

        ax_w = axes[0, col]
        ax_w.plot(t, y, color="#1f4e79", linewidth=0.5)
        for etiqueta, s, f in marcas:
            s, f = s - inicio, f - inicio
            if etiqueta in ("S1", "S2"):
                ax_w.text((s + f) / 2, 1.13, etiqueta, ha="center", va="center", fontsize=7)
            else:
                ax_w.axvspan(s, f, ymin=0.9, ymax=1.0, color=COLOR_EV, alpha=0.85, linewidth=0,
                             label=rotulo)
        ax_w.set_title(titulo, fontsize=9, pad=3)
        ax_w.set_xlim(0, dur)
        ax_w.set_ylim(-1.05, 1.3)
        ax_w.set_yticks([])
        ax_w.set_xticklabels([])
        ax_w.legend(loc="lower right", fontsize=6.5, frameon=False, handlelength=1.2,
                    borderaxespad=0.1)
        if col == 0:
            ax_w.set_ylabel("Amplitud")

        ax_s = axes[1, col]
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=70)
        librosa.display.specshow(
            mel_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel",
            fmax=FMAX, cmap="magma", ax=ax_s)
        ax_s.set_yticks([0, 250, 500, 1000, 2000])
        ax_s.set_ylabel("Frecuencia [Hz]" if col == 0 else "")
        if col > 0:
            ax_s.set_yticklabels([])
        ax_s.set_xlabel("Tiempo [s]")
        ax_s.set_xlim(0, dur)
        for ax in (ax_w, ax_s):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5]))
        ax_s.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))

    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight")
    print(f"Figura guardada en {salida}")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # vista previa en otra ruta (png) sin pisar la figura
        main(salida=Path(sys.argv[1]))
    else:
        main()
