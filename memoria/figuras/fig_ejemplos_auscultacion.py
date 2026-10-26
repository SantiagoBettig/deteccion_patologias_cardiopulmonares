"""
fig_ejemplos_auscultacion.py
============================
Figura 1.1 de la memoria: forma de onda y espectrograma log-Mel de cuatro
grabaciones de auscultación (cardíaco normal, cardíaco con soplo, pulmonar
normal y pulmonar con crepitantes), tomadas de CirCor e ICBHI.

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_ejemplos_auscultacion.py

Salida: memoria/desarrollo/Figures/ejemplos_auscultacion.pdf
"""

import sys
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "desarrollo" / "data" / "raw"
OUT_PDF   = REPO_ROOT / "memoria" / "desarrollo" / "Figures" / "ejemplos_auscultacion.pdf"

# Misma frecuencia de muestreo que usa el proyecto para todas las ramas
SR       = 4000
DUR_SEC  = 6.0
N_FFT    = 256
HOP      = 32
N_MELS   = 64
FMAX     = 2000

# (título, archivo relativo a data/raw, inicio del recorte en segundos)
EJEMPLOS = [
    ("Cardíaco normal",         "CirCor/training_data/50388_TV.wav",                       2.0),
    ("Cardíaco con soplo",      "CirCor/training_data/49628_TV.wav",                       2.0),
    ("Pulmonar normal",         "ICBHI/audio_and_txt_files/187_1b1_Ll_sc_Meditron.wav",        0.0),
    ("Pulmonar con crepitantes", "ICBHI/audio_and_txt_files/178_1b2_Pl_mc_AKGC417L.wav",   0.0),
]


def cargar(ruta: Path, inicio: float) -> np.ndarray:
    y, _ = librosa.load(ruta, sr=SR, offset=inicio, duration=DUR_SEC)
    # Pasa-altos suave solo para la visualización: quita la deriva de baja
    # frecuencia que, si no, domina la escala de la forma de onda
    y = sosfiltfilt(butter(4, 25, btype="highpass", fs=SR, output="sos"), y)
    return y / (np.abs(y).max() + 1e-9)


def main(ejemplos=EJEMPLOS, salida=OUT_PDF):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "font.size": 8,
    })
    fig, axes = plt.subplots(
        5, 2, figsize=(6.3, 5.9), sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 0.45, 1, 2], "hspace": 0.12, "wspace": 0.12},
    )
    for ax in axes[2]:  # fila vacía que separa los ejemplos cardíacos de los pulmonares
        ax.set_visible(False)
    for k, (titulo, rel, inicio) in enumerate(ejemplos):
        fila, col = 3 * (k // 2), k % 2
        y = cargar(RAW_DIR / rel, inicio)
        t = np.arange(len(y)) / SR

        ax_w = axes[fila, col]
        ax_w.plot(t, y, color="#1f4e79", linewidth=0.4)
        ax_w.set_title(titulo, fontsize=9, pad=3)
        ax_w.set_ylim(-1.05, 1.05)
        ax_w.set_yticks([])
        if col == 0:
            ax_w.set_ylabel("Amplitud")

        ax_s = axes[fila + 1, col]
        mel = librosa.feature.melspectrogram(
            y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmax=FMAX)
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=70)
        librosa.display.specshow(
            mel_db, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel",
            fmax=FMAX, cmap="magma", ax=ax_s)
        ax_s.set_yticks([0, 250, 500, 1000, 2000])
        ax_s.set_ylabel("Frecuencia [Hz]" if col == 0 else "")
        if col == 1:
            ax_s.set_yticklabels([])
        ax_s.set_xlabel("Tiempo [s]" if fila == 3 else "")
        ax_s.set_xlim(0, DUR_SEC)
        ax_s.set_xticks(np.arange(0, DUR_SEC + 0.1, 1.0))

    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight")
    print(f"Figura guardada en {salida}")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # vista previa en otra ruta (png) sin pisar la figura
        main(salida=Path(sys.argv[1]))
    else:
        main()
