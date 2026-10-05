"""
preprocessing/plot_before_after.py
=====================================
Grafica, para cada pathology_label, un puñado de ejemplos representativos
mostrando forma de onda y espectrograma Mel antes y después del
preprocesamiento (filtrado + resample + normalización RMS). Sirve tanto
para detectar problemas visualmente ahora como para el informe final.

Uso:
    python preprocessing/plot_before_after.py
"""

import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MASTER_CSV_V2, AUDIO_DIR_V2, PREPROCESSING_REPORTS_DIR
from preprocessing.audio_ops import preprocess, load_audio

RANDOM_SEED = 42
EXAMPLES_PER_CLASS = 3
TARGET_SR = 4000


def _mel_db(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    return librosa.power_to_db(mel, ref=np.max)


def _plot_class(label: str, file_ids: list[str], df: pd.DataFrame) -> None:
    n = len(file_ids)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for row_idx, file_id in enumerate(file_ids):
        raw_y, raw_sr = load_audio(AUDIO_DIR_V2 / file_id)
        proc_y, proc_sr = preprocess(AUDIO_DIR_V2 / file_id, target_sr=TARGET_SR)

        ax_wave_raw, ax_wave_proc, ax_spec_raw, ax_spec_proc = axes[row_idx]

        librosa.display.waveshow(raw_y, sr=raw_sr, ax=ax_wave_raw)
        ax_wave_raw.set_title(f"{file_id} — forma de onda cruda (sr={raw_sr})")

        librosa.display.waveshow(proc_y, sr=proc_sr, ax=ax_wave_proc)
        ax_wave_proc.set_title(f"{file_id} — preprocesada (sr={proc_sr})")

        img1 = librosa.display.specshow(_mel_db(raw_y, raw_sr), sr=raw_sr,
                                         x_axis="time", y_axis="mel", ax=ax_spec_raw)
        ax_spec_raw.set_title("Mel spectrogram — cruda")

        img2 = librosa.display.specshow(_mel_db(proc_y, proc_sr), sr=proc_sr,
                                         x_axis="time", y_axis="mel", ax=ax_spec_proc)
        ax_spec_proc.set_title("Mel spectrogram — preprocesada")

    fig.suptitle(f"pathology_label = {label}", fontsize=14)
    fig.tight_layout()

    PREPROCESSING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PREPROCESSING_REPORTS_DIR / f"before_after_{label}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  ✓ {out_path}")


def plot_before_after() -> None:
    df = pd.read_csv(MASTER_CSV_V2)
    rng = random.Random(RANDOM_SEED)

    print("Generando gráficos antes/después por clase...")
    for label, group in df.groupby("pathology_label"):
        file_ids = group["file_id"].tolist()
        sample = rng.sample(file_ids, k=min(EXAMPLES_PER_CLASS, len(file_ids)))
        _plot_class(label, sample, df)

    print(f"\n✓ Gráficos guardados en: {PREPROCESSING_REPORTS_DIR}\n")


if __name__ == "__main__":
    plot_before_after()
