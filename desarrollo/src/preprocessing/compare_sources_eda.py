"""
preprocessing/compare_sources_eda.py
=======================================
Compara características de audio crudo y preprocesado entre las tres
fuentes (CirCor, ICBHI, HLS-CMDS), para investigar si hay una diferencia
concreta y medible que el modelo gate podría estar explotando además del
sample rate nativo (ya documentado en avances/4).

Guarda:
    - reports/preprocessing/source_comparison_stats.csv
    - reports/preprocessing/source_comparison_boxplots.png

Uso:
    python preprocessing/compare_sources_eda.py
"""

import random
import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unified_dataset"))

from common.paths import MASTER_CSV_V2, AUDIO_DIR_V2, PREPROCESSING_REPORTS_DIR
from parsers.parse_hls import parse_hls
from config import HLS_ROOT
from preprocessing.audio_ops import preprocess, load_audio

RANDOM_SEED = 42
N_SAMPLES_PER_SOURCE = 60


def _stats(y: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) if len(y) else 0.0
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y))) if len(y) else 0.0
    silence_frac = float(np.mean(np.abs(y) < 1e-3)) if len(y) else 0.0
    return {
        "duration_sec": len(y) / sr if sr else 0.0,
        "rms": rms, "peak": peak,
        "spectral_centroid_hz": centroid, "zcr": zcr,
        "silence_frac": silence_frac,
    }


def _collect_rows(source_db: str, sound_type: str, path: Path) -> dict:
    raw_y, raw_sr = load_audio(path)
    proc_y, proc_sr = preprocess(path)

    row = {"source_db": source_db, "sound_type": sound_type, "native_sr": raw_sr}
    for k, v in _stats(raw_y, raw_sr).items():
        row[f"raw_{k}"] = v
    for k, v in _stats(proc_y, proc_sr).items():
        row[f"proc_{k}"] = v
    return row


def compare_sources_eda() -> pd.DataFrame:
    rng = random.Random(RANDOM_SEED)
    rows = []

    # ── CirCor + ICBHI, desde el dataset unificado ─────────────────────────────
    master = pd.read_csv(MASTER_CSV_V2)
    for source_db, group in master.groupby("source_db"):
        sample = group.sample(min(N_SAMPLES_PER_SOURCE, len(group)), random_state=RANDOM_SEED)
        for _, row in sample.iterrows():
            path = AUDIO_DIR_V2 / row["file_id"]
            try:
                rows.append(_collect_rows(source_db, row["sound_type"], path))
            except Exception as e:
                print(f"⚠ Error con {row['file_id']}: {e}")

    # ── HLS-CMDS, desde los datos crudos ─────────────────────────────────────
    hls_df, _ = parse_hls()
    hls_sample = hls_df.sample(min(N_SAMPLES_PER_SOURCE, len(hls_df)), random_state=RANDOM_SEED)
    for _, row in hls_sample.iterrows():
        subfolder = "HS" if row["sound_type"] == "cardiac" else "LS"
        path = HLS_ROOT / subfolder / row["original_filename"]
        try:
            rows.append(_collect_rows("HLS", row["sound_type"], path))
        except Exception as e:
            print(f"⚠ Error con {row['original_filename']}: {e}")

    stats_df = pd.DataFrame(rows)

    print("\n── Medianas por source_db (audio preprocesado) ──")
    print(stats_df.groupby("source_db")[
        ["proc_rms", "proc_peak", "proc_spectral_centroid_hz", "proc_zcr", "proc_silence_frac"]
    ].median().to_string())

    print("\n── Sample rate nativo por source_db ──")
    print(stats_df.groupby("source_db")["native_sr"].agg(["min", "max", "mean"]).to_string())

    PREPROCESSING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(PREPROCESSING_REPORTS_DIR / "source_comparison_stats.csv", index=False)

    metrics = ["raw_duration_sec", "proc_rms", "proc_peak",
               "proc_spectral_centroid_hz", "proc_zcr", "proc_silence_frac"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, metric in zip(axes.flat, metrics):
        stats_df.boxplot(column=metric, by="source_db", ax=ax)
        ax.set_title(metric)
        ax.set_xlabel("")
    fig.suptitle("Comparación de características de audio por fuente (CirCor / ICBHI / HLS)")
    fig.tight_layout()
    fig.savefig(PREPROCESSING_REPORTS_DIR / "source_comparison_boxplots.png", dpi=120)
    plt.close(fig)

    print(f"\n✓ CSV guardado en: {PREPROCESSING_REPORTS_DIR / 'source_comparison_stats.csv'}")
    print(f"✓ Boxplots guardados en: {PREPROCESSING_REPORTS_DIR / 'source_comparison_boxplots.png'}\n")

    return stats_df


if __name__ == "__main__":
    compare_sources_eda()
