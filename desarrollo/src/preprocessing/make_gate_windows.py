"""
preprocessing/make_gate_windows.py
====================================
Genera el dataset de ventanas de audio de duración fija para el modelo gate
(cardíaco vs. pulmonar). No usa segmentación por ciclo — el gate distingue
dominios acústicos, que son reconocibles en cualquier ventana corta.

Para cada archivo del dataset unificado v2:
    1. Preprocesa el audio completo (audio_ops.preprocess).
    2. Lo recorta en ventanas de WINDOW_SEC segundos con HOP_SEC de paso
       (50% overlap). La última ventana parcial se completa con zero-padding
       si tiene al menos HOP_SEC segundos de señal real; si no, se descarta.
    3. Guarda cada ventana como WAV individual y arma un CSV índice.

Uso:
    python preprocessing/make_gate_windows.py

Flags de control (editar las constantes al inicio):
    DRY_RUN : bool — si True, no escribe nada, solo imprime estadísticas.
    LIMIT   : int | None — si se define, procesa solo los primeros N archivos
              (para probar el pipeline antes de correr todo el dataset).
"""

import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    MASTER_CSV_V2, AUDIO_DIR_V2, SPLITS_CSV,
    GATE_WINDOWS_DIR, GATE_WINDOWS_INDEX_CSV,
)
from preprocessing.audio_ops import preprocess

# ── Flags de control ──────────────────────────────────────────────────────────
DRY_RUN = False
LIMIT = None  # ej. 20 para una prueba rápida

# ── Parámetros de ventaneo ────────────────────────────────────────────────────
TARGET_SR = 4000
WINDOW_SEC = 5.0
HOP_SEC = 2.5
WINDOW_SAMPLES = int(WINDOW_SEC * TARGET_SR)
HOP_SAMPLES = int(HOP_SEC * TARGET_SR)
MIN_TAIL_SAMPLES = HOP_SAMPLES  # ventana final descartada si tiene menos que esto


def _slice_windows(y):
    """Devuelve una lista de arrays de largo exacto WINDOW_SAMPLES."""
    windows = []
    n = len(y)
    if n < MIN_TAIL_SAMPLES:
        return windows

    start = 0
    while start < n:
        end = start + WINDOW_SAMPLES
        chunk = y[start:end]
        if len(chunk) < WINDOW_SAMPLES:
            if len(chunk) >= MIN_TAIL_SAMPLES:
                pad = WINDOW_SAMPLES - len(chunk)
                chunk = np.pad(chunk, (0, pad))
                windows.append(chunk)
            break
        windows.append(chunk)
        start += HOP_SAMPLES
    return windows


def make_gate_windows() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV_V2)
    splits = pd.read_csv(SPLITS_CSV)
    df = df.merge(splits, on="subject_id", how="left")

    missing_split = df["split"].isnull().sum()
    if missing_split:
        raise ValueError(
            f"{missing_split} registros sin split asignado — "
            f"correr splits/make_splits.py primero."
        )

    if LIMIT is not None:
        df = df.head(LIMIT)

    index_rows = []
    errors = 0

    for i, row in df.iterrows():
        src_path = AUDIO_DIR_V2 / row["file_id"]
        try:
            y, sr = preprocess(src_path, target_sr=TARGET_SR)
        except Exception as e:
            print(f"⚠ Error procesando {row['file_id']}: {e}")
            errors += 1
            continue

        windows = _slice_windows(y)
        file_stem = Path(row["file_id"]).stem
        split = row["split"]

        for w_idx, chunk in enumerate(windows):
            window_id = f"{file_stem}_{w_idx:03d}"
            index_rows.append({
                "window_id": window_id,
                "file_id": row["file_id"],
                "subject_id": row["subject_id"],
                "split": split,
                "sound_type": row["sound_type"],
                "source_db": row["source_db"],
                "pathology_label": row["pathology_label"],
            })

            if not DRY_RUN:
                out_dir = GATE_WINDOWS_DIR / split
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(out_dir / f"{window_id}.wav", chunk, sr)

        if (i + 1) % 500 == 0:
            print(f"  Procesados {i + 1}/{len(df)} archivos...")

    index_df = pd.DataFrame(index_rows)

    # ── Resumen ───────────────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print("  VENTANAS PARA MODELO GATE — RESUMEN")
    print(sep)
    print(f"  Archivos procesados : {len(df)}")
    print(f"  Errores             : {errors}")
    print(f"  Ventanas generadas  : {len(index_df)}")
    print("\n  Ventanas por split:")
    print(index_df["split"].value_counts().to_string())
    print("\n  Ventanas por sound_type:")
    print(index_df["sound_type"].value_counts().to_string())
    print(sep)

    if DRY_RUN:
        print("\n[DRY RUN] No se escribió ningún archivo.")
        return index_df

    GATE_WINDOWS_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(GATE_WINDOWS_INDEX_CSV, index=False)
    print(f"\n✓ windows_index.csv guardado en: {GATE_WINDOWS_INDEX_CSV}\n")

    return index_df


if __name__ == "__main__":
    make_gate_windows()
