"""
preprocessing/make_cardiac_cycle_windows_dense_train.py
===========================================================
Variante de make_cardiac_cycle_windows.py para el modelo de murmullo:
mismo criterio de ciclos (N_CYCLES=4), pero con más solapamiento SOLO en
el split de train (TRAIN_CYCLE_HOP=1 ciclo en vez de 2) — más ventanas de
entrenamiento por grabación, como augmentation (ver
avances/10_cardiaco_split_dos_tareas.md, sección 12; ojo, son recortes muy
correlacionados entre sí, no "más sujetos" — no resuelve el techo real de
~179 sujetos con murmullo, ver también la sección de k-fold).

val/test usan el mismo CYCLE_HOP=2 que make_cardiac_cycle_windows.py — se
regeneran igual (no se copian) para no depender de que ese script ya se
haya corrido, pero el resultado es idéntico ventana por ventana.

Carpeta de salida separada (data/cardiac_murmur_dense/) para no tocar las
ventanas que ya usan cardiac_model/, cardiac_outcome_model/ y v1-v5 de
cardiac_murmur_model/.

Uso:
    python preprocessing/make_cardiac_cycle_windows_dense_train.py
"""

import soundfile as sf
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    MASTER_CSV_V2, AUDIO_DIR_V2, CARDIAC_SPLITS_CSV,
    CARDIAC_DENSE_WINDOWS_DIR, CARDIAC_DENSE_WINDOWS_INDEX_CSV,
)
from unified_dataset.config import CIRCOR_ROOT
from preprocessing.audio_ops import preprocess
from preprocessing.make_cardiac_cycle_windows import _s1_onsets, _cycle_windows

DRY_RUN = False
LIMIT = None

TARGET_SR = 4000
N_CYCLES = 4
CYCLE_HOP_VAL_TEST = 2  # igual que make_cardiac_cycle_windows.py
TRAIN_CYCLE_HOP = 1     # más denso, solo para train

CIRCOR_ANNOTATIONS_DIR = CIRCOR_ROOT / "training_data"


def make_dense_train_windows() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV_V2)
    df = df[df["sound_type"] == "cardiac"].reset_index(drop=True)

    splits = pd.read_csv(CARDIAC_SPLITS_CSV)
    df = df.merge(splits, on="subject_id", how="left")

    missing_split = df["split"].isnull().sum()
    if missing_split:
        raise ValueError(
            f"{missing_split} registros sin split asignado — "
            f"correr splits/make_cardiac_splits.py primero."
        )

    if LIMIT is not None:
        df = df.head(LIMIT)

    index_rows = []
    errors = 0
    skipped_short = 0

    for i, row in df.iterrows():
        src_path = AUDIO_DIR_V2 / row["file_id"]
        tsv_path = CIRCOR_ANNOTATIONS_DIR / f"{Path(row['original_filename']).stem}.tsv"
        split = row["split"]
        hop = TRAIN_CYCLE_HOP if split == "train" else CYCLE_HOP_VAL_TEST

        try:
            s1_onsets = _s1_onsets(tsv_path)
        except Exception as e:
            print(f"⚠ Error leyendo anotación de {row['file_id']} ({tsv_path.name}): {e}")
            errors += 1
            continue

        windows = _cycle_windows(s1_onsets, N_CYCLES, hop)
        if not windows:
            skipped_short += 1
            continue

        try:
            y, sr = preprocess(src_path, target_sr=TARGET_SR)
        except Exception as e:
            print(f"⚠ Error procesando {row['file_id']}: {e}")
            errors += 1
            continue

        file_stem = Path(row["file_id"]).stem

        for w_idx, (t_start, t_end) in enumerate(windows):
            start_sample = int(t_start * sr)
            end_sample = int(t_end * sr)
            chunk = y[start_sample:end_sample]
            if len(chunk) == 0:
                continue

            window_id = f"{file_stem}_{w_idx:03d}"
            index_rows.append({
                "window_id": window_id,
                "file_id": row["file_id"],
                "subject_id": row["subject_id"],
                "split": split,
                "pathology_label": row["pathology_label"],
                "clinical_outcome": row["clinical_outcome"],
                "auscultation_location": row["auscultation_location"],
                "n_cycles": N_CYCLES,
                "duration_sec": len(chunk) / sr,
                "sex": row["sex"],
                "age_category": row["age_category"],
                "height_cm": row["height_cm"],
                "weight_kg": row["weight_kg"],
            })

            if not DRY_RUN:
                out_dir = CARDIAC_DENSE_WINDOWS_DIR / split
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(out_dir / f"{window_id}.wav", chunk, sr)

        if (i + 1) % 500 == 0:
            print(f"  Procesados {i + 1}/{len(df)} archivos...")

    index_df = pd.DataFrame(index_rows)

    sep = "─" * 55
    print(f"\n{sep}")
    print("  VENTANAS DENSAS EN TRAIN — MODELO DE MURMULLO — RESUMEN")
    print(sep)
    print(f"  Archivos procesados         : {len(df)}")
    print(f"  Errores                     : {errors}")
    print(f"  Descartados (< {N_CYCLES} ciclos)   : {skipped_short}")
    print(f"  Ventanas generadas          : {len(index_df)}")
    if len(index_df):
        print("\n  Ventanas por split:")
        print(index_df["split"].value_counts().to_string())
        print("\n  Ventanas de train por pathology_label:")
        print(index_df[index_df["split"] == "train"]["pathology_label"].value_counts().to_string())
    print(sep)

    if DRY_RUN:
        print("\n[DRY RUN] No se escribió ningún archivo.")
        return index_df

    CARDIAC_DENSE_WINDOWS_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(CARDIAC_DENSE_WINDOWS_INDEX_CSV, index=False)
    print(f"\n✓ windows_index.csv guardado en: {CARDIAC_DENSE_WINDOWS_INDEX_CSV}\n")

    return index_df


if __name__ == "__main__":
    make_dense_train_windows()
