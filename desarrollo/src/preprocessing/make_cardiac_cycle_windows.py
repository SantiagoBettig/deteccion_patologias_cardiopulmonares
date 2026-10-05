"""
preprocessing/make_cardiac_cycle_windows.py
==============================================
Genera el dataset de ventanas para el modelo de patología cardíaca (etapa 2),
recortadas por ciclo cardíaco en vez de duración fija (a diferencia del
modelo gate, ver make_gate_windows.py) — motivación en avances/ (ventaneo
fijo puede cortar a mitad de un ciclo, y pierde la posibilidad de ver
variabilidad ciclo a ciclo, relevante para arritmias o murmullos
intermitentes).

Un ciclo se define con la segmentación S1/sístole/S2/diástole de los .tsv
de CirCor (estados 1/2/3/4; ya generados con el algoritmo de Springer et
al., no se re-segmenta acá): el inicio de un ciclo es el onset de un S1
(estado 1) y termina en el onset del siguiente S1. Cada ventana cubre
N_CYCLES ciclos consecutivos, con paso de CYCLE_HOP ciclos (solapamiento).

Para cada archivo cardíaco del dataset unificado v2:
    1. Ubica el .tsv original (mismo nombre base, en CIRCOR_ROOT/training_data)
       vía la columna original_filename — no existe un .tsv por window_id,
       las anotaciones son del audio crudo completo.
    2. Preprocesa el audio completo (audio_ops.preprocess) — el filtrado y
       resample no desalinean los tiempos del .tsv de forma relevante, así
       que se recorta directo sobre la señal ya preprocesada.
    3. Recorta ventanas de N_CYCLES ciclos consecutivos (sin padding: si un
       archivo tiene menos ciclos completos que N_CYCLES, se descarta
       entero — no tiene sentido rellenar con silencio un ciclo cardíaco).
    4. Guarda cada ventana como WAV individual (duración variable en
       muestras, entre sujetos con distinta frecuencia cardíaca) y arma un
       CSV índice.

Usa el split cardíaco independiente (cardiac_splits.csv, ver
splits/make_cardiac_splits.py), no el split conjunto del gate.

Uso:
    python preprocessing/make_cardiac_cycle_windows.py

Flags de control (editar las constantes al inicio):
    DRY_RUN : bool — si True, no escribe nada, solo imprime estadísticas.
    LIMIT   : int | None — si se define, procesa solo los primeros N archivos
              (para probar el pipeline antes de correr todo el dataset).
"""

import soundfile as sf
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    MASTER_CSV_V2, AUDIO_DIR_V2, CARDIAC_SPLITS_CSV,
    CARDIAC_WINDOWS_DIR, CARDIAC_WINDOWS_INDEX_CSV,
)
from unified_dataset.config import CIRCOR_ROOT
from preprocessing.audio_ops import preprocess

# ── Flags de control ──────────────────────────────────────────────────────────
DRY_RUN = False
LIMIT = None  # ej. 20 para una prueba rápida

# ── Parámetros de ventaneo ────────────────────────────────────────────────────
TARGET_SR = 4000
N_CYCLES = 4
CYCLE_HOP = 2  # 50% de solapamiento entre ventanas consecutivas

CIRCOR_ANNOTATIONS_DIR = CIRCOR_ROOT / "training_data"


def _s1_onsets(tsv_path: Path) -> list[float]:
    """Tiempos de inicio (s) de cada S1 anotado, en orden."""
    ann = pd.read_csv(tsv_path, sep="\t", header=None, names=["start", "end", "state"])
    return ann.loc[ann["state"] == 1, "start"].sort_values().tolist()


def _cycle_windows(s1_onsets: list[float], n_cycles: int, hop_cycles: int) -> list[tuple[float, float]]:
    """
    Devuelve pares (t_start, t_end) en segundos, cada uno cubriendo
    exactamente n_cycles ciclos consecutivos (S1 a S1).
    """
    windows = []
    i = 0
    while i + n_cycles < len(s1_onsets):
        windows.append((s1_onsets[i], s1_onsets[i + n_cycles]))
        i += hop_cycles
    return windows


def make_cardiac_cycle_windows() -> pd.DataFrame:
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

        try:
            s1_onsets = _s1_onsets(tsv_path)
        except Exception as e:
            print(f"⚠ Error leyendo anotación de {row['file_id']} ({tsv_path.name}): {e}")
            errors += 1
            continue

        windows = _cycle_windows(s1_onsets, N_CYCLES, CYCLE_HOP)
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
        split = row["split"]

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
                # Metadata del paciente, para el modelo multimodal de murmullo
                # (ver cardiac_murmur_model/run_v4_metadata.py)
                "sex": row["sex"],
                "age_category": row["age_category"],
                "height_cm": row["height_cm"],
                "weight_kg": row["weight_kg"],
            })

            if not DRY_RUN:
                out_dir = CARDIAC_WINDOWS_DIR / split
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(out_dir / f"{window_id}.wav", chunk, sr)

        if (i + 1) % 500 == 0:
            print(f"  Procesados {i + 1}/{len(df)} archivos...")

    index_df = pd.DataFrame(index_rows)

    # ── Resumen ───────────────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print("  VENTANAS POR CICLO — MODELO CARDÍACO — RESUMEN")
    print(sep)
    print(f"  Archivos procesados         : {len(df)}")
    print(f"  Errores                     : {errors}")
    print(f"  Descartados (< {N_CYCLES} ciclos)   : {skipped_short}")
    print(f"  Ventanas generadas          : {len(index_df)}")
    if len(index_df):
        print("\n  Ventanas por split:")
        print(index_df["split"].value_counts().to_string())
        print("\n  Ventanas por pathology_label:")
        print(index_df["pathology_label"].value_counts().to_string())
        print("\n  Duración de ventana (s): media=%.2f mediana=%.2f" % (
            index_df["duration_sec"].mean(), index_df["duration_sec"].median()))
    print(sep)

    if DRY_RUN:
        print("\n[DRY RUN] No se escribió ningún archivo.")
        return index_df

    CARDIAC_WINDOWS_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(CARDIAC_WINDOWS_INDEX_CSV, index=False)
    print(f"\n✓ windows_index.csv guardado en: {CARDIAC_WINDOWS_INDEX_CSV}\n")

    return index_df


if __name__ == "__main__":
    make_cardiac_cycle_windows()
