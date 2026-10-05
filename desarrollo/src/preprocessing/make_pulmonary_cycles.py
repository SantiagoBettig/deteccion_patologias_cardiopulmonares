"""
preprocessing/make_pulmonary_cycles.py
=========================================
Genera el dataset de ciclos respiratorios para el modelo de sonidos
adventicios (crackles / wheezes) — etapa 2, rama pulmonar. Ver
avances/12_icbhi_fe_de_erratas_equipo.md y la discusión previa sobre por qué
la tarea pulmonar se plantea por ciclo y no por diagnóstico del paciente
(volumen de pacientes por diagnóstico muy chico, sesgo por edad).

Cada línea del .txt de anotación de ICBHI es un ciclo respiratorio:
    inicio (s) | fin (s) | crackle (0/1) | wheeze (0/1)
Una muestra del dataset = un ciclo, con sus dos etiquetas tal cual vienen
del .txt (tarea oficial del challenge ICBHI 2017).

Para cada grabación pulmonar del dataset unificado v2:
    1. Ubica el .txt original (mismo nombre base que original_filename, en
       ICBHI_ROOT/audio_and_txt_files — el nombre en disco, sin la
       corrección de equipo, que solo afecta a la columna equipment).
    2. Preprocesa el audio completo (audio_ops.preprocess: pasa-bajos
       2048 Hz -> resample 4000 Hz -> RMS), igual que el resto del pipeline.
       La normalización RMS es por grabación, no por ciclo: así un ciclo
       con crackles fuertes no queda "aplanado" al nivel de uno normal.
    3. Recorta cada ciclo anotado y lo guarda como WAV individual, SIN
       padding ni recorte a duración fija (eso se decide en el Dataset, para
       poder cambiarlo sin regenerar los WAV).

No asigna split: la evaluación es k-fold por paciente
(splits/make_pulmonary_kfold.py).

Uso:
    python preprocessing/make_pulmonary_cycles.py
    python preprocessing/make_pulmonary_cycles.py --highpass 100   # -> data/pulmonary_hp100/

Flags de control (editar las constantes al inicio):
    DRY_RUN : bool — si True, no escribe nada, solo imprime estadísticas.
    LIMIT   : int | None — si se define, procesa solo los primeros N archivos.
"""

import argparse

import pandas as pd
import soundfile as sf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MASTER_CSV_V2, AUDIO_DIR_V2, pulmonary_cycles_paths
from unified_dataset.config import ICBHI_ROOT
from preprocessing.audio_ops import preprocess

# ── Flags de control ──────────────────────────────────────────────────────────
DRY_RUN = False
LIMIT = None  # ej. 20 para una prueba rápida

TARGET_SR = 4000
MIN_CYCLE_SEC = 0.1  # por debajo de esto el ciclo no tiene ni un frame útil

ICBHI_ANNOTATIONS_DIR = ICBHI_ROOT / "audio_and_txt_files"


def _read_cycles(txt_path: Path) -> pd.DataFrame:
    return pd.read_csv(txt_path, sep="\t", header=None,
                       names=["start", "end", "crackle", "wheeze"])


def make_pulmonary_cycles(highpass_cutoff: float | None = None) -> pd.DataFrame:
    variant = None if highpass_cutoff is None else f"hp{int(highpass_cutoff)}"
    PULMONARY_CYCLES_DIR, PULMONARY_CYCLES_INDEX_CSV = pulmonary_cycles_paths(variant)
    print(f"Pasa-altos: {highpass_cutoff or 'no'} | salida: {PULMONARY_CYCLES_DIR.parent}")
    df = pd.read_csv(MASTER_CSV_V2)
    df = df[df["sound_type"] == "pulmonary"].reset_index(drop=True)
    if LIMIT is not None:
        df = df.head(LIMIT)

    if not DRY_RUN:
        PULMONARY_CYCLES_DIR.mkdir(parents=True, exist_ok=True)

    index_rows = []
    errors = 0
    skipped_short = 0
    clipped = 0

    for i, row in df.iterrows():
        src_path = AUDIO_DIR_V2 / row["file_id"]
        txt_path = ICBHI_ANNOTATIONS_DIR / f"{Path(row['original_filename']).stem}.txt"

        try:
            cycles = _read_cycles(txt_path)
            y, sr = preprocess(src_path, target_sr=TARGET_SR, highpass_cutoff=highpass_cutoff)
        except Exception as e:
            print(f"⚠ Error procesando {row['file_id']} ({txt_path.name}): {e}")
            errors += 1
            continue

        file_stem = Path(row["file_id"]).stem
        audio_len_sec = len(y) / sr

        for c_idx, cyc in cycles.iterrows():
            t_start = max(0.0, float(cyc["start"]))
            t_end = min(float(cyc["end"]), audio_len_sec)
            if t_end < float(cyc["end"]):
                clipped += 1  # anotación que se pasa del final del audio
            if t_end - t_start < MIN_CYCLE_SEC:
                skipped_short += 1
                continue

            chunk = y[int(t_start * sr):int(t_end * sr)]
            cycle_id = f"{file_stem}_{c_idx:03d}"
            index_rows.append({
                "cycle_id": cycle_id,
                "file_id": row["file_id"],
                "subject_id": row["subject_id"],
                "pathology_label": row["pathology_label"],
                "crackle": int(cyc["crackle"]),
                "wheeze": int(cyc["wheeze"]),
                "start_sec": t_start,
                "duration_sec": len(chunk) / sr,
                # Solo para diagnósticos de sesgo (equipo / edad) sobre las
                # predicciones — no entran al modelo.
                "equipment": row["equipment"],
                "auscultation_location": row["auscultation_location"],
                "age_years": row["age_years"],
            })

            if not DRY_RUN:
                sf.write(PULMONARY_CYCLES_DIR / f"{cycle_id}.wav", chunk, sr)

        if (i + 1) % 100 == 0:
            print(f"  Procesados {i + 1}/{len(df)} archivos...")

    index_df = pd.DataFrame(index_rows)

    # ── Resumen ───────────────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print("  CICLOS RESPIRATORIOS — MODELO PULMONAR — RESUMEN")
    print(sep)
    print(f"  Grabaciones procesadas      : {len(df)}")
    print(f"  Errores                     : {errors}")
    print(f"  Ciclos descartados (< {MIN_CYCLE_SEC}s) : {skipped_short}")
    print(f"  Ciclos recortados al final  : {clipped}")
    print(f"  Ciclos generados            : {len(index_df)}")
    if len(index_df):
        combo = (index_df["crackle"].astype(str) + index_df["wheeze"].astype(str)).map(
            {"00": "normal", "10": "crackle", "01": "wheeze", "11": "both"})
        print("\n  Ciclos por clase (4 clases ICBHI):")
        print(combo.value_counts().to_string())
        print("\n  Duración de ciclo (s): media=%.2f mediana=%.2f p90=%.2f max=%.2f" % (
            index_df["duration_sec"].mean(), index_df["duration_sec"].median(),
            index_df["duration_sec"].quantile(0.9), index_df["duration_sec"].max()))
    print(sep)

    if DRY_RUN:
        print("\n[DRY RUN] No se escribió ningún archivo.")
        return index_df

    PULMONARY_CYCLES_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(PULMONARY_CYCLES_INDEX_CSV, index=False)
    print(f"\n✓ cycles_index.csv guardado en: {PULMONARY_CYCLES_INDEX_CSV}\n")
    return index_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--highpass", type=float, default=None,
                        help="Frecuencia de corte (Hz) de un pasa-altos opcional")
    make_pulmonary_cycles(parser.parse_args().highpass)
