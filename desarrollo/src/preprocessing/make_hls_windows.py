"""
preprocessing/make_hls_windows.py
====================================
Genera ventanas preprocesadas de HLS-CMDS para usarlas como set de
VALIDACIÓN EXTERNA del modelo gate (nunca se usan para entrenar).

Por qué: en el dataset de entrenamiento (CirCor + ICBHI), `sound_type` está
perfectamente correlacionado con `source_db` (100% cardíaco = CirCor, 100%
pulmonar = ICBHI — ver avances/3_preliminar_eda.md). Existe el riesgo de que
el gate aprenda a distinguir la firma del equipo/entorno de grabación de
cada dataset en vez de la diferencia acústica real entre sonidos cardíacos
y pulmonares.

HLS-CMDS es el único dataset donde ambos tipos de sonido se grabaron con el
mismo equipo y en el mismo entorno (maniquí clínico). Si el gate entrenado
en CirCor+ICBHI clasifica bien estos audios, es evidencia de que aprendió la
diferencia acústica real; si falla, confirma el sesgo hacia la firma del
dataset de origen. En cualquier caso queda documentado como parte de la
validación del modelo.

Uso:
    python preprocessing/make_hls_windows.py
"""

import soundfile as sf
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unified_dataset"))

from common.paths import GATE_WINDOWS_DIR, GATE_HLS_INDEX_CSV
from parsers.parse_hls import parse_hls
from config import HLS_ROOT
from preprocessing.audio_ops import preprocess
from preprocessing.make_gate_windows import _slice_windows, TARGET_SR

EXTERNAL_SPLIT_NAME = "external_hls"
DRY_RUN = False


def make_hls_windows() -> pd.DataFrame:
    df, _ = parse_hls()

    index_rows = []
    errors = 0

    for _, row in df.iterrows():
        subfolder = "HS" if row["sound_type"] == "cardiac" else "LS"
        src_path = HLS_ROOT / subfolder / row["original_filename"]

        try:
            y, sr = preprocess(src_path, target_sr=TARGET_SR)
        except Exception as e:
            print(f"⚠ Error procesando {row['original_filename']}: {e}")
            errors += 1
            continue

        windows = _slice_windows(y)
        file_stem = Path(row["file_id"]).stem

        for w_idx, chunk in enumerate(windows):
            window_id = f"{file_stem}_{w_idx:03d}"
            index_rows.append({
                "window_id": window_id,
                "file_id": row["file_id"],
                "subject_id": row["subject_id"],
                "split": EXTERNAL_SPLIT_NAME,
                "sound_type": row["sound_type"],
                "source_db": row["source_db"],
                "pathology_label": row["pathology_label"],
            })

            if not DRY_RUN:
                out_dir = GATE_WINDOWS_DIR / EXTERNAL_SPLIT_NAME
                out_dir.mkdir(parents=True, exist_ok=True)
                sf.write(out_dir / f"{window_id}.wav", chunk, sr)

    index_df = pd.DataFrame(index_rows)

    sep = "─" * 55
    print(f"\n{sep}")
    print("  VENTANAS HLS-CMDS (VALIDACIÓN EXTERNA DEL GATE) — RESUMEN")
    print(sep)
    print(f"  Archivos procesados : {len(df)}")
    print(f"  Errores             : {errors}")
    print(f"  Ventanas generadas  : {len(index_df)}")
    print("\n  Ventanas por sound_type:")
    print(index_df["sound_type"].value_counts().to_string())
    print(sep)

    if DRY_RUN:
        print("\n[DRY RUN] No se escribió ningún archivo.")
        return index_df

    GATE_HLS_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(GATE_HLS_INDEX_CSV, index=False)
    print(f"\n✓ hls_external_index.csv guardado en: {GATE_HLS_INDEX_CSV}\n")

    return index_df


if __name__ == "__main__":
    make_hls_windows()
