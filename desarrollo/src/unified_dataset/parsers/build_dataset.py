"""
parsers/build_dataset.py
========================
Script integrador. Ejecutar este archivo para construir el dataset unificado.

Hace tres cosas:
  1. Llama a los parsers de CirCor e ICBHI y concatena los DataFrames.
     HLS-CMDS se descartó del dataset unificado: sus clases (excepto normal)
     están grabadas en maniquí y con muy pocos casos por clase (ver avances/3).
     El parser sigue disponible en parse_hls.py para uso standalone.
  2. Copia los WAVs originales a data/unified/audio/ con el nuevo file_id.
  3. Guarda master.csv en data/unified/.

Uso:
    python parsers/build_dataset.py

Flags opcionales (editar las constantes al inicio):
    COPY_WAVS   : bool — si False, solo genera el CSV sin copiar audio.
    DRY_RUN     : bool — si True, muestra estadísticas pero no escribe nada.
"""

import shutil
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CIRCOR_ROOT, ICBHI_ROOT,
    UNIFIED_DIR, AUDIO_DIR, MASTER_CSV,
    MASTER_COLUMNS,
)
from parsers.parse_circor  import parse_circor
from parsers.parse_icbhi   import parse_icbhi

# ── Flags de control ──────────────────────────────────────────────────────────
COPY_WAVS = True    # Poner en False para solo generar el CSV
DRY_RUN   = False   # Poner en True para ver stats sin escribir nada


# ── Mapeo de file_id → ruta del WAV original ──────────────────────────────────
def _build_source_path_lookup(
    df_cir: pd.DataFrame,
    df_icb: pd.DataFrame,
) -> dict[str, Path]:
    """
    Construye un diccionario {file_id: Path al WAV original}.
    Se usa para copiar los WAVs al directorio unificado.
    """
    lookup = {}

    # CirCor
    for _, row in df_cir.iterrows():
        src = CIRCOR_ROOT / "training_data" / row["original_filename"]
        lookup[row["file_id"]] = src

    # ICBHI
    for _, row in df_icb.iterrows():
        src = ICBHI_ROOT / "audio_and_txt_files" / row["original_filename"]
        lookup[row["file_id"]] = src

    return lookup


def _copy_wavs(df: pd.DataFrame, lookup: dict[str, Path]) -> None:
    """Copia WAVs originales a AUDIO_DIR con el nuevo nombre file_id."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    missing = 0
    for _, row in df.iterrows():
        file_id = row["file_id"]
        src = lookup.get(file_id)
        dst = AUDIO_DIR / file_id

        if src is None or not src.exists():
            missing += 1
            continue
        if not dst.exists():
            shutil.copy2(src, dst)

    if missing:
        print(f"⚠ {missing} WAVs no encontrados en origen (ver paths en config.py).")


def _print_summary(df: pd.DataFrame) -> None:
    """Imprime un resumen estadístico del dataset unificado."""
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  DATASET UNIFICADO — RESUMEN")
    print(sep)
    print(f"  Total grabaciones : {len(df)}")
    print(f"  Columnas          : {len(df.columns)}")
    print()

    print("  Por base de datos:")
    for db, count in df["source_db"].value_counts().items():
        print(f"    {db}: {count}")

    print("\n  Por tipo de sonido:")
    for t, count in df["sound_type"].value_counts().items():
        print(f"    {t}: {count}")

    print("\n  Por subject_type:")
    for t, count in df["subject_type"].value_counts().items():
        print(f"    {t}: {count}")

    print("\n  Por pathology_label:")
    for label, count in df["pathology_label"].value_counts().items():
        print(f"    {label:<35} {count}")

    print("\n  NaN por columna (columnas con datos faltantes):")
    nan_counts = df.isnull().sum()
    for col, n in nan_counts[nan_counts > 0].items():
        pct = n / len(df) * 100
        print(f"    {col:<30} {n:>5}  ({pct:.1f}%)")
    print(sep)


def build_dataset() -> pd.DataFrame:
    print("=" * 55)
    print("  BUILD DATASET — inicio")
    print("=" * 55)

    # ── 1. Parsear cada fuente ────────────────────────────────────────────────
    df_cir, c1 = parse_circor(counter_start=0)
    df_icb, _  = parse_icbhi(counter_start=c1)

    # ── 2. Concatenar ─────────────────────────────────────────────────────────
    df = pd.concat([df_cir, df_icb], ignore_index=True)
    df = df[MASTER_COLUMNS]  # garantizar orden canónico de columnas

    if (df["original_filename"].duplicated().sum() == 0):
        print("\nNo existen archivos duplicados")

    # ── 3. Resumen ────────────────────────────────────────────────────────────
    _print_summary(df)

    if DRY_RUN:
        print("\n[DRY RUN] No se escribió ningún archivo.")
        return df

    # ── 4. Guardar CSV maestro ────────────────────────────────────────────────
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_CSV, index=False, encoding="utf-8")
    print(f"\n✓ master.csv guardado en: {MASTER_CSV}")

    # ── 5. Copiar WAVs ────────────────────────────────────────────────────────
    if COPY_WAVS:
        print(f"Copiando WAVs a {AUDIO_DIR} ...")
        lookup = _build_source_path_lookup(df_cir, df_icb)
        _copy_wavs(df, lookup)
        print("✓ Copia de WAVs completada.")
    else:
        print("COPY_WAVS=False — WAVs no copiados.")

    print("\n✓ Dataset unificado listo.\n")
    return df


if __name__ == "__main__":
    build_dataset()
