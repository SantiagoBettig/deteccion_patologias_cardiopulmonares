"""
splits/make_cardiac_splits.py
================================
Split train/val/test independiente para el modelo de patología cardíaca
(etapa 2) — por subject_id, estratificado por pathology_label, solo sobre
el subconjunto sound_type=="cardiac" del dataset unificado.

No reutiliza el split conjunto del gate (make_splits.py): con un modelo por
dominio ya no hace falta un vocabulario de clases compartido con pulmonar,
según lo decidido en avances/9_revision_clases_etapa2.md.

Uso:
    python splits/make_cardiac_splits.py
"""

import random
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MASTER_CSV_V2, CARDIAC_SPLITS_CSV, CARDIAC_SPLITS_REPORTS_DIR
from splits.split_utils import (
    RANDOM_SEED, MIN_SUBJECTS_TO_SPLIT,
    pick_subject_pathology, assign_splits_for_class,
)


def make_cardiac_splits() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV_V2)
    df = df[df["sound_type"] == "cardiac"].reset_index(drop=True)

    # pathology_label no es constante por subject_id para sujetos con
    # murmullo parcial (audible solo en algunas ubicaciones, ver
    # unified_dataset/parsers/parse_circor.py) — se agrega con prioridad.
    subjects = df.groupby("subject_id").agg(
        pathology_label=("pathology_label", pick_subject_pathology),
        n_records=("file_id", "count"),
    ).reset_index()

    rng = random.Random(RANDOM_SEED)
    split_map: dict[str, str] = {}
    warnings = []

    for label, group in subjects.groupby("pathology_label"):
        subject_ids = group["subject_id"].tolist()
        if len(subject_ids) < MIN_SUBJECTS_TO_SPLIT:
            warnings.append(
                f"  ⚠ '{label}': solo {len(subject_ids)} sujeto(s) — "
                f"van todos a train, no se puede estratificar."
            )
        split_map.update(assign_splits_for_class(subject_ids, rng))

    subjects["split"] = subjects["subject_id"].map(split_map)
    df["split"] = df["subject_id"].map(split_map)

    # ── Resumen ───────────────────────────────────────────────────────────────
    sep = "─" * 55
    print(f"\n{sep}")
    print("  SPLIT CARDÍACO TRAIN/VAL/TEST — RESUMEN")
    print(sep)
    print("\n  Sujetos por split:")
    print(subjects["split"].value_counts().to_string())

    print("\n  Registros por split:")
    print(df["split"].value_counts().to_string())

    print("\n  Sujetos por split x pathology_label:")
    subj_table = pd.crosstab(subjects["pathology_label"], subjects["split"])
    print(subj_table.to_string())

    if warnings:
        print("\n  Advertencias:")
        for w in warnings:
            print(w)
    print(sep)

    # ── Guardar cardiac_splits.csv ───────────────────────────────────────────
    CARDIAC_SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    subjects[["subject_id", "split"]].to_csv(CARDIAC_SPLITS_CSV, index=False)
    print(f"\n✓ cardiac_splits.csv guardado en: {CARDIAC_SPLITS_CSV}")

    # ── Guardar resumen para el informe ──────────────────────────────────────
    CARDIAC_SPLITS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rec_table = pd.crosstab(df["pathology_label"], df["split"])
    summary = subj_table.add_suffix("_sujetos").join(rec_table.add_suffix("_registros"))
    summary.to_csv(CARDIAC_SPLITS_REPORTS_DIR / "summary.csv")
    print(f"✓ Resumen guardado en: {CARDIAC_SPLITS_REPORTS_DIR / 'summary.csv'}\n")

    return df


if __name__ == "__main__":
    make_cardiac_splits()
