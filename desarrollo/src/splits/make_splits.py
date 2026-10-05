"""
splits/make_splits.py
======================
Split train/val/test por subject_id, estratificado por pathology_label.

Un mismo sujeto nunca queda repartido entre splits (evita data leakage al
ventanear/recortar por ciclo más adelante). Las clases con menos de
MIN_SUBJECTS_TO_SPLIT sujetos van enteras a train, ya que no hay forma
significativa de repartirlas — se documenta como limitación conocida.

Este split conjunto (cardíaco+pulmonar) es solo para el modelo gate. Los
modelos de patología de la etapa 2 usan splits independientes por dominio
(ver make_cardiac_splits.py) — no reutilizan este, según lo decidido en
avances/9_revision_clases_etapa2.md.

Uso:
    python splits/make_splits.py
"""

import random
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MASTER_CSV_V2, SPLITS_CSV, SPLITS_REPORTS_DIR
from splits.split_utils import (
    RANDOM_SEED, MIN_SUBJECTS_TO_SPLIT,
    pick_subject_pathology, assign_splits_for_class,
)


def make_splits() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV_V2)

    # sound_type es constante por subject_id. pathology_label no siempre lo
    # es para CirCor (un sujeto con murmullo puede tener grabaciones en
    # ubicaciones donde no fue audible) — se agrega con PATHOLOGY_PRIORITY.
    subjects = df.groupby("subject_id").agg(
        pathology_label=("pathology_label", pick_subject_pathology),
        sound_type=("sound_type", "first"),
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
    print("  SPLIT TRAIN/VAL/TEST — RESUMEN")
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

    # ── Guardar splits.csv ───────────────────────────────────────────────────
    SPLITS_CSV.parent.mkdir(parents=True, exist_ok=True)
    subjects[["subject_id", "split"]].to_csv(SPLITS_CSV, index=False)
    print(f"\n✓ splits.csv guardado en: {SPLITS_CSV}")

    # ── Guardar resumen para el informe ──────────────────────────────────────
    SPLITS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    rec_table = pd.crosstab(df["pathology_label"], df["split"])
    summary = subj_table.add_suffix("_sujetos").join(rec_table.add_suffix("_registros"))
    summary.to_csv(SPLITS_REPORTS_DIR / "summary.csv")
    print(f"✓ Resumen guardado en: {SPLITS_REPORTS_DIR / 'summary.csv'}\n")

    return df


if __name__ == "__main__":
    make_splits()
