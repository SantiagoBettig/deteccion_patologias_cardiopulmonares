"""
splits/make_cardiac_murmur_kfold.py
=======================================
Genera K folds estratificados por sujeto (por pathology_label) sobre el
subconjunto cardíaco, para validar de forma más confiable la versión
ganadora del modelo de murmullo (v1) — ver
avances/10_cardiaco_split_dos_tareas.md, sección 14: con test/val de un
solo split teniendo solo ~27 sujetos con murmullo cada uno, una diferencia
entre versiones podía deberse a qué sujetos "tocaron" en el split, no a
una mejora real.

Para cada fold i (0..K-1):
    - test  = sujetos del grupo i
    - resto = sujetos de los otros K-1 grupos, repartidos en train/val
              (misma proporción que splits/split_utils.py)

Cada sujeto pasa por el rol de test exactamente una vez a lo largo de los
K folds. Guarda un CSV por fold (subject_id, split) en
data/unified_v2/cardiac_murmur_kfold/fold{i}.csv.

Uso:
    python splits/make_cardiac_murmur_kfold.py
"""

import random
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MASTER_CSV_V2, UNIFIED_V2_DIR
from splits.split_utils import RANDOM_SEED, pick_subject_pathology

K = 5
VAL_FRAC_OF_REMAINING = 0.15  # sobre los sujetos que no son test en este fold

KFOLD_DIR = UNIFIED_V2_DIR / "cardiac_murmur_kfold"


def make_kfold_assignments(k: int = K) -> list[pd.DataFrame]:
    df = pd.read_csv(MASTER_CSV_V2)
    df = df[df["sound_type"] == "cardiac"].reset_index(drop=True)

    subjects = df.groupby("subject_id").agg(
        pathology_label=("pathology_label", pick_subject_pathology),
    ).reset_index()

    rng = random.Random(RANDOM_SEED)

    # Por clase, repartir sujetos en k grupos casi iguales (estratificado).
    groups: list[list[str]] = [[] for _ in range(k)]
    for label, class_df in subjects.groupby("pathology_label"):
        ids = class_df["subject_id"].tolist()
        rng.shuffle(ids)
        for idx, sid in enumerate(ids):
            groups[idx % k].append(sid)

    print("Sujetos por grupo (antes de armar cada fold):")
    for i, g in enumerate(groups):
        print(f"  grupo {i}: {len(g)} sujetos")

    fold_dfs = []
    KFOLD_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(k):
        test_ids = set(groups[i])
        remaining_ids = [sid for j in range(k) if j != i for sid in groups[j]]

        # split train/val estratificado sobre "remaining", por clase
        remaining_df = subjects[subjects["subject_id"].isin(remaining_ids)]
        split_map: dict[str, str] = {}
        for label, class_df in remaining_df.groupby("pathology_label"):
            ids = class_df["subject_id"].tolist()
            rng.shuffle(ids)
            n_val = max(1, round(len(ids) * VAL_FRAC_OF_REMAINING))
            for sid in ids[:n_val]:
                split_map[sid] = "val"
            for sid in ids[n_val:]:
                split_map[sid] = "train"
        for sid in test_ids:
            split_map[sid] = "test"

        fold_df = subjects[["subject_id"]].copy()
        fold_df["split"] = fold_df["subject_id"].map(split_map)
        fold_df["pathology_label"] = subjects["pathology_label"]

        counts = fold_df["split"].value_counts().to_dict()
        print(f"\nFold {i}: train={counts.get('train', 0)} val={counts.get('val', 0)} "
              f"test={counts.get('test', 0)}")
        print(pd.crosstab(fold_df["pathology_label"], fold_df["split"]).to_string())

        fold_csv = KFOLD_DIR / f"fold{i}.csv"
        fold_df[["subject_id", "split"]].to_csv(fold_csv, index=False)
        print(f"✓ {fold_csv}")
        fold_dfs.append(fold_df)

    return fold_dfs


if __name__ == "__main__":
    make_kfold_assignments()
