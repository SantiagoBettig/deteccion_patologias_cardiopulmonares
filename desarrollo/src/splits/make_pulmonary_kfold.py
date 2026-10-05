"""
splits/make_pulmonary_kfold.py
=================================
Genera K folds por paciente sobre el subconjunto pulmonar (ICBHI), para el
modelo de sonidos adventicios por ciclo (pulmonary_adventitious_model/).

A diferencia del modelo de murmullo — donde el k-fold se agregó al final,
después de que el split único mostrara una mejora que no se sostuvo
(avances/10, sección 14) — acá se usa k-fold desde la primera corrida: con
126 pacientes, un split único de test tendría ~19 pacientes y el resultado
dependería demasiado de cuáles tocaron.

Estratificado por pathology_label (constante por paciente en ICBHI), para
que cada fold tenga pacientes de todos los diagnósticos que alcancen: la
proporción de crackles/wheezes varía mucho por diagnóstico (COPD ~53% de
ciclos anormales, sanos ~6%), así que estratificar por diagnóstico también
estabiliza la proporción de etiquetas entre folds.

Para cada fold i (0..K-1):
    - test  = pacientes del grupo i
    - resto = pacientes de los otros K-1 grupos, repartidos en train/val
              (VAL_FRAC_OF_REMAINING por diagnóstico, al menos 1 si hay >= 2)

Cada paciente pasa por test exactamente una vez. Guarda un CSV por fold
(subject_id, split) en data/unified_v2/pulmonary_kfold/fold{i}.csv.

Uso:
    python splits/make_pulmonary_kfold.py
"""

import random
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import MASTER_CSV_V2, PULMONARY_KFOLD_DIR
from splits.split_utils import RANDOM_SEED

K = 5
VAL_FRAC_OF_REMAINING = 0.15


def make_pulmonary_kfold(k: int = K) -> list[pd.DataFrame]:
    df = pd.read_csv(MASTER_CSV_V2)
    df = df[df["sound_type"] == "pulmonary"]

    subjects = df.groupby("subject_id").agg(
        pathology_label=("pathology_label", "first"),
        n_recordings=("file_id", "count"),
    ).reset_index()
    print(f"Pacientes pulmonares: {len(subjects)}")

    rng = random.Random(RANDOM_SEED)

    # Por diagnóstico, repartir pacientes en k grupos (round-robin tras
    # mezclar). Se ordena de más a menos grabaciones antes de repartir para
    # que los pacientes de COPD con muchas grabaciones (hasta 66) no caigan
    # todos en el mismo grupo — mezcla dentro de bloques de k.
    groups: list[list[str]] = [[] for _ in range(k)]
    for _, class_df in subjects.groupby("pathology_label"):
        class_df = class_df.sort_values("n_recordings", ascending=False)
        ids = class_df["subject_id"].tolist()
        for block_start in range(0, len(ids), k):
            block = ids[block_start:block_start + k]
            rng.shuffle(block)
            for offset, sid in enumerate(block):
                groups[offset].append(sid)
        # rotar el orden de los grupos para que el sobrante de cada clase no
        # caiga siempre en los primeros grupos
        groups = groups[1:] + groups[:1]

    fold_dfs = []
    PULMONARY_KFOLD_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(k):
        test_ids = set(groups[i])
        remaining = subjects[~subjects["subject_id"].isin(test_ids)]

        split_map: dict[str, str] = {sid: "test" for sid in test_ids}
        for _, class_df in remaining.groupby("pathology_label"):
            ids = class_df["subject_id"].tolist()
            rng.shuffle(ids)
            n_val = round(len(ids) * VAL_FRAC_OF_REMAINING)
            if len(ids) >= 2:
                n_val = max(1, n_val)
            for sid in ids[:n_val]:
                split_map[sid] = "val"
            for sid in ids[n_val:]:
                split_map[sid] = "train"

        fold_df = subjects.copy()
        fold_df["split"] = fold_df["subject_id"].map(split_map)

        rec = fold_df.groupby("split")["n_recordings"].sum().to_dict()
        print(f"\nFold {i}: pacientes {fold_df['split'].value_counts().to_dict()} | "
              f"grabaciones {rec}")
        print(pd.crosstab(fold_df["pathology_label"], fold_df["split"]).to_string())

        fold_csv = PULMONARY_KFOLD_DIR / f"fold{i}.csv"
        fold_df[["subject_id", "split"]].to_csv(fold_csv, index=False)
        print(f"✓ {fold_csv}")
        fold_dfs.append(fold_df)

    return fold_dfs


if __name__ == "__main__":
    make_pulmonary_kfold()
