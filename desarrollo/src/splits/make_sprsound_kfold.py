"""
splits/make_sprsound_kfold.py
================================
K folds por paciente sobre SPRSound (preprocessing/make_sprsound_windows.py),
con el mismo esquema que make_pulmonary_kfold.py: cada paciente pasa por
test una sola vez, y el resto de cada fold se reparte en train/val
(VAL_FRAC_OF_REMAINING por estrato, al menos 1 si hay >= 2).

Estratificado por la clase más "rara" que tiene cada paciente entre sus
ventanas (both > wheeze > crackle > normal): en SPRSound casi todos los
pacientes son anormales por diagnóstico (neumonía), así que estratificar por
diagnóstico, como en ICBHI, no reparte las ventanas anormales. Dentro de
cada estrato se ordena por cantidad de ventanas antes de repartir, para que
los pacientes con muchas grabaciones (hasta 83) no caigan juntos.

Salida: data/sprsound/kfold/fold{i}.csv (subject_id, split)

Uso:
    python splits/make_sprsound_kfold.py
"""

import random
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SPRSOUND_KFOLD_DIR, sprsound_windows_index
from splits.make_pulmonary_kfold import VAL_FRAC_OF_REMAINING
from splits.split_utils import RANDOM_SEED

K = 5
STRATA = {5: "both", 4: "wheeze", 3: "crackle", 2: "normal"}   # prioridad: la clase más rara


def make_sprsound_kfold(k: int = K) -> list[pd.DataFrame]:
    w = pd.read_csv(sprsound_windows_index(4.0))
    labels = w.groupby("subject_id")["label"].agg(set)
    subjects = pd.DataFrame({
        "stratum": labels.map(lambda s: next(STRATA[c] for c in STRATA if c in s)),
        "n_windows": w.groupby("subject_id").size(),
    }).reset_index()
    print(f"Pacientes: {len(subjects)}\n{subjects['stratum'].value_counts().to_string()}")

    rng = random.Random(RANDOM_SEED)
    groups: list[list[str]] = [[] for _ in range(k)]
    for _, class_df in subjects.groupby("stratum"):
        ids = class_df.sort_values("n_windows", ascending=False)["subject_id"].tolist()
        for block_start in range(0, len(ids), k):
            block = ids[block_start:block_start + k]
            rng.shuffle(block)
            for offset, sid in enumerate(block):
                groups[offset].append(sid)
        groups = groups[1:] + groups[:1]

    SPRSOUND_KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    fold_dfs = []
    for i in range(k):
        test_ids = set(groups[i])
        split_map = {sid: "test" for sid in test_ids}
        for _, class_df in subjects[~subjects["subject_id"].isin(test_ids)].groupby("stratum"):
            ids = class_df["subject_id"].tolist()
            rng.shuffle(ids)
            n_val = round(len(ids) * VAL_FRAC_OF_REMAINING)
            if len(ids) >= 2:
                n_val = max(1, n_val)
            split_map.update({sid: "val" for sid in ids[:n_val]})
            split_map.update({sid: "train" for sid in ids[n_val:]})
        fold_df = subjects.assign(split=subjects["subject_id"].map(split_map))
        win = w.merge(fold_df[["subject_id", "split"]], on="subject_id")
        print(f"\nFold {i}: ventanas por split y clase\n"
              f"{pd.crosstab(win['label_name'], win['split']).to_string()}")
        fold_df[["subject_id", "split"]].to_csv(SPRSOUND_KFOLD_DIR / f"fold{i}.csv", index=False)
        fold_dfs.append(fold_df)

    tests = pd.concat([f[f["split"] == "test"] for f in fold_dfs])
    assert tests["subject_id"].is_unique and len(tests) == len(subjects)
    print(f"\n✓ {SPRSOUND_KFOLD_DIR} — cada uno de los {len(tests)} pacientes pasa por test una vez")
    return fold_dfs


if __name__ == "__main__":
    make_sprsound_kfold()
