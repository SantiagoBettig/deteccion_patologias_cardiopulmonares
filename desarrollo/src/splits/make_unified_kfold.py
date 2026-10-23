"""
splits/make_unified_kfold.py
===============================
Folds del modelo único sin gate (avances/21 §3.5). Los pacientes de CirCor
e ICBHI no se superponen, así que el fold i es la UNIÓN del fold i de
murmullo (cardiac_murmur_kfold/) y del fold i pulmonar (pulmonary_kfold/):
cada paciente queda en train/val/test en el mismo fold que en su rama, y la
comparación con la capa 2 es paciente a paciente.

No se sortea nada: solo se concatenan los CSV ya existentes y se verifica
que no haya pacientes repetidos y que cada uno pase por test una sola vez.

Salida: data/unified_v2/unified_kfold/fold{i}.csv (subject_id, split, domain)

Uso:
    python splits/make_unified_kfold.py
"""

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_MURMUR_KFOLD_DIR, MASTER_CSV_V2, PULMONARY_KFOLD_DIR, UNIFIED_KFOLD_DIR

K = 5


def make_unified_kfold(k: int = K) -> list[pd.DataFrame]:
    master = pd.read_csv(MASTER_CSV_V2)
    UNIFIED_KFOLD_DIR.mkdir(parents=True, exist_ok=True)
    fold_dfs = []

    for i in range(k):
        card = pd.read_csv(CARDIAC_MURMUR_KFOLD_DIR / f"fold{i}.csv").assign(domain="cardiac")
        pulm = pd.read_csv(PULMONARY_KFOLD_DIR / f"fold{i}.csv").assign(domain="pulmonary")
        fold_df = pd.concat([card, pulm], ignore_index=True)
        assert fold_df["subject_id"].is_unique, f"fold {i}: pacientes repetidos"
        assert set(fold_df["subject_id"]) == set(master["subject_id"]), f"fold {i}: no cubre el maestro"

        n_rec = master.merge(fold_df, on="subject_id").groupby(["domain", "split"]).size().unstack()
        print(f"\nFold {i}: grabaciones por dominio y split\n{n_rec.to_string()}")

        fold_csv = UNIFIED_KFOLD_DIR / f"fold{i}.csv"
        fold_df.to_csv(fold_csv, index=False)
        print(f"✓ {fold_csv}")
        fold_dfs.append(fold_df)

    tests = pd.concat([f[f["split"] == "test"] for f in fold_dfs])
    assert tests["subject_id"].is_unique and len(tests) == master["subject_id"].nunique()
    print(f"\nCada uno de los {len(tests)} pacientes pasa por test exactamente una vez.")
    return fold_dfs


if __name__ == "__main__":
    make_unified_kfold()
