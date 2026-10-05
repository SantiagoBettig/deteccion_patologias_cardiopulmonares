"""
splits/split_utils.py
=======================
Utilidades de split compartidas entre el split conjunto del gate
(make_splits.py) y los splits independientes por dominio de la etapa 2
(make_cardiac_splits.py, y a futuro el pulmonar).

Reparte por subject_id (nunca se separa un mismo sujeto entre splits) y
estratifica por pathology_label. Las clases con menos de
MIN_SUBJECTS_TO_SPLIT sujetos van enteras a train.
"""

import random
import pandas as pd

RANDOM_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
MIN_SUBJECTS_TO_SPLIT = 3  # por debajo de esto, la clase entera va a train

# Para agregar pathology_label a nivel sujeto cuando no es constante entre
# sus registros (CirCor: un sujeto con murmullo puede tener grabaciones en
# ubicaciones donde no fue audible, ver unified_dataset/parsers/parse_circor.py)
# — gana la clase más específica.
PATHOLOGY_PRIORITY = ["heart_murmur", "abnormal_heart_unspecified", "normal_heart"]


def pick_subject_pathology(labels: pd.Series) -> str:
    values = set(labels)
    for label in PATHOLOGY_PRIORITY:
        if label in values:
            return label
    return labels.iloc[0]  # dominios sin prioridad definida: labels ya es constante


def assign_splits_for_class(subject_ids: list[str], rng: random.Random) -> dict[str, str]:
    """Reparte una lista de subject_ids de una misma clase en train/val/test."""
    n = len(subject_ids)
    shuffled = subject_ids[:]
    rng.shuffle(shuffled)

    if n < MIN_SUBJECTS_TO_SPLIT:
        return {sid: "train" for sid in shuffled}

    n_test = max(1, round(n * TEST_FRAC))
    n_val = max(1, round(n * VAL_FRAC))
    n_train = n - n_val - n_test

    assignment = {}
    for sid in shuffled[:n_train]:
        assignment[sid] = "train"
    for sid in shuffled[n_train:n_train + n_val]:
        assignment[sid] = "val"
    for sid in shuffled[n_train + n_val:]:
        assignment[sid] = "test"
    return assignment
