"""
cardiac_murmur_model/diagnostics_metadata_only.py
=====================================================
Diagnóstico de atajo demográfico — mismo tipo de chequeo que
gate_model/diagnostics_source_db.py (ahí se comprobó que el gate v1
aprendía a distinguir source_db en vez de sonido real, ver
avances/4_gate_model_baseline_y_sesgo.md).

Entrena un clasificador chico (regresión logística) que ve SOLO la
metadata del paciente (sexo, edad, altura, peso) — sin audio — y trata de
predecir heart_murmur. Si este diagnóstico predice bien, hay riesgo de que
la mejora del modelo multimodal (v4) venga de un atajo demográfico y no de
una interacción real con el audio. Si predice mal (lo esperado, según el
análisis de correlación cruda en avances/10, sección 8), es una señal de
que cualquier mejora de v4 es más confiable.

No usa PyTorch/audio — corre en segundos sobre el CSV de metadata solo.

Uso:
    python cardiac_murmur_model/diagnostics_metadata_only.py
"""

import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import CARDIAC_WINDOWS_INDEX_CSV
from cardiac_murmur_model.dataset import compute_meta_stats, encode_metadata


def run_diagnostic(index_csv=CARDIAC_WINDOWS_INDEX_CSV) -> dict:
    df = pd.read_csv(index_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    meta_stats = compute_meta_stats(train_df)

    def to_xy(sub_df):
        X = sub_df.apply(lambda row: encode_metadata(row, meta_stats), axis=1).tolist()
        y = (sub_df["pathology_label"] == "heart_murmur").astype(int).tolist()
        return X, y

    X_train, y_train = to_xy(train_df)
    X_val, y_val = to_xy(val_df)
    X_test, y_test = to_xy(test_df)

    # class_weight="balanced" — mismo espíritu que pos_weight en los modelos
    # de audio, para que la regresión no colapse en predecir "sin murmullo".
    best_c, best_f2 = 1.0, -1.0
    for c in [0.01, 0.1, 1.0, 10.0]:
        clf = LogisticRegression(class_weight="balanced", C=c, max_iter=1000)
        clf.fit(X_train, y_train)
        preds_val = clf.predict(X_val)
        r, p = recall_score(y_val, preds_val), precision_score(y_val, preds_val, zero_division=0)
        f2 = (5 * p * r / (4 * p + r)) if (4 * p + r) > 0 else 0.0
        if f2 > best_f2:
            best_f2, best_c = f2, c

    clf = LogisticRegression(class_weight="balanced", C=best_c, max_iter=1000)
    clf.fit(X_train, y_train)
    preds_test = clf.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, preds_test),
        "precision": precision_score(y_test, preds_test, zero_division=0),
        "recall": recall_score(y_test, preds_test, zero_division=0),
        "f1": f1_score(y_test, preds_test, zero_division=0),
    }

    print(f"Regresión logística solo-metadata (C={best_c}) — test:")
    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}")
    print(f"  (referencia: predecir siempre 'sin murmullo' da accuracy "
          f"{1 - sum(y_test)/len(y_test):.4f})")

    return metrics


if __name__ == "__main__":
    run_diagnostic()
