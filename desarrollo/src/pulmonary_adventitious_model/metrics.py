"""
pulmonary_adventitious_model/metrics.py
==========================================
Métricas del modelo de sonidos adventicios.

Métrica principal — **score oficial del challenge ICBHI 2017**, sobre las 4
clases por ciclo (normal / crackle / wheeze / both):

    Sp    = ciclos normales predichos normales / total de ciclos normales
    Se    = ciclos anormales predichos con la clase EXACTA / total de anormales
    Score = (Se + Sp) / 2

(Un ciclo "both" predicho como "crackle" cuenta como error para Se.) Es la
métrica que reporta la literatura sobre ICBHI, así que permite ubicar el
resultado.

Además se reportan, por etiqueta (crackle, wheeze): AUC (independiente del
umbral — la mejor medida de separabilidad pura), precision, recall, F1.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

CLASS_NAMES = ["normal", "crackle", "wheeze", "both"]
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def to_4class(crackle: np.ndarray, wheeze: np.ndarray) -> np.ndarray:
    """(0,0)->0 normal, (1,0)->1 crackle, (0,1)->2 wheeze, (1,1)->3 both."""
    return crackle.astype(int) + 2 * wheeze.astype(int)


def icbhi_score(true4: np.ndarray, pred4: np.ndarray) -> dict:
    normal = true4 == 0
    sp = float((pred4[normal] == 0).mean()) if normal.any() else float("nan")
    se = float((pred4[~normal] == true4[~normal]).mean()) if (~normal).any() else float("nan")
    return {"se": se, "sp": sp, "icbhi_score": (se + sp) / 2}


def predict_4class(probs: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    return to_4class(probs[:, 0] > thresholds[0], probs[:, 1] > thresholds[1])


def calibrate_thresholds(labels: np.ndarray, probs: np.ndarray) -> tuple[tuple[float, float], float]:
    """
    Busca el par (umbral_crackle, umbral_wheeze) que maximiza el score ICBHI
    sobre una grilla (19 x 19). Se hace en conjunto, no por separado, porque
    el score depende de acertar la combinación exacta. Usar solo sobre
    validación.
    """
    true4 = to_4class(labels[:, 0], labels[:, 1])
    best, best_score = (0.5, 0.5), -1.0
    for tc in THRESHOLD_GRID:
        c = probs[:, 0] > tc
        for tw in THRESHOLD_GRID:
            s = icbhi_score(true4, to_4class(c, probs[:, 1] > tw))["icbhi_score"]
            if s > best_score:
                best_score, best = s, (float(tc), float(tw))
    return best, best_score


def full_metrics(labels: np.ndarray, probs: np.ndarray, thresholds: tuple[float, float]) -> dict:
    """Score ICBHI (4 clases) + métricas por etiqueta, con los umbrales dados."""
    true4 = to_4class(labels[:, 0], labels[:, 1])
    out = icbhi_score(true4, predict_4class(probs, thresholds))
    out["accuracy_4class"] = float((predict_4class(probs, thresholds) == true4).mean())
    for j, name in enumerate(["crackle", "wheeze"]):
        y, p = labels[:, j], probs[:, j]
        pred = (p > thresholds[j]).astype(int)
        out[f"{name}_auc"] = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
        out[f"{name}_precision"] = precision_score(y, pred, zero_division=0)
        out[f"{name}_recall"] = recall_score(y, pred, zero_division=0)
        out[f"{name}_f1"] = f1_score(y, pred, zero_division=0)
        out[f"{name}_threshold"] = thresholds[j]
    return out
