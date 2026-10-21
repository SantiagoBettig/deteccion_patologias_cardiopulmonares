"""
pulmonary_adventitious_model/calibration_study.py
====================================================
Estudio de calibración de umbrales (ver avances/17). Sin reentrenar nada:
usa las probabilidades de validación y de test ya guardadas de cada corrida
(val_probs_fold{i}.csv + oof_predictions.csv) y compara distintas formas de
elegir los dos umbrales (crackle, wheeze) sobre validación, midiendo el
score ICBHI resultante en test.

Motivación: con solo 16 pacientes de validación por fold, el método actual
(máximo exacto de una grilla 19x19 del score ICBHI) a veces elige umbrales
que no se trasladan a test. Caso extremo: fold 4 de la variante A (0.502 con
los umbrales de val, 0.580 con los ideales).

Métodos (fijados ANTES de mirar resultados, para no elegir a posteriori el
que mejor le quede a estos 5 folds):

    grid_max    el actual: máximo del score ICBHI en la grilla 19x19 (val).
    grid_smooth máximo de la superficie de score suavizada (promedio 3x3 en
                la grilla): prefiere mesetas estables a picos aislados, que
                con 16 pacientes suelen ser ruido.
    bootstrap   remuestrea los pacientes de val con reposición (200 veces),
                busca el máximo de la grilla en cada muestra y usa la mediana
                de los umbrales: incorpora explícitamente la variabilidad
                entre pacientes, que es la fuente del ruido.
    youden      cada etiqueta por separado, el umbral que maximiza
                sensibilidad + especificidad (binaria) en val: problema más
                simple (2 x 19 en vez de 361 combinaciones), menos ruidoso,
                pero no optimiza directamente el score de 4 clases.
    prevalence  cada etiqueta por separado, el umbral que hace que la
                proporción de ciclos predichos positivos en val iguale la
                proporción real de positivos en val.

    oracle      (solo referencia, no es un método válido) umbrales elegidos
                mirando test: el techo de lo que se puede ganar calibrando.

Criterio de lectura: un método reemplaza al actual solo si mejora de forma
CONSISTENTE entre corridas y folds (y en especial, si evita caídas como la
del fold 4), no por una diferencia chica en el promedio.

Uso:
    python pulmonary_adventitious_model/calibration_study.py --runs v4_hp100 v8_cnn6_ft_A v8_cnn6_ft_B v8_cnn6_ft_C
"""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PULMONARY_ADV_REPORTS_DIR
from pulmonary_adventitious_model.metrics import THRESHOLD_GRID, icbhi_score, predict_4class, to_4class
from pulmonary_adventitious_model.run_kfold import K

N_BOOTSTRAP = 200
METHODS = ["grid_max", "grid_smooth", "bootstrap", "youden", "prevalence", "oracle"]


def score_matrix(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Score ICBHI para cada par de umbrales de la grilla: (19, 19)."""
    true4 = to_4class(y[:, 0], y[:, 1])
    S = np.zeros((len(THRESHOLD_GRID), len(THRESHOLD_GRID)))
    crackle_pred = [p[:, 0] > t for t in THRESHOLD_GRID]
    for a, c in enumerate(crackle_pred):
        for b, tw in enumerate(THRESHOLD_GRID):
            S[a, b] = icbhi_score(true4, to_4class(c, p[:, 1] > tw))["icbhi_score"]
    return S


def argmax_thresholds(S: np.ndarray) -> tuple[float, float]:
    a, b = np.unravel_index(np.nanargmax(S), S.shape)
    return float(THRESHOLD_GRID[a]), float(THRESHOLD_GRID[b])


def choose(method: str, y_va, p_va, subj_va, y_te, p_te, rng) -> tuple[float, float]:
    if method == "grid_max":
        return argmax_thresholds(score_matrix(y_va, p_va))
    if method == "grid_smooth":
        return argmax_thresholds(uniform_filter(score_matrix(y_va, p_va), size=3, mode="nearest"))
    if method == "bootstrap":
        subjects = np.unique(subj_va)
        rows_by_subject = {s: np.flatnonzero(subj_va == s) for s in subjects}
        picks = []
        for _ in range(N_BOOTSTRAP):
            sample = rng.choice(subjects, size=len(subjects), replace=True)
            idx = np.concatenate([rows_by_subject[s] for s in sample])
            if y_va[idx].sum(0).min() == 0:   # sin positivos de alguna etiqueta: muestra inútil
                continue
            picks.append(argmax_thresholds(score_matrix(y_va[idx], p_va[idx])))
        return tuple(float(v) for v in np.median(np.array(picks), axis=0))
    if method == "youden":
        out = []
        for j in range(2):
            pos, neg = y_va[:, j] == 1, y_va[:, j] == 0
            j_stat = [(p_va[pos, j] > t).mean() + (p_va[neg, j] <= t).mean() for t in THRESHOLD_GRID]
            out.append(float(THRESHOLD_GRID[int(np.argmax(j_stat))]))
        return tuple(out)
    if method == "prevalence":
        return tuple(float(np.quantile(p_va[:, j], 1 - y_va[:, j].mean())) for j in range(2))
    if method == "oracle":
        return argmax_thresholds(score_matrix(y_te, p_te))
    raise ValueError(method)


def study(runs: list[str]) -> None:
    rng = np.random.default_rng(42)
    rows = []
    for run in runs:
        d = PULMONARY_ADV_REPORTS_DIR / run
        oof = pd.read_csv(d / "oof_predictions.csv")
        for i in range(K):
            va = pd.read_csv(d / f"val_probs_fold{i}.csv")
            te = oof[oof.fold == i]
            y_va, p_va = va[["crackle", "wheeze"]].to_numpy(), va[["p_crackle", "p_wheeze"]].to_numpy()
            y_te, p_te = te[["crackle", "wheeze"]].to_numpy(), te[["p_crackle", "p_wheeze"]].to_numpy()
            true4_te = to_4class(y_te[:, 0], y_te[:, 1])
            for method in METHODS:
                t = choose(method, y_va, p_va, va["subject_id"].to_numpy(), y_te, p_te, rng)
                s = icbhi_score(true4_te, predict_4class(p_te, t))
                rows.append({"run": run, "fold": i, "method": method, "t_crackle": t[0], "t_wheeze": t[1], **s})
        print(f"✓ {run}")

    df = pd.DataFrame(rows)
    oracle = df[df.method == "oracle"].set_index(["run", "fold"])["icbhi_score"]
    df["loss_vs_oracle"] = df.apply(lambda r: oracle[(r.run, r.fold)] - r.icbhi_score, axis=1)
    base = df[df.method == "grid_max"].set_index(["run", "fold"])["icbhi_score"]
    df["delta_vs_grid_max"] = df.apply(lambda r: r.icbhi_score - base[(r.run, r.fold)], axis=1)

    out_dir = PULMONARY_ADV_REPORTS_DIR / "calibration_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "per_run_fold_method.csv", index=False)

    summary = df.groupby(["run", "method"], sort=False).agg(
        score_mean=("icbhi_score", "mean"), score_std=("icbhi_score", "std"),
        worst_fold=("icbhi_score", "min"), loss_vs_oracle=("loss_vs_oracle", "mean")).reset_index()
    summary.to_csv(out_dir / "summary_by_run.csv", index=False)
    overall = df[df.method != "oracle"].groupby("method", sort=False).agg(
        delta_mean=("delta_vs_grid_max", "mean"),
        folds_better=("delta_vs_grid_max", lambda s: int((s > 0.001).sum())),
        folds_worse=("delta_vs_grid_max", lambda s: int((s < -0.001).sum())),
        worst_delta=("delta_vs_grid_max", "min"),
        loss_vs_oracle=("loss_vs_oracle", "mean"),
        max_loss_vs_oracle=("loss_vs_oracle", "max")).reset_index()
    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    print("\nScore ICBHI en test por corrida y método (media, desvío, peor fold, pérdida media vs oráculo):")
    print(summary.to_string(index=False))
    print(f"\nComparado con grid_max (el actual), sobre {len(runs)} corridas x {K} folds = {len(runs) * K} casos:")
    print(overall.to_string(index=False))
    print("\nScore por fold (filas: corrida/fold; columnas: método):")
    print(df.pivot_table(index=["run", "fold"], columns="method", values="icbhi_score")[METHODS].to_string())
    print(f"\n✓ Reportes en: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    study(parser.parse_args().runs)
