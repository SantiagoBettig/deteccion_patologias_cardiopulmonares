"""
unified_model/evaluate_sprsound.py
=====================================
Evaluación del paso 3 de SPRSound (avances/23): la rama pulmonar chica
(receta de abl_pulm) entrenada solo con SPRSound (run_kfold.py --data
sprsound), en dos tests:

- SPRSound: sus pacientes de test de cada fold (data/sprsound/kfold).
- ICBHI (test cruzado): las ventanas pulmonares de test de ICBHI del mismo
  fold (role "test_icbhi"). El modelo nunca vio ICBHI: mide si lo aprendido
  se transfiere.

Métricas, todas con el mismo código para la referencia --ref (abl_pulm,
ICBHI → ICBHI):

- AUC por ventana de crackle y wheeze (como run_kfold.window_metrics).
- AUC normal vs anormal por grabación (avances/22 §6, la que tiene que
  moverse): score = media sobre las ventanas de 1 − P(pulm_normal). El
  script del §6 quedó fuera del repo; con esta definición abl_pulm da 0.532
  (el §6 reporta 0.540).
- Score ICBHI de capa 2 (evaluate_integration.evaluate_branch, umbrales y
  regla de agregación calibrados en val). En el test cruzado se calibra con
  el val de SPRSound: ningún dato de ICBHI elige nada. En las corridas
  conjuntas (run_kfold.py --data joint) ICBHI se calibra con su val_icbhi.
- Test "conjunto": el test de SPRSound y el de ICBHI juntos, como una sola
  base (calibrado con el val de las dos en las corridas conjuntas). Lo
  domina SPRSound (~77% de las grabaciones) y premia separar las bases por
  equipo, así que se lee al lado de los dos por separado.

Por grabación de SPRSound, has_crackles/has_wheezes salen de todos sus
eventos (data/sprsound/recordings.csv).

Salida: reports/sprsound/eval/
    summary.csv               una fila por (modelo, test)
    per_fold.csv              AUC por fold
    sprsound_by_release.csv   AUC normal vs anormal por año de SPRSound

Uso:
    python unified_model/evaluate_sprsound.py [--runs spr_inv spr_sqrt] [--ref abl_pulm]
"""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SPRSOUND_RECORDINGS_CSV, SPRSOUND_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import PULMONARY, aggregate, evaluate_branch, load_recordings
from unified_model.evaluate_unified import load_window_probs, window_thresholds

K = 5
STRATEGY = "pooled_fixed"
PULM_COLS = ["p_crackle", "p_wheeze"]


def auc(y, s) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def recording_scores(w: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (grabación, fold): score normal vs anormal y etiqueta."""
    g = w.groupby(["file_id", "subject_id", "fold"])
    r = (1 - g["p_pulm_normal"].mean()).rename("score").reset_index()
    r = r.merge(labels[["file_id", "has_crackles", "has_wheezes"]], on="file_id")
    r["abnormal"] = ((r["has_crackles"] + r["has_wheezes"]) > 0).astype(int)
    return r


def evaluate_test(name: str, test: str, w_cal: pd.DataFrame, w_test: pd.DataFrame,
                  labels: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """w_cal: ventanas de val (calibran umbrales y regla); w_test: ventanas
    de test (role "test"). labels: file_id -> has_crackles/has_wheezes."""
    per_fold = []
    for f in range(K):
        t = w_test[w_test.fold == f]
        crackle, wheeze = t["label"].isin([3, 5]), t["label"].isin([4, 5])
        r = recording_scores(t, labels)
        per_fold.append({"model": name, "test": test, "fold": f, "n_recordings": len(r),
                         "auc_crackle_window": auc(crackle, t["p_crackle"]),
                         "auc_wheeze_window": auc(wheeze, t["p_wheeze"]),
                         "auc_normal_vs_abnormal": auc(r["abnormal"], r["score"])})
    per_fold = pd.DataFrame(per_fold)

    wthr, _ = window_thresholds(w_cal)
    w = pd.concat([w_cal, w_test], ignore_index=True)
    rec = aggregate(w, PULM_COLS, wthr).merge(labels[["file_id"] + PULMONARY.labels], on="file_id")
    branch = evaluate_branch(rec, PULMONARY, STRATEGY, f"{name} → {test}")["per_fold"]
    branch = branch[branch["rule"] == "auto"]

    pooled = recording_scores(w_test, labels)
    row = {"model": name, "test": test, "n_recordings": len(pooled),
           "auc_normal_vs_abnormal_pooled": auc(pooled["abnormal"], pooled["score"]),
           "rule": "/".join(branch["chosen"])}
    for c in ["auc_crackle_window", "auc_wheeze_window", "auc_normal_vs_abnormal"]:
        row[c] = f"{per_fold[c].mean():.3f} ± {per_fold[c].std():.3f}"
    for c in ["icbhi_score", "se", "sp"]:
        row[c] = f"{branch[c].mean():.3f} ± {branch[c].std():.3f}"
    return row, per_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=["spr_inv", "spr_sqrt"])
    parser.add_argument("--ref", default="abl_pulm")
    args = parser.parse_args()
    out = SPRSOUND_REPORTS_DIR / "eval"
    out.mkdir(parents=True, exist_ok=True)

    spr_labels = pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    icb_labels = load_recordings()
    both_labels = pd.concat([d[["file_id"] + PULMONARY.labels] for d in (spr_labels, icb_labels)], ignore_index=True)
    rows, folds, by_release = [], [], []

    ref = load_window_probs(UNIFIED_MODEL_REPORTS_DIR / "runs" / args.ref)
    r, f = evaluate_test(args.ref, "ICBHI", ref[ref.role == "val"], ref[ref.role == "test"], icb_labels)
    rows.append(r)
    folds.append(f)

    for run in args.runs:
        w = load_window_probs(SPRSOUND_REPORTS_DIR / "runs" / run)
        joint_val = (w.role == "val_icbhi").any()
        for test, roles, labels in (("SPRSound", ["test"], spr_labels), ("ICBHI", ["test_icbhi"], icb_labels),
                                    ("conjunto", ["test", "test_icbhi"], both_labels)):
            # con --data joint, ICBHI se calibra con su propio val (y el conjunto con
            # los dos); si no, todo con el de SPRSound
            cal = {"SPRSound": ["val"], "ICBHI": ["val_icbhi"], "conjunto": ["val", "val_icbhi"]}[test] \
                if joint_val else ["val"]
            r, f = evaluate_test(run, test, w[w.role.isin(cal)].assign(role="val"),
                                 w[w.role.isin(roles)].assign(role="test"), labels)
            rows.append(r)
            folds.append(f)
        sc = recording_scores(w[w.role == "test"], spr_labels).merge(spr_labels[["file_id", "release"]], on="file_id")
        for rel, g in sc.groupby("release"):
            by_release.append({"model": run, "release": rel, "n_recordings": len(g),
                               "abnormal_frac": g["abnormal"].mean(), "auc_normal_vs_abnormal": auc(g["abnormal"], g["score"])})

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(folds).to_csv(out / "per_fold.csv", index=False)
    rel = pd.DataFrame(by_release)
    rel.to_csv(out / "sprsound_by_release.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(f"\nAUC normal vs anormal por año de SPRSound (test, todos los folds)\n{rel.round(3).to_string(index=False)}")
    print(f"\n✓ {out}")


if __name__ == "__main__":
    main()
