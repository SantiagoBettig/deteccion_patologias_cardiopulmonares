"""
unified_model/filter_variants.py
===================================
Filtro de variantes del modelo de 6 clases (avances/25): compara corridas
en los folds del filtro (0 y 3) SOLO con validación, sin mirar el test.

Métrica de decisión (acordada con el usuario): media de F2 de murmullo y
score pulmonar por grabación, con las reglas de agregación y umbrales de
evaluate_integration calibrados en las grabaciones de val de cada fold. El
valor es optimista (calibra y mide sobre lo mismo), pero igual para todas
las variantes. Como en evaluate_circor_spr, las grabaciones de SPRSound pasan
completas por el checkpoint del fold.

Salida: reports/unified_model/circor_spr/filter_folds03.csv

Uso:
    python unified_model/filter_variants.py --runs cs_base cs_wide cs_res2
"""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SPRSOUND_RECORDINGS_CSV, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import MURMUR, PULMONARY, RULES, aggregate, best_f2_threshold, load_recordings
from pulmonary_adventitious_model.metrics import calibrate_thresholds
from unified_model.evaluate_circor_spr import FULL_COLS, OUT_DIR
from unified_model.evaluate_full_recordings import infer, test_sets

FOLDS = (0, 3)


def fold_scores(run: str, fold: int, card: pd.DataFrame, pulm: pd.DataFrame, recs: pd.DataFrame,
                spr: pd.DataFrame) -> dict:
    """Mejor regla en val de cada tarea, con sus umbrales por ventana (como evaluate_unified)."""
    c, p = card[card.fold == fold], pulm[pulm.fold == fold]
    t_m, _ = best_f2_threshold(c["label"] == 1, c["prob"])
    idx = p[p["from_index"]]
    labels = np.stack([idx["label"].isin([3, 5]), idx["label"].isin([4, 5])], axis=1).astype(int)
    (t_c, t_w), _ = calibrate_thresholds(labels, idx[["p_crackle", "p_wheeze"]].to_numpy())
    rec_m = aggregate(c, ["prob"], {"prob": {fold: t_m}}).merge(recs[["file_id", "murmur"]], on="file_id")
    rec_p = (aggregate(p[~p["from_index"]], ["p_crackle", "p_wheeze"], {"p_crackle": {fold: t_c}, "p_wheeze": {fold: t_w}})
             .merge(spr[["file_id"] + PULMONARY.labels], on="file_id"))
    f2 = max(MURMUR.calibrate(rec_m, r)[1] for r in RULES)
    score = max(PULMONARY.calibrate(rec_p, r)[1] for r in RULES)
    return {"run": run, "fold": fold, "f2_murmur": f2, "score_pulm": score, "decision": (f2 + score) / 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args()
    sets = test_sets()
    recs, spr = load_recordings(), pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    rows = []
    for run in args.runs:
        run_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / run
        w = pd.concat([pd.read_csv(run_dir / f"window_probs_fold{f}.csv") for f in FOLDS], ignore_index=True)
        w = w[w.role == "val"].rename(columns={"p_murmur": "prob"})
        full = infer(run_dir, run, "SPRSound", sets, None, FULL_COLS, folds=FOLDS).query("role == 'val'")
        full = full.assign(domain="pulmonary", from_index=False)
        index = w[w.domain == "pulmonary"].assign(from_index=True)
        pulm = pd.concat([full, index], ignore_index=True)   # el índice solo calibra los umbrales por ventana
        card = w[w.domain == "cardiac"]
        for f in FOLDS:
            rows.append(fold_scores(run, f, card, pulm, recs, spr))
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "filter_folds03.csv", index=False)
    print(out.round(4).to_string(index=False))
    print("\nMedia de los dos folds:")
    print(out.groupby("run", sort=False)[["f2_murmur", "score_pulm", "decision"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
