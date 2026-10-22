"""
integration/recalibrate_hls.py
=================================
¿Cuánto se recupera en HLS-CMDS recalibrando solo los UMBRALES con unas
pocas grabaciones etiquetadas del dominio nuevo, sin reentrenar? (avances/20
§3.4 y §5, avances/21).

Partición sin fuga: HLS tiene dos maniquíes (femenino "F_" y masculino
"M_"). Se calibra con uno y se evalúa con el otro, en las dos direcciones.
Se recalibra:
- el umbral del gate (probabilidad media por grabación), con la máxima
  accuracy balanceada sobre HS + LS del maniquí de calibración;
- el umbral de murmullo de cada modelo de fold, con la regla congelada
  ("frac"): con máximo F2 (el mismo criterio que in-domain) y con máxima
  accuracy balanceada (con 48% de murmullos, F2 premia predecir todo
  positivo);
- los umbrales crackle/wheeze de cada modelo de fold, con la regla
  congelada ("max"), con máximo score ICBHI (sobre las grabaciones de LS con
  equivalente en ICBHI).
La regla de agregación y los modelos no cambian. Se compara contra los
umbrales congelados de avances/20 sobre las mismas mitades de test.

Lee reports/integration/external_hls/recordings_by_fold.csv (salida de
external_hls.py): no corre ningún modelo.

Uso:
    python integration/recalibrate_hls.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import fbeta_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from integration.external_hls import OUT_DIR
from integration.evaluate_integration import GRID, K, icbhi_from_4class
from pulmonary_adventitious_model.metrics import calibrate_thresholds, to_4class


def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean())


def best_threshold(y, score, criterion: str) -> float:
    def value(t):
        if criterion == "f2":
            return fbeta_score(y, score > t, beta=2.0, zero_division=0)
        return balanced_accuracy(y, score > t)
    return float(max(GRID, key=value))


def murmur_metrics(y, pred) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {"recall": pred[y].mean(), "specificity": (~pred[~y]).mean(),
            "precision": y[pred].mean() if pred.any() else np.nan,
            "f2": fbeta_score(y, pred, beta=2.0, zero_division=0), "balanced_acc": balanced_accuracy(y, pred)}


def pulmonary_metrics(d: pd.DataFrame, pc, pw) -> dict:
    true4 = to_4class(d.has_crackles.astype(int).to_numpy(), d.has_wheezes.astype(int).to_numpy())
    pred4 = to_4class(np.asarray(pc).astype(int), np.asarray(pw).astype(int))
    out = icbhi_from_4class(true4, pred4)
    out["crackle_recall"] = np.asarray(pc)[d.has_crackles.to_numpy() == 1].mean()
    out["wheeze_recall"] = np.asarray(pw)[d.has_wheezes.to_numpy() == 1].mean()
    return out


def main() -> None:
    df = pd.read_csv(OUT_DIR / "recordings_by_fold.csv")
    df = df[df["set"].isin(["HS", "LS"])].copy()
    df["maniqui"] = df["audio_id"].str.split("/").str[1].str[0]
    gate_rec = df.drop_duplicates("audio_id")

    rows, gate_rows = [], []
    for cal_sex, test_sex in [("F", "M"), ("M", "F")]:
        direction = f"calibra {cal_sex} -> evalúa {test_sex}"

        # ── Gate (un solo modelo) ─────────────────────────────────────────────
        gc, gt = gate_rec[gate_rec.maniqui == cal_sex], gate_rec[gate_rec.maniqui == test_sex]
        t_gate = best_threshold(gc.sound_type == "cardiac", gc.gate_mean, "balanced")
        for name, t in [("congelado (0.5)", 0.5), (f"recalibrado ({t_gate:.2f})", t_gate)]:
            routed = np.where(gt.gate_mean > t, "cardiac", "pulmonary")
            gate_rows.append({"direccion": direction, "umbral": name,
                              "acc_cardiacas": (routed[gt.sound_type == "cardiac"] == "cardiac").mean(),
                              "acc_pulmonares": (routed[gt.sound_type == "pulmonary"] == "pulmonary").mean()})

        # ── Ramas, por modelo de fold ─────────────────────────────────────────
        for k in range(K):
            d = df[df.fold == k]
            hs_c, hs_t = d[(d.set == "HS") & (d.maniqui == cal_sex)], d[(d.set == "HS") & (d.maniqui == test_sex)]
            ls = d[(d.set == "LS") & (d.lung_icbhi)]
            ls_c, ls_t = ls[ls.maniqui == cal_sex], ls[ls.maniqui == test_sex]
            base = {"direccion": direction, "fold": k}

            rows.append({**base, "rama": "murmullo", "umbral": "congelado", **murmur_metrics(hs_t.murmur, hs_t.pred)})
            for crit in ("f2", "balanced"):
                t = best_threshold(hs_c.murmur, hs_c.score, crit)
                rows.append({**base, "rama": "murmullo", "umbral": f"recalibrado ({crit})", "thr": t,
                             **murmur_metrics(hs_t.murmur, hs_t.score > t)})

            rows.append({**base, "rama": "pulmonar", "umbral": "congelado",
                         **pulmonary_metrics(ls_t, ls_t.pred_crackle, ls_t.pred_wheeze)})
            (tc, tw), _ = calibrate_thresholds(ls_c[["has_crackles", "has_wheezes"]].to_numpy(),
                                               ls_c[["s_crackle", "s_wheeze"]].to_numpy())
            rows.append({**base, "rama": "pulmonar", "umbral": "recalibrado (icbhi)", "thr_crackle": tc, "thr_wheeze": tw,
                         **pulmonary_metrics(ls_t, ls_t.s_crackle > tc, ls_t.s_wheeze > tw)})

            # ── Sistema completo: gate -> rama, todo congelado vs. todo recalibrado
            t_m = best_threshold(hs_c.murmur, hs_c.score, "balanced")
            test = d[(d.maniqui == test_sex) & ((d.set == "HS") | d.lung_icbhi)]
            true = np.where(test.sound_type == "cardiac", test.murmur,
                            2 + to_4class(test.has_crackles.fillna(0).astype(int), test.has_wheezes.fillna(0).astype(int)))
            for name, tg, mpred, cpred, wpred in [
                ("congelado", 0.5, test.pred, test.pred_crackle, test.pred_wheeze),
                ("recalibrado", t_gate, (test.score > t_m).astype(int), (test.s_crackle > tc).astype(int),
                 (test.s_wheeze > tw).astype(int)),
            ]:
                system = np.where(test.gate_mean > tg, mpred, 2 + to_4class(cpred.to_numpy(), wpred.to_numpy()))
                rows.append({**base, "rama": "sistema", "umbral": name, "accuracy_6clases": (system == true).mean()})

    per = pd.DataFrame(rows)
    per.to_csv(OUT_DIR / "recalibration_per_fold.csv", index=False)
    gate = pd.DataFrame(gate_rows)
    gate.to_csv(OUT_DIR / "recalibration_gate.csv", index=False)

    metrics = {"murmullo": ["recall", "specificity", "precision", "f2", "balanced_acc"],
               "pulmonar": ["se", "sp", "icbhi_score", "crackle_recall", "wheeze_recall"],
               "sistema": ["accuracy_6clases"]}
    summaries = []
    pd.set_option("display.width", 200)
    print("== Gate ==")
    print(gate.round(3).to_string(index=False))
    for rama, cols in metrics.items():
        p = per[per.rama == rama]
        g = p.groupby("umbral", sort=False)[cols]
        s = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
        s.insert(0, "rama", rama)
        summaries.append(s.reset_index())
        print(f"\n== {rama} (media ± desvío: 2 direcciones x 5 modelos) ==")
        print(s.to_string())
    pd.concat(summaries, ignore_index=True).to_csv(OUT_DIR / "recalibration_summary.csv", index=False)
    print(f"\n✓ Reportes en: {OUT_DIR}")


if __name__ == "__main__":
    main()
