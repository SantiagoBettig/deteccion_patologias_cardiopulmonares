"""
unified_model/recalibrate_external.py
========================================
¿Cuánto se recupera afuera recalibrando solo los UMBRALES con unas pocas
grabaciones etiquetadas del equipo nuevo, sin reentrenar? (avances/28 §8,
como avances/21 con el pipeline viejo). Lee las predicciones condicionales
de external_validation.py --conditional (salidas de tarea condicionadas al
dominio: el ruteo perfecto) y no corre ningún modelo.

Decidido con el usuario:
- Regla de agregación "mean" (media de las probabilidades por ventana) con
  un umbral por tarea. La regla congelada, "frac", depende además del umbral
  por ventana y afuera vale 0 en casi todas las grabaciones: recalibrar solo
  el umbral por grabación no la arregla.
- HLS: como avances/21, se calibra con un maniquí (F o M) y se evalúa con el
  otro, en las dos direcciones. Se recalibra:
    * el umbral de dominio (P(cardíaco) media), con la máxima accuracy
      balanceada sobre HS + LS;
    * el de murmullo, con máximo F2 (el criterio interno) y con máxima
      accuracy balanceada (con 48% de murmullos, F2 premia predecir todo
      positivo);
    * crackle y wheeze, juntos, con máximo score ICBHI sobre las grabaciones
      de LS con equivalente en ICBHI.
- ICBHI: solo tiene grabaciones pulmonares, así que el umbral de dominio no
  se puede recalibrar (queda el congelado). Los umbrales crackle/wheeze se
  recalibran POR EQUIPO: los pacientes se reparten al azar en 5 partes (un
  paciente queda en la misma parte en todos los equipos); se calibra con una
  parte (~20% de los pacientes de cada equipo, "pocas grabaciones del equipo
  nuevo") y se evalúa con las otras cuatro, rotando las 5.

Grillas de 0.01 a 0.99 con paso 0.01 (las medias de probabilidad afuera
pueden quedar debajo de 0.05, el mínimo de la grilla interna).

Referencia: los umbrales congelados (regla "frac", avances/28 §1) sobre las
mismas grabaciones de evaluación.

Resultados por modelo de fold (media ± desvío sobre direcciones/partes ×
5 folds) y del ensamble (sobre direcciones/partes).

Salida: reports/external_validation/recalibration/
    hls_per_split.csv, hls_summary.csv
    icbhi_per_split.csv, icbhi_summary.csv, icbhi_by_device.csv
    recalibration_run.txt (si se redirige la salida)

Uso:
    python unified_model/recalibrate_external.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import fbeta_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import EXTERNAL_REPORTS_DIR
from integration.evaluate_integration import icbhi_from_4class
from pulmonary_adventitious_model.metrics import to_4class

IN_DIR = EXTERNAL_REPORTS_DIR / "condicional"
OUT_DIR = EXTERNAL_REPORTS_DIR / "recalibration"
GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
ENS = 5
N_PARTS, SEED = 5, 42


# ── Búsqueda de umbrales ─────────────────────────────────────────────────────
def balanced_accuracy(y, pred) -> float:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return 0.5 * (pred[y].mean() + (~pred[~y]).mean())


def best_binary(y, score, criterion: str) -> float:
    y, score = np.asarray(y).astype(int), np.asarray(score)
    def value(t):
        p = score > t
        return fbeta_score(y, p, beta=2.0, zero_division=0) if criterion == "f2" else balanced_accuracy(y, p)
    return float(max(GRID, key=value))


def best_pulmonary(yc, yw, sc, sw) -> tuple[float, float]:
    """(umbral crackle, umbral wheeze) con máximo score ICBHI, vectorizado."""
    true4 = to_4class(np.asarray(yc).astype(int), np.asarray(yw).astype(int))
    pc = (np.asarray(sc)[None] > GRID[:, None]).astype(int)          # (G, n)
    pw = (np.asarray(sw)[None] > GRID[:, None]).astype(int)
    pred = pc[:, None, :] + 2 * pw[None, :, :]                        # (G, G, n)
    normal = true4 == 0
    sp = (pred[..., normal] == 0).mean(-1) if normal.any() else np.full(pred.shape[:2], np.nan)
    se = (pred[..., ~normal] == true4[~normal]).mean(-1) if (~normal).any() else np.full(pred.shape[:2], np.nan)
    score = np.nanmean(np.stack([se, sp]), axis=0)
    i, j = np.unravel_index(np.nanargmax(score), score.shape)
    return float(GRID[i]), float(GRID[j])


# ── Métricas ──────────────────────────────────────────────────────────────────
def pulm_metrics(d: pd.DataFrame, pred4: np.ndarray, routed: np.ndarray | None = None) -> dict:
    true4 = to_4class(d["has_crackles"].astype(int).to_numpy(), d["has_wheezes"].astype(int).to_numpy())
    m = icbhi_from_4class(true4, pred4)
    yc, yw = d["has_crackles"].astype(bool).to_numpy(), d["has_wheezes"].astype(bool).to_numpy()
    out = {"pulm_se": m["se"], "pulm_sp": m["sp"], "pulm_score": m["icbhi_score"],
           "crackle_recall": np.isin(pred4[yc], [1, 3]).mean() if yc.any() else np.nan,
           "wheeze_recall": np.isin(pred4[yw], [2, 3]).mean() if yw.any() else np.nan}
    if routed is not None:
        out["pulm_score_sistema"] = icbhi_from_4class(true4, np.where(routed, pred4, -1))["icbhi_score"]
    return out


def murmur_metrics(y, pred, tag: str) -> dict:
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    return {f"murm_{tag}_recall": pred[y].mean(), f"murm_{tag}_spec": (~pred[~y]).mean(),
            f"murm_{tag}_f2": fbeta_score(y, pred, beta=2.0, zero_division=0),
            f"murm_{tag}_bal_acc": balanced_accuracy(y, pred)}


def summarize(per_split: pd.DataFrame, by: list) -> pd.DataFrame:
    cols = [c for c in per_split.columns if c not in by + ["fold", "split"]]
    rows = []
    for key, g in per_split.groupby(by, sort=False):
        folds, ens = g[g.fold != "ens"], g[g.fold == "ens"]
        r = dict(zip(by, key if isinstance(key, tuple) else (key,)))
        for c in cols:
            r[c] = f"{folds[c].mean():.3f} ± {folds[c].std():.3f}"
            r[f"{c} (ens)"] = round(ens[c].mean(), 3)
        rows.append(r)
    return pd.DataFrame(rows)


# ── HLS: maniquí contra maniquí ──────────────────────────────────────────────
def hls() -> pd.DataFrame:
    p = pd.read_csv(IN_DIR / "predictions_HLS.csv")
    p = p[p.set.isin(["HS", "LS"])].copy()
    p["maniqui"] = p.file_id.str.split("/").str[1].str[0]
    p["is_cardiac"] = (p.set == "HS").astype(int)
    rows = []
    for (system, fold), g in p.groupby(["system", "fold"]):
        for cal_m, test_m in (("F", "M"), ("M", "F")):
            cal, test = g[g.maniqui == cal_m], g[g.maniqui == test_m]
            hs_c, ls_c = cal[cal.set == "HS"], cal[(cal.set == "LS") & cal.lung_icbhi]
            hs_t, ls_t = test[test.set == "HS"], test[(test.set == "LS") & test.lung_icbhi]
            t_dom = best_binary(cal.is_cardiac, cal.p_cardiac, "bal_acc")
            t_mf2 = best_binary(hs_c.murmur, hs_c.m_murmur, "f2")
            t_mba = best_binary(hs_c.murmur, hs_c.m_murmur, "bal_acc")
            tc, tw = best_pulmonary(ls_c.has_crackles, ls_c.has_wheezes, ls_c.m_crackle, ls_c.m_wheeze)
            base = {"system": system, "fold": "ens" if fold == ENS else fold, "split": f"{cal_m}→{test_m}"}
            for setting in ("congelado", "recalibrado"):
                if setting == "congelado":
                    dom = test.pred_domain.eq("cardiac").to_numpy()
                    murm_f2 = murm_ba = hs_t.pred_murmur.to_numpy()
                    pred4 = ls_t.pred_4class.to_numpy()
                    murm_sys = test.pred_murmur.to_numpy()
                    pred4_all = test.pred_4class.to_numpy()
                else:
                    dom = (test.p_cardiac > t_dom).to_numpy()
                    murm_f2, murm_ba = (hs_t.m_murmur > t_mf2).to_numpy(), (hs_t.m_murmur > t_mba).to_numpy()
                    pred4 = to_4class((ls_t.m_crackle > tc).to_numpy(), (ls_t.m_wheeze > tw).to_numpy())
                    murm_sys = (test.m_murmur > t_mba).to_numpy()
                    pred4_all = to_4class((test.m_crackle > tc).to_numpy(), (test.m_wheeze > tw).to_numpy())
                is_c = test.is_cardiac.to_numpy().astype(bool)
                # sistema sobre HS + LS con equivalente: dominio y después la tarea
                keep = is_c | (test.lung_icbhi.to_numpy() & ~is_c)
                true = np.where(is_c, test.murmur, 2 + to_4class(test.has_crackles.fillna(0).astype(int).to_numpy(),
                                                                  test.has_wheezes.fillna(0).astype(int).to_numpy()))
                sys_cls = np.where(dom, murm_sys.astype(int), 2 + pred4_all)
                rows.append({**base, "setting": setting,
                             "dominio_HS": dom[is_c].mean(), "dominio_LS": (~dom[~is_c]).mean(),
                             "dominio_bal_acc": balanced_accuracy(is_c, dom),
                             **murmur_metrics(hs_t.murmur, murm_f2, "f2crit"),
                             **murmur_metrics(hs_t.murmur, murm_ba, "bacrit"),
                             **pulm_metrics(ls_t, pred4),
                             "accuracy_6clases_sistema": float((sys_cls[keep] == true[keep]).mean())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "hls_per_split.csv", index=False)
    s = summarize(out, ["system", "setting"])
    s.to_csv(OUT_DIR / "hls_summary.csv", index=False)
    return s


# ── ICBHI: por equipo, con una parte de los pacientes ────────────────────────
def icbhi() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(IN_DIR / "predictions_ICBHI.csv")
    subjects = np.array(sorted(p.subject_id.unique()))
    rng = np.random.default_rng(SEED)
    part = dict(zip(subjects, rng.permutation(len(subjects)) % N_PARTS))
    p["part"] = p.subject_id.map(part)
    rows, dev_rows = [], []
    for (system, fold), g in p.groupby(["system", "fold"]):
        for k in range(N_PARTS):
            evals = []
            for dev, gd in g.groupby("equipment"):
                cal, test = gd[gd.part == k], gd[gd.part != k]
                if cal.empty or test.empty:
                    continue
                tc, tw = best_pulmonary(cal.has_crackles, cal.has_wheezes, cal.m_crackle, cal.m_wheeze)
                pred4 = to_4class((test.m_crackle > tc).to_numpy(), (test.m_wheeze > tw).to_numpy())
                evals.append(test.assign(pred_recal=pred4))
                base = {"system": system, "fold": "ens" if fold == ENS else fold, "split": k, "equipment": dev}
                dev_rows.append({**base, "setting": "congelado", **pulm_metrics(test, test.pred_4class.to_numpy())})
                dev_rows.append({**base, "setting": "recalibrado", **pulm_metrics(test, pred4)})
            t = pd.concat(evals)
            routed = t.pred_domain.eq("pulmonary").to_numpy()
            base = {"system": system, "fold": "ens" if fold == ENS else fold, "split": k}
            rows.append({**base, "setting": "congelado", "n": len(t), **pulm_metrics(t, t.pred_4class.to_numpy(), routed)})
            rows.append({**base, "setting": "recalibrado", "n": len(t), **pulm_metrics(t, t.pred_recal.to_numpy(), routed)})
    out, dev = pd.DataFrame(rows), pd.DataFrame(dev_rows)
    out.to_csv(OUT_DIR / "icbhi_per_split.csv", index=False)
    dev.to_csv(OUT_DIR / "icbhi_by_device_per_split.csv", index=False)
    s = summarize(out, ["system", "setting"])
    s.to_csv(OUT_DIR / "icbhi_summary.csv", index=False)
    sd = summarize(dev, ["system", "equipment", "setting"])
    sd.to_csv(OUT_DIR / "icbhi_by_device.csv", index=False)
    return s, sd


def main() -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    s = hls()
    print("== HLS: maniquí contra maniquí (media ± desvío de 2 direcciones × 5 folds | ensamble) ==")
    print(s.to_string(index=False))
    s, sd = icbhi()
    print("\n== ICBHI: recalibración por equipo con 1/5 de los pacientes (5 partes × 5 folds | ensamble) ==")
    print(s.to_string(index=False))
    print(f"\n== ICBHI por equipo (ensamble) ==\n"
          f"{sd[['system', 'equipment', 'setting'] + [c for c in sd.columns if c.endswith('(ens)')]].to_string(index=False)}")
    print(f"\n✓ Reportes en: {OUT_DIR}")


if __name__ == "__main__":
    main()
