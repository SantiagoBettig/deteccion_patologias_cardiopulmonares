"""
unified_model/external_validation.py
=======================================
Validación externa del modelo sin gate (avances/27 §8.1, avances/28) con ICBHI
completo y HLS-CMDS. Lee las probabilidades por ventana de
external_inference.py; acá no se corre ningún modelo.

Sistemas (criterio B, 5 modelos de fold cada uno):
- cnn6_ours64 y cs_wide: modelos de 6 clases sin gate. Dominio por grabación
  = P(cardíaco) media de las ventanas > 0.5 (evaluate_unified).
- pipeline: el pipeline separado con SPRSound de avances/25: gate k-fold
  (kfold_cs, misma regla) → rama cardíaca abl_card + rama pulmonar spr_inv.

Configuración congelada, la misma de evaluate_circor_spr.py, calculada solo
con la validación interna de cada fold (CirCor y SPRSound):
- umbral por ventana de cada tarea (para la regla "frac"),
- regla de agregación y umbrales por grabación con la estrategia
  pooled_fixed (calibrate_folds). Con las bases externas no se elige nada.

Dos formas de combinar los 5 folds:
- por fold (principal): cada modelo con su configuración; se reporta media
  ± desvío entre los 5 (como avances/20);
- ensamble: las probabilidades por ventana promediadas entre los 5 modelos,
  con la regla elegida (la misma en los 5 folds con pooled_fixed) y la media
  de los umbrales de los 5 folds. Es lo que se desplegaría.

Métricas:
- ICBHI (920 grabaciones, todas pulmonares): accuracy de dominio, AUC normal
  contra anormal (media de 1 − P(pulm_normal)), capa 2 con ruteo perfecto
  (score ICBHI, AUC y recall de crackle y wheeze) y sistema (accuracy de
  6 clases y score ICBHI contando los errores de dominio como fallos). Todo
  también por equipo y por grupo de edad, y por edad dentro de los dos
  equipos que tienen niños (AKG C417L y Littmann Classic II SE).
- HLS (como avances/20 §3): HS (murmullo, 50), LS (pulmonar, las 33 con
  equivalente en ICBHI; rhonchi y pleural rub aparte), sistema HS + LS, y
  Mix (componentes solos contra la mezcla).
- Las AUC (murmullo, crackle, wheeze, normal contra anormal) usan la media
  de las probabilidades por ventana, que no depende de umbrales: la regla
  elegida suele ser "frac", y fuera de dominio puede valer 0 en todas las
  grabaciones (AUC 0.5 por empate). Las predicciones usan la regla elegida.
- KAUH (Littmann 3200, pulmonar, 112 sujetos; avances/28 §11): las mismas
  métricas que ICBHI, por modo de filtro del estetoscopio (campana,
  diafragma, extendido).
- PASCAL B (DigiScope, cardíaco, 461 grabaciones): murmullo contra el resto
  (extrasístole = sin murmullo), en todas, limpias y ruidosas.
- Diferencias apareadas del ensamble entre sistemas, con IC 95% bootstrap
  por paciente (ICBHI) o por grabación (HLS).

Con --conditional, las salidas de tarea de los modelos de 6 clases se
condicionan al dominio en cada ventana: P(murmullo | cardíaco) =
p_murmur / P(cardíaco) y P(crackle | pulmonar) = p_crackle / P(pulmonar)
(igual wheeze y pulm_normal), también en la validación interna con la que se
congela la configuración. Es el "ruteo perfecto" de verdad: sin esto, si el
modelo pone toda la masa en el dominio equivocado, la cabeza del otro queda
aplastada. Las ramas del pipeline ya son condicionales (no cambian). Salida
en reports/external_validation/condicional/.

Salida: reports/external_validation/
    frozen_config.csv            regla y umbrales por sistema, tarea y fold
    predictions_<base>.csv       una fila por (sistema, grabación, fold o "ens")
    metrics_<base>.csv           métricas por sistema, grupo y fold, con "ens"
    summary_<base>.csv           media ± desvío entre folds y ensamble
    types_hls.csv                qué predice cada sistema por tipo de sonido de HLS
    paired_<base>.csv            diferencias apareadas del ensamble

Uso:
    python unified_model/external_validation.py [--systems cnn6_ours64 cs_wide pipeline] [--conditional]
"""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (EXTERNAL_REPORTS_DIR, external_paths, HLS_RECORDINGS_CSV, SPRSOUND_RECORDINGS_CSV, SPRSOUND_REPORTS_DIR,
                          UNIFIED_MODEL_REPORTS_DIR)
from integration.evaluate_integration import (MURMUR, PULMONARY, aggregate, binary_metrics, calibrate_folds,
                                              icbhi_from_4class, load_recordings)
from pulmonary_adventitious_model.metrics import predict_4class, to_4class
from unified_model.evaluate_full_recordings import infer, test_sets
from unified_model.evaluate_unified import DOMAIN_THRESHOLD, load_window_probs, window_thresholds
from unified_model.external_inference import OUT_DIR as PROBS_DIR

K = 5
ENS = K                     # el ensamble se maneja como un "fold" más
STRATEGY = "pooled_fixed"
N_BOOT, SEED = 2000, 42
RUN_DIRS = {"cnn6_ours64": UNIFIED_MODEL_REPORTS_DIR / "runs" / "cnn6_ours64",
            "cs_wide": UNIFIED_MODEL_REPORTS_DIR / "runs" / "cs_wide",
            "abl_card": UNIFIED_MODEL_REPORTS_DIR / "runs" / "abl_card",
            "spr_inv": SPRSOUND_REPORTS_DIR / "runs" / "spr_inv"}
SYSTEMS = {   # sistema -> (dominio, murmullo, pulmonar): modelo que da cada salida
    "cnn6_ours64": ("cnn6_ours64", "cnn6_ours64", "cnn6_ours64"),
    "cs_wide": ("cs_wide", "cs_wide", "cs_wide"),
    "pipeline": ("gate_kfold_cs", "abl_card", "spr_inv"),
}
CHILD_DEVICES = ["AKG C417L Microphone", "3M Littmann Classic II SE"]
OUT_DIR = EXTERNAL_REPORTS_DIR
CONDITIONAL = False          # --conditional
SIX_CLASS = {"cnn6_ours64", "cs_wide"}
EPS = 1e-6


def condition(w: pd.DataFrame, run: str) -> pd.DataFrame:
    """Salidas de tarea condicionadas al dominio (solo con --conditional y
    en los modelos de 6 clases). Acepta p_murmur o prob (nombre de MurmurTask)."""
    if not (CONDITIONAL and run in SIX_CLASS):
        return w
    w = w.copy()
    card, pulm = w["p_cardiac"].clip(lower=EPS), (1 - w["p_cardiac"]).clip(lower=EPS)
    for c in ("p_murmur", "prob"):
        if c in w:
            w[c] = (w[c] / card).clip(upper=1.0)
    for c in ("p_crackle", "p_wheeze", "p_pulm_normal"):
        if c in w:
            w[c] = (w[c] / pulm).clip(upper=1.0)
    return w


# ── Configuración congelada (validación interna) ──────────────────────────────
def frozen_config(run: str, sets: dict, recs: pd.DataFrame, spr: pd.DataFrame) -> dict:
    """{"wthr": umbrales por ventana, "murmur"/"pulmonary": (calib, chosen)},
    con el ensamble agregado como fold ENS (media de los umbrales)."""
    run_dir = RUN_DIRS[run]
    w = load_window_probs(run_dir)
    if CONDITIONAL and run in SIX_CLASS:   # p_cardiac de la ventana desde las 6 probabilidades del índice
        w = condition(w.assign(p_cardiac=w["p_card_normal"] + w["prob"]), run)
    wthr, _ = window_thresholds(w)
    cfg = {"wthr": wthr}
    domains = set(w["domain"])
    if "cardiac" in domains:
        card = w[(w.domain == "cardiac") & w.role.isin(["val", "test"])]
        rec = aggregate(card, ["prob"], wthr).merge(recs[["file_id", "murmur"]], on="file_id")
        cfg["murmur"] = calibrate_folds(rec, MURMUR, STRATEGY)
    if "pulmonary" in domains:
        pulm = condition(infer(run_dir, run, "SPRSound", sets), run)   # cache de evaluate_circor_spr
        rec = aggregate(pulm, ["p_crackle", "p_wheeze"], wthr).merge(
            spr[["file_id", "has_crackles", "has_wheezes"]], on="file_id")
        cfg["pulmonary"] = calibrate_folds(rec, PULMONARY, STRATEGY)
    for c in wthr:
        wthr[c][ENS] = float(np.nanmean([wthr[c][f] for f in range(K)]))
    for task in ("murmur", "pulmonary"):
        if task in cfg:
            calib, chosen = cfg[task]
            rules = set(chosen.values())
            assert len(rules) == 1, f"{run}/{task}: la regla cambia entre folds ({rules})"
            rule = rules.pop()
            thr = tuple(np.mean([calib[f, rule][0] for f in range(K)], axis=0))
            calib[ENS, rule], chosen[ENS] = (thr, np.nan), rule
    return cfg


def config_table(configs: dict) -> pd.DataFrame:
    rows = []
    for run, cfg in configs.items():
        for task in ("murmur", "pulmonary"):
            if task not in cfg:
                continue
            calib, chosen = cfg[task]
            for f in range(K + 1):
                thr = calib[f, chosen[f]][0]
                rows.append({"modelo": run, "tarea": task, "fold": f if f < K else "ens", "regla": chosen[f],
                             **({"umbral_ventana": cfg["wthr"]["prob"][f], "umbral_grabacion": thr[0]}
                                if task == "murmur" else
                                {"umbral_ventana_crackle": cfg["wthr"]["p_crackle"][f],
                                 "umbral_ventana_wheeze": cfg["wthr"]["p_wheeze"][f],
                                 "umbral_crackle": thr[0], "umbral_wheeze": thr[1]})})
    return pd.DataFrame(rows)


# ── Predicciones por grabación ────────────────────────────────────────────────
def load_probs(model: str, data: str) -> pd.DataFrame:
    """Ventanas de los 5 folds más el ensamble (media entre folds) como fold ENS."""
    w = condition(pd.read_csv(PROBS_DIR / f"{model}_{data}.csv"), model)
    cols = [c for c in w.columns if c.startswith("p")]
    ens = w.groupby(["file_id", "window_idx"], as_index=False)[cols].mean().assign(fold=ENS)
    return pd.concat([w, ens], ignore_index=True)


def predict_system(system: str, data: str, configs: dict) -> pd.DataFrame:
    """Una fila por (grabación, fold): dominio, murmullo, pulmonar, puntaje
    de anormalidad y clase del sistema (0-5)."""
    dom_model, card_model, pulm_model = SYSTEMS[system]
    keys = ["file_id", "fold"]
    d = load_probs(dom_model, data)
    col = "prob_cardiac" if "prob_cardiac" in d else "p_cardiac"
    out = d.groupby(keys)[col].mean().rename("p_cardiac").reset_index()
    out["pred_domain"] = np.where(out["p_cardiac"] > DOMAIN_THRESHOLD, "cardiac", "pulmonary")

    cfg = configs[card_model]
    w = load_probs(card_model, data).rename(columns={"p_murmur": "prob"})
    agg = aggregate(w, ["prob"], cfg["wthr"], keys)
    calib, chosen = cfg["murmur"]
    rule = agg["fold"].map(chosen)
    agg["s_murmur"] = [r[f"{ru}__prob"] for (_, r), ru in zip(agg.iterrows(), rule)]
    agg["pred_murmur"] = (agg["s_murmur"] > agg["fold"].map(lambda f: calib[f, chosen[f]][0][0])).astype(int)
    agg = agg.rename(columns={"mean__prob": "m_murmur"})
    out = out.merge(agg[keys + ["s_murmur", "m_murmur", "pred_murmur"]], on=keys)

    cfg = configs[pulm_model]
    w = load_probs(pulm_model, data)
    agg = aggregate(w, ["p_crackle", "p_wheeze"], cfg["wthr"], keys)
    calib, chosen = cfg["pulmonary"]
    parts = []
    for f, a in agg.groupby("fold"):
        rule, thr = chosen[f], calib[f, chosen[f]][0]
        s = a[[f"{rule}__p_crackle", f"{rule}__p_wheeze"]].to_numpy()
        parts.append(pd.DataFrame({"file_id": a["file_id"], "fold": f, "s_crackle": s[:, 0], "s_wheeze": s[:, 1],
                                   "m_crackle": a["mean__p_crackle"], "m_wheeze": a["mean__p_wheeze"],
                                   "pred_crackle": (s[:, 0] > thr[0]).astype(int),
                                   "pred_wheeze": (s[:, 1] > thr[1]).astype(int),
                                   "pred_4class": predict_4class(s, thr)}))
    out = out.merge(pd.concat(parts), on=keys)
    abn = (1 - w.groupby(keys)["p_pulm_normal"].mean()).rename("s_abnormal").reset_index()
    out = out.merge(abn, on=keys)
    out["system_class"] = np.where(out["pred_domain"] == "cardiac", out["pred_murmur"], 2 + out["pred_4class"])
    return out.assign(system=system)


# ── Métricas ──────────────────────────────────────────────────────────────────
def auc(y, s) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def pulmonary_block(d: pd.DataFrame) -> dict:
    """d: grabaciones pulmonares con etiquetas has_crackles/has_wheezes."""
    yc, yw = d["has_crackles"].astype(int), d["has_wheezes"].astype(int)
    true4 = to_4class(yc.to_numpy(), yw.to_numpy())
    routed = d["pred_domain"].eq("pulmonary").to_numpy()
    l2 = icbhi_from_4class(true4, d["pred_4class"].to_numpy())
    sys_ = icbhi_from_4class(true4, np.where(routed, d["pred_4class"], -1))
    return {
        "n": len(d), "accuracy_dominio": routed.mean(),
        "auc_normal_anormal": auc(((yc + yw) > 0).astype(int), d["s_abnormal"]),
        "se": l2["se"], "sp": l2["sp"], "score_icbhi": l2["icbhi_score"],
        "accuracy_4clases": float((d["pred_4class"].to_numpy() == true4).mean()),
        "crackle_auc": auc(yc, d["m_crackle"]), "wheeze_auc": auc(yw, d["m_wheeze"]),
        "crackle_recall": d.loc[yc == 1, "pred_crackle"].mean(), "wheeze_recall": d.loc[yw == 1, "pred_wheeze"].mean(),
        "score_icbhi_sistema": sys_["icbhi_score"],
        "accuracy_6clases_sistema": float((d["system_class"].to_numpy() == 2 + true4).mean()),
    }


def murmur_block(d: pd.DataFrame) -> dict:
    """d: grabaciones cardíacas con etiqueta murmur."""
    y = d["murmur"].astype(int)
    m = binary_metrics(y, d["pred_murmur"], d["m_murmur"])
    routed = d["pred_domain"].eq("cardiac")
    return {"n": len(d), "accuracy_dominio": routed.mean(),
            "murmullo_recall": m["recall"], "murmullo_precision": m["precision"], "murmullo_f2": m["f2"],
            "murmullo_auc": m["auc"], "murmullo_positivos": d["pred_murmur"].mean(),
            "murmullo_f2_sistema": binary_metrics(y, (routed & (d["pred_murmur"] == 1)).astype(int))["f2"],
            "accuracy_6clases_sistema": float((d["system_class"] == y).mean())}


def per_fold(pred: pd.DataFrame, groups: dict, block) -> pd.DataFrame:
    rows = []
    for (system, fold), p in pred.groupby(["system", "fold"]):
        for gname, mask in groups.items():
            d = p[p["file_id"].isin(mask)]
            if len(d):
                rows.append({"system": system, "group": gname, "fold": "ens" if fold == ENS else fold, **block(d)})
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    """Media ± desvío entre los 5 folds y la cifra del ensamble."""
    cols = [c for c in metrics.columns if c not in ("system", "group", "fold")]
    folds = metrics[metrics.fold != "ens"].copy()
    folds[cols] = folds[cols].astype(float)
    g = folds.groupby(["system", "group"], sort=False)[cols]
    out = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    ens = metrics[metrics.fold == "ens"].set_index(["system", "group"])[cols].astype(float).round(3)
    out = out.join(ens.add_suffix(" (ens)"))
    return out[[c for pair in zip(cols, [f"{c} (ens)" for c in cols]) for c in pair]].reset_index()


def paired_bootstrap(pred: pd.DataFrame, labels: pd.DataFrame, a: str, b: str, metric, unit: str) -> dict:
    """Diferencia a − b de una métrica del ensamble, con IC 95% bootstrap
    remuestreando unidades (pacientes o grabaciones)."""
    pa = pred[(pred.system == a) & (pred.fold == ENS)].merge(labels, on="file_id")
    pb = pred[(pred.system == b) & (pred.fold == ENS)].merge(labels, on="file_id")
    pa, pb = pa.sort_values("file_id").reset_index(drop=True), pb.sort_values("file_id").reset_index(drop=True)
    assert (pa.file_id == pb.file_id).all()
    units = pa[unit].to_numpy()
    uniq = np.unique(units)
    idx_by_unit = {u: np.flatnonzero(units == u) for u in uniq}
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(N_BOOT):
        idx = np.concatenate([idx_by_unit[u] for u in rng.choice(uniq, len(uniq))])
        boot.append(metric(pa.iloc[idx]) - metric(pb.iloc[idx]))
    diff = metric(pa) - metric(pb)
    return {"a": a, "b": b, "a_valor": metric(pa), "b_valor": metric(pb), "diferencia": diff,
            "ic95": f"[{np.nanpercentile(boot, 2.5):.4f}, {np.nanpercentile(boot, 97.5):.4f}]"}


def icbhi(systems: list, configs: dict) -> None:
    recs = load_recordings().query("source_db == 'ICB'").copy()
    recs["abnormal"] = ((recs.has_crackles + recs.has_wheezes) > 0).astype(int)
    pred = pd.concat([predict_system(s, "ICBHI", configs) for s in systems], ignore_index=True)
    pred = pred.merge(recs[["file_id", "subject_id", "has_crackles", "has_wheezes", "equipment", "age_group"]],
                      on="file_id")
    pred.to_csv(OUT_DIR / "predictions_ICBHI.csv", index=False)
    groups = {"todas": set(recs.file_id)}
    for col, prefix in (("equipment", "equipo"), ("age_group", "edad")):
        for v, g in recs.groupby(col):
            groups[f"{prefix}: {v}"] = set(g.file_id)
    kids_dev = recs[recs.equipment.isin(CHILD_DEVICES)]
    for v, g in kids_dev.groupby("age_group"):
        groups[f"AKG + LittC2SE, edad: {v}"] = set(g.file_id)
    metrics = per_fold(pred, groups, pulmonary_block)
    metrics.to_csv(OUT_DIR / "metrics_ICBHI.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT_DIR / "summary_ICBHI.csv", index=False)
    show = ["system", "group", "n", "accuracy_dominio", "auc_normal_anormal", "score_icbhi", "crackle_auc",
            "wheeze_auc", "accuracy_6clases_sistema"]
    print("\n== ICBHI completo (media ± desvío de 5 folds | ensamble) ==")
    print(summary[[c for c in summary.columns if c.replace(" (ens)", "") in show]].to_string(index=False))

    labels = recs[["file_id", "subject_id", "abnormal", "has_crackles", "has_wheezes"]]
    m_auc = lambda d: auc(d["abnormal"], d["s_abnormal"])
    m_score = lambda d: pulmonary_block(d.assign(has_crackles=d.has_crackles, has_wheezes=d.has_wheezes))["score_icbhi_sistema"]
    rows = []
    for a, b in [("cnn6_ours64", "cs_wide"), ("cnn6_ours64", "pipeline"), ("cs_wide", "pipeline")]:
        if a in systems and b in systems:
            pr = pred.drop(columns=["subject_id", "has_crackles", "has_wheezes"])
            rows.append({"metrica": "auc_normal_anormal", **paired_bootstrap(pr, labels, a, b, m_auc, "subject_id")})
            rows.append({"metrica": "score_icbhi_sistema", **paired_bootstrap(pr, labels, a, b, m_score, "subject_id")})
    if rows:
        paired = pd.DataFrame(rows)
        paired.to_csv(OUT_DIR / "paired_ICBHI.csv", index=False)
        print(f"\n== ICBHI — diferencias apareadas del ensamble (bootstrap por paciente) ==\n{paired.round(4).to_string(index=False)}")


def hls(systems: list, configs: dict) -> None:
    recs = pd.read_csv(HLS_RECORDINGS_CSV)
    pred = pd.concat([predict_system(s, "HLS", configs) for s in systems], ignore_index=True)
    pred = pred.merge(recs, on="file_id")
    pred.to_csv(OUT_DIR / "predictions_HLS.csv", index=False)
    ids = lambda q: set(recs.query(q).file_id)

    m_card = per_fold(pred, {"HS": ids("set == 'HS'"), "Mix-H": ids("set == 'Mix-H'"),
                             "Mix-M (murmullo)": ids("set == 'Mix-M'")}, murmur_block)
    m_pulm = per_fold(pred, {"LS (33, equivalente ICBHI)": ids("set == 'LS' and lung_icbhi"),
                             "Mix-L (equivalente ICBHI)": ids("set == 'Mix-L' and lung_icbhi"),
                             "Mix-M (pulmonar, equivalente ICBHI)": ids("set == 'Mix-M' and lung_icbhi")},
                      pulmonary_block)
    # Sistema HS + LS: accuracy de 6 clases sobre las 50 cardíacas y las 33 pulmonares con equivalente
    sys_ids = ids("set == 'HS'") | ids("set == 'LS' and lung_icbhi")
    true = recs.set_index("file_id")
    rows = []
    for (system, fold), p in pred[pred.file_id.isin(sys_ids)].groupby(["system", "fold"]):
        t = np.where(true.loc[p.file_id, "set"] == "HS", true.loc[p.file_id, "murmur"],
                     2 + to_4class(true.loc[p.file_id, "has_crackles"].fillna(0).astype(int).to_numpy(),
                                   true.loc[p.file_id, "has_wheezes"].fillna(0).astype(int).to_numpy()))
        dom_ok = np.where(true.loc[p.file_id, "set"] == "HS", p.pred_domain == "cardiac", p.pred_domain == "pulmonary")
        rows.append({"system": system, "group": "HS + LS (83)", "fold": "ens" if fold == ENS else fold, "n": len(p),
                     "accuracy_dominio": dom_ok.mean(), "accuracy_6clases_sistema": float((p.system_class.to_numpy() == t).mean())})
    m_sys = pd.DataFrame(rows)
    # Mezclas: a qué dominio van
    rows = []
    for (system, fold), p in pred[pred.set == "Mix-M"].groupby(["system", "fold"]):
        rows.append({"system": system, "group": "Mix-M (dominio)", "fold": "ens" if fold == ENS else fold,
                     "n": len(p), "fraccion_a_cardiaco": (p.pred_domain == "cardiac").mean()})
    m_mix = pd.DataFrame(rows)

    for name, m in (("murmur", m_card), ("pulmonary", m_pulm), ("system", m_sys), ("mix_domain", m_mix)):
        m.to_csv(OUT_DIR / f"metrics_HLS_{name}.csv", index=False)
        s = summarize(m)
        s.to_csv(OUT_DIR / f"summary_HLS_{name}.csv", index=False)
        print(f"\n== HLS — {name} (media ± desvío de 5 folds | ensamble) ==\n{s.to_string(index=False)}")

    # Qué predice cada sistema por tipo de sonido (ensamble y media de folds)
    rows = []
    for (system, set_, typ), p in pred[pred.set.isin(["HS", "LS"])].groupby(["system", "set", "hls_type"]):
        for tag, q in (("ens", p[p.fold == ENS]), ("folds", p[p.fold != ENS])):
            r = {"system": system, "set": set_, "hls_type": typ, "combinacion": tag, "n": q.file_id.nunique(),
                 "a_cardiaco": (q.pred_domain == "cardiac").mean(), "murmullo": q.pred_murmur.mean()}
            for k, name in enumerate(["normal", "crackle", "wheeze", "both"]):
                r[f"pulm_{name}"] = (q.pred_4class == k).mean()
            rows.append(r)
    types = pd.DataFrame(rows)
    types.to_csv(OUT_DIR / "types_hls.csv", index=False)
    print(f"\n== HLS — por tipo de sonido (ensamble) ==\n"
          f"{types[types.combinacion == 'ens'].drop(columns='combinacion').round(2).to_string(index=False)}")

    lab = recs.assign(subject_id=recs.file_id)
    sys_lab = lab[lab.file_id.isin(sys_ids)].copy()
    sys_lab["true_class"] = np.where(sys_lab.set == "HS", sys_lab.murmur,
                                     2 + to_4class(sys_lab.has_crackles.fillna(0).astype(int).to_numpy(),
                                                   sys_lab.has_wheezes.fillna(0).astype(int).to_numpy()))
    hs_lab = lab[lab.set == "HS"][["file_id", "subject_id", "murmur"]]
    ls_lab = lab[(lab.set == "LS") & lab.lung_icbhi][["file_id", "subject_id", "has_crackles", "has_wheezes"]]
    pr = pred[["system", "fold", "file_id", "pred_domain", "system_class", "m_murmur", "m_crackle", "m_wheeze"]]
    rows = []
    for a, b in [("cnn6_ours64", "cs_wide"), ("cnn6_ours64", "pipeline"), ("cs_wide", "pipeline")]:
        if a in systems and b in systems:
            rows.append({"metrica": "accuracy_6clases_sistema (HS + LS)", **paired_bootstrap(
                pr, sys_lab[["file_id", "subject_id", "true_class"]], a, b,
                lambda d: float((d.system_class == d.true_class).mean()), "subject_id")})
            rows.append({"metrica": "murmullo_auc (HS)", **paired_bootstrap(
                pr, hs_lab, a, b, lambda d: auc(d.murmur, d.m_murmur), "subject_id")})
            rows.append({"metrica": "crackle_auc (LS)", **paired_bootstrap(
                pr, ls_lab, a, b, lambda d: auc(d.has_crackles.astype(int), d.m_crackle), "subject_id")})
    if rows:
        paired = pd.DataFrame(rows)
        paired.to_csv(OUT_DIR / "paired_HLS.csv", index=False)
        print(f"\n== HLS — diferencias apareadas del ensamble (bootstrap por grabación) ==\n{paired.round(4).to_string(index=False)}")


PAIRS = [("cnn6_ours64", "cs_wide"), ("cnn6_ours64", "pipeline"), ("cs_wide", "pipeline")]


def kauh(systems: list, configs: dict) -> None:
    """KAUH (Littmann 3200, pulmonar; avances/28 §11): cada modo de filtro por separado."""
    recs = pd.read_csv(external_paths("kauh")[0])
    recs["abnormal"] = ((recs.has_crackles + recs.has_wheezes) > 0).astype(int)
    pred = pd.concat([predict_system(s, "KAUH", configs) for s in systems], ignore_index=True)
    pred = pred.merge(recs[["file_id", "subject_id", "modo", "has_crackles", "has_wheezes", "lung_icbhi",
                            "diagnostico", "sonidos"]], on="file_id")
    pred.to_csv(OUT_DIR / "predictions_KAUH.csv", index=False)
    eq = recs[recs.lung_icbhi]
    groups = {"todas (3 modos)": set(eq.file_id)}
    for m, g in eq.groupby("modo"):
        groups[f"modo: {m}"] = set(g.file_id)
    metrics = per_fold(pred, groups, pulmonary_block)
    metrics.to_csv(OUT_DIR / "metrics_KAUH.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT_DIR / "summary_KAUH.csv", index=False)
    show = ["system", "group", "accuracy_dominio", "auc_normal_anormal", "score_icbhi", "crackle_auc", "wheeze_auc",
            "accuracy_6clases_sistema"]
    print("\n== KAUH (Littmann 3200, pulmonar; media ± desvío de 5 folds | ensamble) ==")
    print(summary[[c for c in summary.columns if c.replace(" (ens)", "") in show]].to_string(index=False))
    rows = []
    pr = pred.drop(columns=["subject_id", "has_crackles", "has_wheezes"])
    for mode, g in eq.groupby("modo"):
        lab = g[["file_id", "subject_id", "abnormal"]]
        for a, b in PAIRS:
            if a in systems and b in systems:
                rows.append({"metrica": f"auc_normal_anormal ({mode})", **paired_bootstrap(
                    pr, lab, a, b, lambda d: auc(d["abnormal"], d["s_abnormal"]), "subject_id")})
    if rows:
        paired = pd.DataFrame(rows)
        paired.to_csv(OUT_DIR / "paired_KAUH.csv", index=False)
        print(f"\n== KAUH — diferencias apareadas del ensamble (bootstrap por sujeto) ==\n{paired.round(4).to_string(index=False)}")


def pascal(systems: list, configs: dict) -> None:
    """PASCAL B (DigiScope, cardíaco; avances/28 §11): murmullo contra el resto."""
    recs = pd.read_csv(external_paths("pascal")[0])
    pred = pd.concat([predict_system(s, "PASCAL", configs) for s in systems], ignore_index=True)
    pred = pred.merge(recs[["file_id", "subject_id", "murmur", "clase", "ruidosa"]], on="file_id")
    pred.to_csv(OUT_DIR / "predictions_PASCAL.csv", index=False)
    groups = {"todas": set(recs.file_id), "limpias": set(recs[~recs.ruidosa].file_id),
              "ruidosas": set(recs[recs.ruidosa].file_id),
              "sin extrasístole": set(recs[recs.clase != "extrasistole"].file_id)}
    metrics = per_fold(pred, groups, murmur_block)
    metrics.to_csv(OUT_DIR / "metrics_PASCAL.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(OUT_DIR / "summary_PASCAL.csv", index=False)
    print(f"\n== PASCAL B (DigiScope, cardíaco; media ± desvío de 5 folds | ensamble) ==\n{summary.to_string(index=False)}")
    lab = recs[["file_id", "subject_id", "murmur"]]
    pr = pred.drop(columns=["subject_id", "murmur"])
    rows = []
    for a, b in PAIRS:
        if a in systems and b in systems:
            rows.append({"metrica": "murmullo_auc", **paired_bootstrap(
                pr, lab, a, b, lambda d: auc(d.murmur, d.m_murmur), "subject_id")})
            rows.append({"metrica": "accuracy_6clases_sistema", **paired_bootstrap(
                pr, lab, a, b, lambda d: float((d.system_class == d.murmur).mean()), "subject_id")})
    if rows:
        paired = pd.DataFrame(rows)
        paired.to_csv(OUT_DIR / "paired_PASCAL.csv", index=False)
        print(f"\n== PASCAL — diferencias apareadas del ensamble (bootstrap por paciente) ==\n{paired.round(4).to_string(index=False)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", choices=list(SYSTEMS), default=list(SYSTEMS))
    parser.add_argument("--data", nargs="+", choices=["ICBHI", "HLS", "KAUH", "PASCAL"],
                        default=["ICBHI", "HLS", "KAUH", "PASCAL"])
    parser.add_argument("--conditional", action="store_true", help="salidas de tarea condicionadas al dominio")
    args = parser.parse_args()
    global OUT_DIR, CONDITIONAL
    if args.conditional:
        CONDITIONAL, OUT_DIR = True, OUT_DIR / "condicional"
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sets, recs, spr = test_sets(), load_recordings(), pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    runs = sorted({m for s in args.systems for m in SYSTEMS[s][1:]})
    configs = {r: frozen_config(r, sets, recs, spr) for r in runs}
    cfg = config_table(configs)
    cfg.to_csv(OUT_DIR / "frozen_config.csv", index=False)
    print(f"== Configuración congelada (validación interna, {STRATEGY}) ==\n{cfg.round(3).to_string(index=False)}")
    if "ICBHI" in args.data:
        icbhi(args.systems, configs)
    if "HLS" in args.data:
        hls(args.systems, configs)
    if "KAUH" in args.data:
        kauh(args.systems, configs)
    if "PASCAL" in args.data:
        pascal(args.systems, configs)
    print(f"\n✓ Reportes en: {OUT_DIR}")


if __name__ == "__main__":
    main()
