"""
unified_model/internal_analysis.py
=====================================
Dos análisis del test interno (CirCor + SPRSound, pacientes de test de cada
fold; avances/29), para los tres sistemas de avances/28: cnn6_ours64,
cs_wide y el pipeline separado (gate kfold_cs → abl_card + spr_inv). No
corre modelos: usa las probabilidades por ventana ya guardadas (índice de
CirCor y grabaciones completas de SPRSound, criterio B).

1. Duración de la grabación: ¿cuántos segundos hacen falta? Cada grabación
   de test se recorta a sus primeros N segundos (solo las ventanas que
   entran enteras; la primera siempre queda) y se vuelve a agregar con la
   configuración congelada de cada fold (regla y umbrales calibrados con las
   grabaciones completas de validación, como se usaría el sistema). N en
   4, 6, 8, 10, 15, 20 s y la grabación completa.
2. Subgrupos (grabación completa): CirCor por grupo de edad, sexo y foco de
   auscultación; SPRSound por edad, sexo y foco. AUC e IC 95% bootstrap por
   paciente (1000 repeticiones).

Métricas, sobre todos los pacientes de test (cada grabación una vez):
- accuracy de dominio y de 6 clases del sistema (con los errores de
  dominio como fallos, como evaluate_unified.layer1);
- murmullo (CirCor): F2 del sistema, recall, especificidad y AUC (media de
  P(murmullo) por ventana);
- pulmonar (SPRSound): score ICBHI con ruteo perfecto y del sistema, y AUC
  normal-anormal (media de 1 − P(pulm_normal)).

Salida: reports/unified_model/internal_analysis/
    duration.csv, duration.png
    subgroups.csv, subgroups.png

Uso:
    python unified_model/internal_analysis.py [--plot]   # --plot: solo las figuras
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import fbeta_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_REPORTS_DIR, SPRSOUND_RECORDINGS_CSV, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import aggregate, icbhi_from_4class, load_recordings
from pulmonary_adventitious_model.metrics import predict_4class, to_4class
from unified_model.evaluate_full_recordings import infer, test_sets
from unified_model.evaluate_unified import load_window_probs
from unified_model.external_validation import RUN_DIRS, SYSTEMS, frozen_config

OUT = UNIFIED_MODEL_REPORTS_DIR / "internal_analysis"
DURATIONS = [4, 6, 8, 10, 15, 20, None]   # None = grabación completa
KEYS = ["file_id", "subject_id", "fold", "role"]
N_BOOT, SEED = 1000, 42
COLOR = {"cnn6_ours64": "#2a78d6", "cs_wide": "#eb6834", "pipeline": "#1baf7a"}
MARKER = {"cnn6_ours64": "o", "cs_wide": "s", "pipeline": "^"}
LABEL = {"cnn6_ours64": "CNN6 (4.57 M)", "cs_wide": "cs_wide (95k)", "pipeline": "Pipeline separado"}
INK2, GRID = "#52514e", "#e4e3df"
CIRCOR_AGE = {"neonate": "neonato", "infant": "lactante", "child": "niño", "adolescent": "adolescente"}
CIRCOR_FOCUS = {"right_upper_sternal_border": "aórtico", "left_upper_sternal_border": "pulmonar",
                "left_lower_sternal_border": "tricuspídeo", "apex": "mitral", "other": "otro"}
AGE_LABELS = ["< 1 año", "1-3 años", "3-6 años", "6-12 años", "≥ 12 años"]
SUBGROUP_ORDER = ([f"edad: {a}" for a in ["neonato", "lactante", "niño", "adolescente", "sin dato"] + AGE_LABELS]
                  + ["sexo: F", "sexo: M"]
                  + [f"foco: {f}" for f in ["aórtico", "pulmonar", "tricuspídeo", "mitral", "p1", "p2", "p3", "p4"]])

plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
                     "savefig.bbox": "tight", "axes.titleweight": "bold", "axes.titlesize": 10.5})


# ── Ventanas internas (test y val, criterio B) ───────────────────────────────
def windows(run: str, sets: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(ventanas cardíacas de CirCor, ventanas de SPRSound) del modelo, con p_murmur, p_crackle, ..."""
    run_dir = RUN_DIRS[run]
    w = load_window_probs(run_dir)
    card = w[(w.domain == "cardiac") & w.role.isin(["val", "test"])].rename(columns={"prob": "p_murmur"})
    card = card if len(card) else None
    pulm = infer(run_dir, run, "SPRSound", sets) if run != "abl_card" else None
    return card, pulm


def gate_windows() -> pd.DataFrame:
    d = GATE_REPORTS_DIR / "runs" / "kfold_cs"
    return pd.concat([pd.read_csv(p) for p in sorted(d.glob("window_probs_fold?.csv"))], ignore_index=True)


def truncate(w: pd.DataFrame, max_sec: float | None, win: float) -> pd.DataFrame:
    if max_sec is None:
        return w
    return w[(w.start_sec + win <= max_sec + 1e-6) | (w.window_idx == 0)]


# ── Predicción por grabación (solo test) ─────────────────────────────────────
def murmur_pred(w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    w = w.rename(columns={"p_murmur": "prob"})
    agg = aggregate(w[w.role == "test"], ["prob"], cfg["wthr"], KEYS)
    calib, chosen = cfg["murmur"]
    agg["pred_murmur"] = [int(r[f"{chosen[f]}__prob"] > calib[f, chosen[f]][0][0])
                          for f, (_, r) in zip(agg.fold, agg.iterrows())]
    return agg.rename(columns={"mean__prob": "m_murmur"})[KEYS + ["m_murmur", "pred_murmur"]]


def pulm_pred(w: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    t = w[w.role == "test"]
    agg = aggregate(t, ["p_crackle", "p_wheeze"], cfg["wthr"], KEYS)
    calib, chosen = cfg["pulmonary"]
    parts = []
    for f, a in agg.groupby("fold"):
        rule, thr = chosen[f], calib[f, chosen[f]][0]
        s = a[[f"{rule}__p_crackle", f"{rule}__p_wheeze"]].to_numpy()
        parts.append(a[KEYS].assign(pred_4class=predict_4class(s, thr)))
    abn = (1 - t.groupby(KEYS)["p_pulm_normal"].mean()).rename("s_abnormal").reset_index()
    return pd.concat(parts).merge(abn, on=KEYS)


def domain_pred(ws: list[pd.DataFrame], col: str) -> pd.DataFrame:
    w = pd.concat([x[x.role == "test"][KEYS + [col]] for x in ws], ignore_index=True)
    d = w.groupby(KEYS)[col].mean().rename("p_cardiac").reset_index()
    d["pred_cardiac"] = d.p_cardiac > 0.5
    return d


def system_predictions(system: str, max_sec, src: dict, configs: dict, labels: dict) -> pd.DataFrame:
    dom_m, card_m, pulm_m = SYSTEMS[system]
    if dom_m == "gate_kfold_cs":
        dom = domain_pred([truncate(src["gate"], max_sec, 5.0)], "prob_cardiac")
    else:
        card, pulm = src[dom_m]
        dom = domain_pred([truncate(card, max_sec, 4.0), truncate(pulm, max_sec, 4.0)], "p_cardiac")
    m = murmur_pred(truncate(src[card_m][0], max_sec, 4.0), configs[card_m])
    p = pulm_pred(truncate(src[pulm_m][1], max_sec, 4.0), configs[pulm_m])
    c = dom.merge(m, on=KEYS).merge(labels["circor"], on="file_id")
    c["true_class"], c["task_class"] = c.murmur, c.pred_murmur
    c["domain_ok"] = c.pred_cardiac
    q = dom.merge(p, on=KEYS).merge(labels["spr"], on="file_id")
    q["true_class"] = 2 + to_4class(q.has_crackles.to_numpy(), q.has_wheezes.to_numpy())
    q["task_class"] = 2 + q.pred_4class
    q["domain_ok"] = ~q.pred_cardiac
    out = pd.concat([c.assign(domain="cardiac"), q.assign(domain="pulmonary")], ignore_index=True)
    out["system_ok"] = out.domain_ok & (out.task_class == out.true_class)
    return out.assign(system=system)


# ── Métricas ──────────────────────────────────────────────────────────────────
def auc(y, s) -> float:
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else np.nan


def metrics(d: pd.DataFrame) -> dict:
    c, p = d[d.domain == "cardiac"], d[d.domain == "pulmonary"]
    out = {"n": len(d), "accuracy_dominio": d.domain_ok.mean(), "accuracy_6clases_sistema": d.system_ok.mean()}
    if len(c):
        y, pred = c.murmur.astype(int), (c.domain_ok & (c.pred_murmur == 1)).astype(int)
        out.update({"n_card": len(c), "murmullos": int(y.sum()), "murmullo_auc": auc(y, c.m_murmur),
                    "murmullo_f2_sistema": fbeta_score(y, pred, beta=2.0, zero_division=0),
                    "murmullo_recall": c.loc[y == 1, "pred_murmur"].mean() if y.any() else np.nan,
                    "murmullo_especificidad": 1 - c.loc[y == 0, "pred_murmur"].mean() if (y == 0).any() else np.nan})
    if len(p):
        true4 = (p.true_class - 2).to_numpy()
        l2 = icbhi_from_4class(true4, p.pred_4class.to_numpy())
        sys_ = icbhi_from_4class(true4, np.where(p.domain_ok, p.pred_4class, -1))
        out.update({"n_pulm": len(p), "anormales": int((true4 > 0).sum()),
                    "auc_normal_anormal": auc((true4 > 0).astype(int), p.s_abnormal),
                    "pulm_se": l2["se"], "pulm_sp": l2["sp"], "score_icbhi": l2["icbhi_score"],
                    "score_icbhi_sistema": sys_["icbhi_score"]})
    return out


def bootstrap_ci(d: pd.DataFrame, fn, rng) -> str:
    subj = d.subject_id.to_numpy()
    uniq = np.unique(subj)
    idx = {u: np.flatnonzero(subj == u) for u in uniq}
    vals = []
    for _ in range(N_BOOT):
        take = np.concatenate([idx[u] for u in rng.choice(uniq, len(uniq))])
        vals.append(fn(d.iloc[take]))
    return f"[{np.nanpercentile(vals, 2.5):.3f}, {np.nanpercentile(vals, 97.5):.3f}]"


# ── Figuras ───────────────────────────────────────────────────────────────────
def fig_duration(t: pd.DataFrame) -> None:
    cols = [("accuracy_dominio", "Accuracy de dominio"), ("accuracy_6clases_sistema", "Accuracy de 6 clases"),
            ("murmullo_auc", "AUC murmullo (CirCor)"), ("auc_normal_anormal", "AUC normal-anormal (SPRSound)")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    xs = [d for d in DURATIONS if d is not None] + [25]
    for ax, (c, title) in zip(axes, cols):
        for s in SYSTEMS:
            v = t[t.system == s].set_index("duracion")[c]
            ax.plot(xs, [v[k] for k in [str(d) if d else "completa" for d in DURATIONS]], color=COLOR[s],
                    marker=MARKER[s], lw=2, markersize=6, label=LABEL[s])
        ax.set_xticks(xs, [str(d) for d in DURATIONS if d] + ["compl."])
        ax.set_xlabel("Segundos usados")
        ax.set_title(title, loc="left")
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Test interno: rendimiento según los primeros N segundos de cada grabación", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "duration.png")
    plt.close(fig)


def fig_subgroups(sg: pd.DataFrame) -> None:
    panels = [("CirCor", "murmullo_auc", "AUC murmullo"), ("SPRSound", "auc_normal_anormal", "AUC normal-anormal")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, (base, col, title) in zip(axes, panels):
        d = sg[(sg.base == base)]
        groups = sorted(dict.fromkeys(d.subgrupo), key=lambda g: SUBGROUP_ORDER.index(g) if g in SUBGROUP_ORDER else 99)
        y = np.arange(len(groups))
        for k, s in enumerate(SYSTEMS):
            e = d[d.system == s].set_index("subgrupo").reindex(groups)
            lo_hi = e[f"{col}_ic95"].str.strip("[]").str.split(", ", expand=True).astype(float)
            ax.errorbar(e[col], y + (k - 1) * 0.22, xerr=[e[col] - lo_hi[0], lo_hi[1] - e[col]], fmt=MARKER[s],
                        color=COLOR[s], markersize=6, elinewidth=1, capsize=2, label=LABEL[s])
        ax.set_yticks(y, [f"{g} (n = {int(d[d.subgrupo == g].n.iloc[0])})" for g in groups], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{base}: {title}", loc="left")
        ax.set_xlabel("AUC (IC 95% bootstrap por paciente)")
        ax.grid(axis="y", visible=False)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "subgroups.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--plot" in sys.argv:   # solo las figuras, desde los CSV
        fig_duration(pd.read_csv(OUT / "duration.csv").assign(duracion=lambda d: d.duracion.astype(str)))
        fig_subgroups(pd.read_csv(OUT / "subgroups.csv"))
        return
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    sets, recs = test_sets(), load_recordings()
    spr = pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    configs = {r: frozen_config(r, sets, recs, spr) for r in ("cnn6_ours64", "cs_wide", "abl_card", "spr_inv")}
    src = {r: windows(r, sets) for r in ("cnn6_ours64", "cs_wide", "abl_card", "spr_inv")}
    src["gate"] = gate_windows()
    cir = recs[recs.source_db == "CIR"][["file_id", "murmur", "age_category", "sex", "auscultation_location"]]
    labels = {"circor": cir, "spr": spr[["file_id", "has_crackles", "has_wheezes", "age_years", "sex",
                                         "auscultation_location"]]}

    # 1. Duración
    rows, full = [], {}
    dur = pd.concat([recs.set_index("file_id").duration_sec,
                     (spr.set_index("file_id").n_frames / 8000.0)]) if "n_frames" in spr else None
    for s in SYSTEMS:
        for d in DURATIONS:
            pr = system_predictions(s, d, src, configs, labels)
            if d is None:
                full[s] = pr
            rows.append({"system": s, "duracion": str(d) if d else "completa", **metrics(pr)})
        print(f"  duración: {s}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "duration.csv", index=False)
    show = ["system", "duracion", "accuracy_dominio", "accuracy_6clases_sistema", "murmullo_auc",
            "murmullo_f2_sistema", "auc_normal_anormal", "score_icbhi_sistema"]
    print(f"\n== Duración (test interno, todos los pacientes) ==\n{t[show].round(3).to_string(index=False)}")
    fig_duration(t)

    # 2. Subgrupos (grabación completa)
    age_bins = [-1, 1, 3, 6, 12, 19]
    age_lab = AGE_LABELS
    rng = np.random.default_rng(SEED)
    rows = []
    for s, pr in full.items():
        c, p = pr[pr.domain == "cardiac"].copy(), pr[pr.domain == "pulmonary"].copy()
        c["edad"] = c.age_category.map(CIRCOR_AGE).fillna("sin dato")
        c["auscultation_location"] = c.auscultation_location.map(CIRCOR_FOCUS)
        p["edad"] = pd.cut(p.age_years, age_bins, labels=age_lab).astype(str)
        for base, d, by in (("CirCor", c, [("edad", "edad"), ("sex", "sexo"), ("auscultation_location", "foco")]),
                            ("SPRSound", p, [("edad", "edad"), ("sex", "sexo"), ("auscultation_location", "foco")])):
            col = "murmullo_auc" if base == "CirCor" else "auc_normal_anormal"
            fn = ((lambda x: auc(x.murmur.astype(int), x.m_murmur)) if base == "CirCor"
                  else (lambda x: auc((x.true_class > 2).astype(int), x.s_abnormal)))
            for key, name in by:
                for v, g in d.groupby(key):
                    if len(g) < 20:
                        continue
                    m = metrics(g)
                    rows.append({"system": s, "base": base, "subgrupo": f"{name}: {v}", **m,
                                 f"{col}_ic95": bootstrap_ci(g, fn, rng),
                                 "accuracy_6clases_ic95": bootstrap_ci(g, lambda x: x.system_ok.mean(), rng)})
        print(f"  subgrupos: {s}", flush=True)
    sg = pd.DataFrame(rows)
    sg.to_csv(OUT / "subgroups.csv", index=False)
    show = ["system", "base", "subgrupo", "n", "murmullos", "murmullo_auc", "murmullo_auc_ic95", "murmullo_f2_sistema",
            "anormales", "auc_normal_anormal", "auc_normal_anormal_ic95", "score_icbhi", "accuracy_6clases_sistema"]
    print(f"\n== Subgrupos (test interno, grabación completa) ==\n{sg[[c for c in show if c in sg]].round(3).to_string(index=False)}")
    fig_subgroups(sg)
    print(f"\n✓ {OUT}")


if __name__ == "__main__":
    main()
