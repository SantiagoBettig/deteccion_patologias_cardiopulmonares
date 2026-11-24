"""
unified_model/plot_external.py
=================================
Figuras de la validación externa (avances/28). Lee las tablas de
external_validation.py (absolutas y --conditional) y de
recalibrate_external.py, y los resultados internos de
evaluate_circor_spr.py --run cnn6_ours64 --base-run cs_wide --gate-run kfold_cs
(reports/unified_model/circor_spr/cnn6_ours64_vs_cs_wide/). No corre modelos.

Figuras (reports/external_validation/figures/):
    dominio.png               accuracy de dominio: interno, ICBHI, HLS-HS, HLS-LS
    p_cardiaco.png            P(cardíaco) por grabación (ensamble) en cada base externa
    auc_interno_externo.png   AUC por tarea: test interno contra ICBHI y HLS (condicional)
    roc.png                   ROC del ensamble: ICBHI normal-anormal, HLS murmullo, HLS crackle
    icbhi_equipo.png          AUC normal-anormal en ICBHI por equipo (condicional, ensamble)
    recalibracion.png         congelado contra recalibrado (ensamble)

Paleta: las tres primeras posiciones de la paleta categórica de referencia
(validadas para todos los pares en modo claro), con un marcador distinto por
sistema para no depender solo del color.

Uso:
    python unified_model/plot_external.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import EXTERNAL_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR

COND = EXTERNAL_REPORTS_DIR / "condicional"
RECAL = EXTERNAL_REPORTS_DIR / "recalibration"
INTERNAL = UNIFIED_MODEL_REPORTS_DIR / "circor_spr" / "cnn6_ours64_vs_cs_wide"
OUT = EXTERNAL_REPORTS_DIR / "figures"
ENS = 5

SYSTEMS = ["cnn6_ours64", "cs_wide", "pipeline"]
LABEL = {"cnn6_ours64": "CNN6 (4.57 M)", "cs_wide": "cs_wide (95k)", "pipeline": "Pipeline separado"}
COLOR = {"cnn6_ours64": "#2a78d6", "cs_wide": "#eb6834", "pipeline": "#1baf7a"}
MARKER = {"cnn6_ours64": "o", "cs_wide": "s", "pipeline": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 11,
    "axes.titleweight": "bold", "text.color": INK,
})


def parse(v) -> tuple[float, float]:
    """'0.601 ± 0.02' -> (0.601, 0.02); número -> (número, nan)."""
    if isinstance(v, str) and "±" in v:
        a, b = v.split("±")
        return float(a), float(b)
    return float(v), np.nan


def legend(ax, **kw):
    ax.legend(handles=[plt.Line2D([], [], color=COLOR[s], marker=MARKER[s], lw=0, markersize=7, label=LABEL[s])
                       for s in SYSTEMS], **kw)


def grouped(ax, groups: list, values: dict, errors: dict | None = None, ylim=(0, 1.05)):
    """Barras agrupadas finas con separación de 2 px entre barras."""
    x = np.arange(len(groups))
    w = 0.24
    for k, s in enumerate(SYSTEMS):
        xs = x + (k - 1) * w
        v = np.array(values[s], dtype=float)
        e = None if errors is None else np.array(errors[s], dtype=float)
        ax.bar(xs, v, width=w * 0.92, color=COLOR[s], label=LABEL[s], edgecolor="white", linewidth=1)
        if e is not None:
            ax.errorbar(xs, v, yerr=e, fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
        for xi, vi in zip(xs, v):
            if not np.isnan(vi):
                ax.text(xi, vi + 0.015 + (0 if e is None else np.nan_to_num(e[list(xs).index(xi)])),
                        f"{vi:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)


# ── 1. Dominio ────────────────────────────────────────────────────────────────
def fig_domain() -> None:
    internal = pd.read_csv(INTERNAL / "layer1_system.csv", index_col=0)
    col = {"cnn6_ours64": "cnn6_ours64 (B)", "cs_wide": "cs_wide (B)",
           "pipeline": [c for c in internal.columns if c.startswith("pipeline")][0]}
    icb = pd.read_csv(COND / "summary_ICBHI.csv").query("group == 'todas'").set_index("system")
    hm = pd.read_csv(COND / "summary_HLS_murmur.csv").query("group == 'HS'").set_index("system")
    hp = pd.read_csv(COND / "summary_HLS_pulmonary.csv").set_index("system")
    hp = hp[hp.group.str.startswith("LS")]
    pas = pd.read_csv(COND / "summary_PASCAL.csv").query("group == 'todas'").set_index("system")
    kau = pd.read_csv(COND / "summary_KAUH.csv").query("group == 'todas (3 modos)'").set_index("system")
    groups = ["Interno\n(CirCor +\nSPRSound)", "PASCAL B\n(461 card.,\nDigiScope)",
              "KAUH\n(112 × 3 pulm.,\nLittmann 3200)", "ICBHI\n(920 pulm.)", "HLS-HS\n(50 card.)",
              "HLS-LS\n(33 pulm.)"]
    vals, errs = {}, {}
    for s in SYSTEMS:
        trip = [parse(pas.loc[s, "accuracy_dominio"]), parse(kau.loc[s, "accuracy_dominio"]),
                parse(icb.loc[s, "accuracy_dominio"]), parse(hm.loc[s, "accuracy_dominio"]),
                parse(hp.loc[s, "accuracy_dominio"])]
        vals[s] = [internal.loc["accuracy_dominio", col[s]]] + [t[0] for t in trip]
        errs[s] = [0] + [t[1] for t in trip]
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped(ax, groups, vals, errs, ylim=(0, 1.12))
    ax.axvline(2.5, color=INK2, lw=0.8, ls=":")
    ax.text(1.5, 1.09, "mismo equipo que CirCor", ha="center", fontsize=7.5, color=INK2)
    ax.text(4, 1.09, "otros equipos", ha="center", fontsize=7.5, color=INK2)
    ax.set_ylabel("Accuracy de dominio por grabación")
    ax.set_title("Dominio: casi perfecto adentro, se cae afuera", loc="left")
    legend(ax, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.32))
    fig.savefig(OUT / "dominio.png")
    plt.close(fig)


# ── 2. Distribución de P(cardíaco) ───────────────────────────────────────────
def fig_p_cardiac() -> None:
    icb = pd.read_csv(COND / "predictions_ICBHI.csv").query("fold == @ENS")
    hls = pd.read_csv(COND / "predictions_HLS.csv").query("fold == @ENS")
    panels = [("ICBHI (pulmonares)", icb, "pulmonary"),
              ("HLS-HS (cardíacas)", hls[hls.set == "HS"], "cardiac"),
              ("HLS-LS (pulmonares)", hls[hls.set == "LS"], "pulmonary")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (title, d, true) in zip(axes, panels):
        for k, s in enumerate(SYSTEMS):
            v = d.loc[d.system == s, "p_cardiac"].to_numpy()
            ax.boxplot(v, positions=[k], widths=0.5, showfliers=False,
                       medianprops={"color": INK, "linewidth": 1.5},
                       boxprops={"color": INK2}, whiskerprops={"color": INK2}, capprops={"color": INK2})
            ax.scatter(k + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=COLOR[s], alpha=0.45,
                       marker=MARKER[s], linewidths=0)
        ax.axhline(0.5, color=INK2, lw=1, ls="--")
        ax.set_xticks(range(3), ["CNN6", "cs_wide", "Pipeline"])
        ax.set_title(title, loc="left")
        ax.text(2.45, 0.52 if true == "cardiac" else 0.46, "cardíaco ↑" if true == "cardiac" else "pulmonar ↓",
                ha="right", va="bottom" if true == "cardiac" else "top", fontsize=8, color=INK2)
    axes[0].set_ylabel("P(cardíaco) media por grabación")
    fig.suptitle("Probabilidad de dominio por grabación (ensamble; umbral 0.5 punteado)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "p_cardiaco.png")
    plt.close(fig)


# ── 3. AUC interno contra externo ────────────────────────────────────────────
def fig_auc() -> None:
    mi = pd.read_csv(INTERNAL / "murmur_comparison.csv").set_index("setting")
    pi = pd.read_csv(INTERNAL / "pulmonary_comparison.csv").set_index("setting")
    ri = pd.read_csv(INTERNAL / "sprsound_recording_auc.csv").set_index("setting")
    key_m = {"cnn6_ours64": "cnn6_ours64 (B)", "cs_wide": "cs_wide (B)", "pipeline": "abl_card"}
    key_p = {"cnn6_ours64": "cnn6_ours64 (B)", "cs_wide": "cs_wide (B)", "pipeline": "spr_inv"}
    icb = pd.read_csv(COND / "summary_ICBHI.csv").query("group == 'todas'").set_index("system")
    hm = pd.read_csv(COND / "summary_HLS_murmur.csv").query("group == 'HS'").set_index("system")
    hp = pd.read_csv(COND / "summary_HLS_pulmonary.csv").set_index("system")
    hp = hp[hp.group.str.startswith("LS")]
    pas = pd.read_csv(COND / "summary_PASCAL.csv").query("group == 'todas'").set_index("system")
    kau = pd.read_csv(COND / "summary_KAUH.csv").query("group == 'todas (3 modos)'").set_index("system")
    na = (np.nan, np.nan)
    tasks = {   # bases: interno, mismo equipo (PASCAL B o KAUH), ICBHI, HLS
        "Murmullo": lambda s: [parse(mi.loc[key_m[s], "auc"]), parse(pas.loc[s, "murmullo_auc"]), na,
                               parse(hm.loc[s, "murmullo_auc"])],
        "Normal-anormal (pulmonar)": lambda s: [parse(ri.loc[key_p[s], "auc_normal_anormal_por_fold"]),
                                                parse(kau.loc[s, "auc_normal_anormal"]),
                                                parse(icb.loc[s, "auc_normal_anormal"]),
                                                parse(hp.loc[s, "auc_normal_anormal"])],
        "Crackle": lambda s: [parse(pi.loc[key_p[s], "crackle_auc"]), parse(kau.loc[s, "crackle_auc"]),
                              parse(icb.loc[s, "crackle_auc"]), parse(hp.loc[s, "crackle_auc"])],
        "Wheeze": lambda s: [parse(pi.loc[key_p[s], "wheeze_auc"]), parse(kau.loc[s, "wheeze_auc"]),
                             parse(icb.loc[s, "wheeze_auc"]), parse(hp.loc[s, "wheeze_auc"])],
    }
    bases = ["Interno", "Equipo de\nCirCor", "ICBHI", "HLS"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=True)
    for ax, (task, get) in zip(axes, tasks.items()):
        for k, s in enumerate(SYSTEMS):
            v = np.array(get(s))
            x = np.arange(4) + (k - 1) * 0.18
            ax.errorbar(x, v[:, 0], yerr=v[:, 1], fmt=MARKER[s], color=COLOR[s], markersize=7,
                        elinewidth=1, capsize=2, label=LABEL[s])
        ax.axhline(0.5, color=INK2, lw=1, ls=":")
        ax.set_xticks(range(4), bases)
        ax.set_title(task, loc="left", fontsize=10)
        ax.set_xlim(-0.5, 3.5)
        if task == "Murmullo":
            ax.text(2, 0.75, "sin grabaciones\ncardíacas", ha="center", va="center", fontsize=7.5, color=INK2)
    axes[0].set_ylim(0.4, 1.02)
    axes[0].set_ylabel("AUC por grabación (media ± desvío, 5 folds)")
    fig.legend(handles=[plt.Line2D([], [], color=COLOR[s], marker=MARKER[s], lw=0, markersize=7, label=LABEL[s])
                        for s in SYSTEMS], loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Tareas con ruteo perfecto: test interno contra bases externas (salidas condicionales; "
                 "equipo de CirCor = PASCAL B en murmullo, KAUH en pulmonar)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "auc_interno_externo.png")
    plt.close(fig)


# ── 4. ROC ────────────────────────────────────────────────────────────────────
def fig_roc() -> None:
    icb = pd.read_csv(COND / "predictions_ICBHI.csv").query("fold == @ENS")
    icb["y"] = ((icb.has_crackles + icb.has_wheezes) > 0).astype(int)
    hls = pd.read_csv(COND / "predictions_HLS.csv").query("fold == @ENS")
    hs = hls[hls.set == "HS"].assign(y=lambda d: d.murmur)
    ls = hls[(hls.set == "LS") & hls.lung_icbhi].assign(y=lambda d: d.has_crackles.astype(int))
    panels = [("ICBHI: normal contra anormal", icb, "s_abnormal"),
              ("HLS-HS: murmullo", hs, "m_murmur"),
              ("HLS-LS: crackle", ls, "m_crackle")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, (title, d, col) in zip(axes, panels):
        for s in SYSTEMS:
            g = d[d.system == s]
            fpr, tpr, _ = roc_curve(g.y, g[col])
            ax.plot(fpr, tpr, color=COLOR[s], lw=2, label=f"{LABEL[s]}: {roc_auc_score(g.y, g[col]):.2f}")
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":")
        ax.set_title(title, loc="left")
        ax.set_xlabel("1 − especificidad")
        ax.set_aspect("equal")
        ax.grid(axis="both")
        ax.legend(loc="lower right", fontsize=7.5, title="AUC (ensamble)", title_fontsize=7.5)
    axes[0].set_ylabel("Sensibilidad")
    fig.tight_layout()
    fig.savefig(OUT / "roc.png")
    plt.close(fig)


# ── 5. ICBHI por equipo ───────────────────────────────────────────────────────
def fig_device() -> None:
    s = pd.read_csv(COND / "summary_ICBHI.csv")
    s = s[s.group.str.startswith("equipo")]
    devs = ["equipo: 3M Littmann Classic II SE", "equipo: WelchAllyn Meditron Master Elite",
            "equipo: AKG C417L Microphone", "equipo: 3M Littmann 3200 Electronic"]
    names = ["Littmann\nClassic II SE", "Meditron", "AKG C417L", "Littmann\n3200"]
    n = s[s.system == "cnn6_ours64"].set_index("group")["n (ens)"]
    names = [f"{nm}\n(n = {int(n[d])})" for nm, d in zip(names, devs)]
    vals = {sy: [s[(s.system == sy) & (s.group == d)]["auc_normal_anormal (ens)"].iloc[0] for d in devs]
            for sy in SYSTEMS}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    grouped(ax, names, vals, ylim=(0.3, 0.9))
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_ylabel("AUC normal contra anormal (ensamble)")
    ax.set_title("ICBHI por equipo de grabación (salidas condicionales)", loc="left")
    legend(ax, loc="upper right", ncol=3, fontsize=8)
    fig.savefig(OUT / "icbhi_equipo.png")
    plt.close(fig)


# ── 6. Recalibración ──────────────────────────────────────────────────────────
def fig_recalibration() -> None:
    h = pd.read_csv(RECAL / "hls_summary.csv").set_index(["system", "setting"])
    i = pd.read_csv(RECAL / "icbhi_summary.csv").set_index(["system", "setting"])
    panels = [("HLS: dominio\n(accuracy balanceada)", h, "dominio_bal_acc (ens)"),
              ("HLS: score ICBHI\npulmonar", h, "pulm_score (ens)"),
              ("HLS: accuracy de 6 clases\ndel sistema", h, "accuracy_6clases_sistema (ens)"),
              ("ICBHI: score ICBHI\n(recalibrado por equipo)", i, "pulm_score (ens)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, (title, t, col) in zip(axes, panels):
        for k, s in enumerate(SYSTEMS):
            a, b = t.loc[(s, "congelado"), col], t.loc[(s, "recalibrado"), col]
            ax.annotate("", xy=(k, b), xytext=(k, a),
                        arrowprops={"arrowstyle": "-|>", "color": COLOR[s], "lw": 2, "shrinkA": 4, "shrinkB": 4})
            ax.scatter([k], [a], s=40, facecolors="white", edgecolors=COLOR[s], linewidths=1.5,
                       marker=MARKER[s], zorder=3)
            ax.scatter([k], [b], s=50, color=COLOR[s], marker=MARKER[s], zorder=3)
            ax.text(k + 0.12, b, f"{b:.2f}", fontsize=7.5, color=INK2, va="center")
        ax.set_xticks(range(3), ["CNN6", "cs_wide", "Pipeline"])
        ax.set_xlim(-0.5, 2.6)
        ax.set_title(title, loc="left", fontsize=9.5)
    axes[0].set_ylim(0.25, 1.0)
    axes[0].set_ylabel("Ensamble")
    fig.legend(handles=[
        plt.Line2D([], [], marker="o", lw=0, markerfacecolor="white", markeredgecolor=INK2, label="Umbrales congelados"),
        plt.Line2D([], [], marker="o", lw=0, color=INK2, label="Recalibrados")],
        loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Recalibración de umbrales con pocas grabaciones del equipo nuevo", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "recalibracion.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_domain, fig_p_cardiac, fig_auc, fig_roc, fig_device, fig_recalibration):
        f()
        print(f"  {f.__name__}", flush=True)
    print(f"✓ {OUT}")


if __name__ == "__main__":
    main()
