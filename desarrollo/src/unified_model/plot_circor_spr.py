"""
unified_model/plot_circor_spr.py
===================================
Figuras del modelo de 6 clases CirCor + SPRSound (avances/24, run_kfold.py
--data circor_spr), con las ablaciones abl_card (solo CirCor) y spr_inv
(solo SPRSound) de referencia. Lee los history_fold*.csv y
window_probs_fold*.csv de cada corrida y los reportes de
evaluate_circor_spr.py.

Salida: reports/unified_model/circor_spr/figures/
    learning_curves.png      loss, criterios de selección B y A y AUC de cada tarea en val, por época y fold
    roc_windows.png          ROC por ventana de murmullo, crackle y wheeze (test de todos los folds juntos)
    confusion_windows.png    matriz de confusión de 6 clases por ventana (argmax, test), por fila
    confusion_system.png     capa 1 por grabación: clase real contra salida del sistema (o dominio erróneo)
    recording_metrics.png    métricas por grabación del modelo de 6 clases contra las ablaciones (por fold)
    domain_separation.png    P(cardíaco) media por grabación, según el dominio real

Uso:
    python unified_model/plot_circor_spr.py [--run cs_base] [--card-ref abl_card] [--pulm-ref spr_inv]
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SPRSOUND_RECORDINGS_CSV, SPRSOUND_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import SYSTEM_CLASSES
from unified_model.evaluate_full_recordings import OUT_DIR as FULL_DIR
from unified_model.evaluate_sprsound import auc, recording_scores
from unified_model.evaluate_unified import K, load_window_probs

EVAL_DIR = UNIFIED_MODEL_REPORTS_DIR / "circor_spr"
CLASSES = ["card normal", "card murmullo", "pulm normal", "pulm crackle", "pulm wheeze", "pulm both"]
MODEL_COLORS = {"modelo de 6 clases": "tab:blue", "ablación": "tab:orange"}


def learning_curves(run: str, d: Path, out: Path) -> None:
    panels = [("loss (línea: val, guiones: train)", None), ("selección B (val; punto: checkpoint principal)", "select_domain"),
              ("selección A (val; punto: su checkpoint)", "select"), ("AUC murmullo (val)", "auc_murmur"),
              ("AUC crackle (val)", "auc_crackle"), ("AUC wheeze (val)", "auc_wheeze")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    colors = plt.cm.tab10(np.arange(K))
    for f in range(K):
        h = pd.read_csv(d / f"history_fold{f}.csv")
        for ax, (title, col) in zip(axes.flat, panels):
            if col is None:
                ax.plot(h["epoch"], h["train_loss"], color=colors[f], ls="--", lw=1)
                ax.plot(h["epoch"], h["val_loss"], color=colors[f], lw=1.5, label=f"fold {f}")
                continue
            ax.plot(h["epoch"], h[col], color=colors[f], lw=1.5)
            if col.startswith("select"):
                best = h.loc[h[col].idxmax()]
                ax.scatter(best["epoch"], best[col], color=colors[f], s=40, zorder=3)
    for ax, (title, _) in zip(axes.flat, panels):
        ax.set_title(f"{run} — {title}", fontsize=10)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("época")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=110)
    plt.close(fig)


def roc_windows(probs: dict[str, pd.DataFrame], out: Path) -> None:
    """probs: nombre -> ventanas de test; las ablaciones solo tienen su dominio.
    La curva es el promedio vertical de las ROC de cada fold y la leyenda da la
    media de los AUC por fold (como las tablas): juntar los folds en una sola
    ROC mezcla modelos con calibraciones distintas y cambia el AUC."""
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, task, domain, pos, col in ((axes[0], "murmullo", "cardiac", [1], "p_murmur"),
                                       (axes[1], "crackle", "pulmonary", [3, 5], "p_crackle"),
                                       (axes[2], "wheeze", "pulmonary", [4, 5], "p_wheeze")):
        for (name, w), color in zip(probs.items(), MODEL_COLORS.values()):
            for run, g in w[w.domain == domain].groupby("run"):
                tprs, aucs = [], []
                for _, f in g.groupby("fold"):
                    y = f["label"].isin(pos)
                    fpr, tpr, _ = roc_curve(y, f[col])
                    tprs.append(np.interp(grid, fpr, tpr))
                    aucs.append(roc_auc_score(y, f[col]))
                tprs = np.array(tprs)
                ax.plot(grid, tprs.mean(0), color=color, lw=2,
                        label=f"{run} ({np.mean(aucs):.3f} ± {np.std(aucs, ddof=1):.3f})")
                ax.fill_between(grid, tprs.min(0), tprs.max(0), color=color, alpha=0.15, lw=0)
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.set_title(f"ROC por ventana — {task} (test; banda: rango entre folds)", fontsize=10)
        ax.set_xlabel("1 − especificidad")
        ax.set_ylabel("sensibilidad")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_windows.png", dpi=110)
    plt.close(fig)


def heatmap(ax, cm: np.ndarray, rows: list[str], cols: list[str], title: str) -> None:
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for (a, b), v in np.ndenumerate(cm):
        ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_xlabel("predicha", fontsize=9)
    ax.set_ylabel("real", fontsize=9)
    ax.set_title(title, fontsize=10)


def confusion_windows(w: pd.DataFrame, run: str, out: Path) -> None:
    p = w[["p_card_normal", "p_card_murmur", "p_pulm_normal", "p_pulm_crackle", "p_pulm_wheeze", "p_pulm_both"]]
    cm = confusion_matrix(w["label"], p.to_numpy().argmax(1), labels=range(6), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5.2))
    heatmap(ax, cm, CLASSES, CLASSES, f"{run} — 6 clases por ventana (test, argmax), por fila")
    fig.tight_layout()
    fig.savefig(out / "confusion_windows.png", dpi=110)
    plt.close(fig)


def confusion_system(run_eval: Path, run: str, out: Path) -> None:
    g = pd.read_csv(run_eval / "layer1_all_recordings.csv")
    cm = confusion_matrix(g["true_class"], g["system_class"], labels=range(7))[:6]
    n = cm.sum(1)
    cm = cm / n[:, None]
    acc = (g["true_class"] == g["system_class"]).mean()
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    heatmap(ax, cm, [f"{c} (n={k})" for c, k in zip(SYSTEM_CLASSES, n)], SYSTEM_CLASSES + ["dominio erróneo"],
            f"{run} — sistema por grabación (test, todos los pacientes)\naccuracy de 6 clases {acc:.3f}, por fila")
    fig.tight_layout()
    fig.savefig(out / "confusion_system.png", dpi=110)
    plt.close(fig)


def per_fold_metric(setting_dir: Path, task: str, metric: str) -> np.ndarray:
    p = pd.read_csv(setting_dir / f"{task}_per_fold.csv")
    return p[(p.rule == "auto") & (p.strategy == "pooled_fixed")].sort_values("fold")[metric].to_numpy()


def rec_auc_per_fold(tag: str, spr: pd.DataFrame) -> np.ndarray:
    w = pd.read_csv(FULL_DIR / f"{tag}_SPRSound_windows.csv")
    sc = recording_scores(w[w.role == "test"], spr)
    return np.array([auc(g["abnormal"], g["score"]) for _, g in sc.groupby("fold")])


def recording_metrics(run: str, card_ref: str, pulm_ref: str, out: Path) -> None:
    spr = pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    main_dir, card_dir, pulm_dir = EVAL_DIR / f"{run}_B", EVAL_DIR / card_ref, EVAL_DIR / pulm_ref
    metrics = [("F2 murmullo", lambda d: per_fold_metric(d, "murmur", "f2"), card_dir),
               ("AUC murmullo", lambda d: per_fold_metric(d, "murmur", "auc"), card_dir),
               ("score pulmonar", lambda d: per_fold_metric(d, "pulmonary", "icbhi_score"), pulm_dir),
               ("AUC crackle", lambda d: per_fold_metric(d, "pulmonary", "crackle_auc"), pulm_dir),
               ("AUC wheeze", lambda d: per_fold_metric(d, "pulmonary", "wheeze_auc"), pulm_dir),
               ("AUC normal-anormal\n(SPRSound)", None, pulm_dir)]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(metrics))
    width = 0.36
    for k, (label, color) in enumerate(MODEL_COLORS.items()):
        vals = []
        for name, fn, ref_dir in metrics:
            if fn is None:
                vals.append(rec_auc_per_fold(run if k == 0 else pulm_ref, spr))
            else:
                vals.append(fn(main_dir if k == 0 else ref_dir))
        mean = [v.mean() for v in vals]
        pos = x + (k - 0.5) * width
        ax.bar(pos, mean, width - 0.04, color=color, alpha=0.85,
               label=f"{run} (modelo de 6 clases)" if k == 0 else f"ablaciones ({card_ref} / {pulm_ref})")
        for p, v in zip(pos, vals):   # cada fold, para ver la dispersión
            ax.scatter(np.full(len(v), p), v, color="black", s=10, zorder=3)
        for p, m in zip(pos, mean):
            ax.text(p, 0.42, f"{m:.3f}", ha="center", va="bottom", fontsize=8, color="white")
    ax.set_xticks(x, [m[0] for m in metrics], fontsize=9)
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("media en 5 folds (puntos: cada fold)")
    ax.set_title("Métricas por grabación (test, estrategia pooled_fixed)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "recording_metrics.png", dpi=110)
    plt.close(fig)


def domain_separation(run_eval: Path, run: str, out: Path) -> None:
    g = pd.read_csv(run_eval / "domain_by_recording.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 41)
    for domain, label, color in (("cardiac", "CirCor (cardíaco)", "tab:red"), ("pulmonary", "SPRSound (pulmonar)", "tab:green")):
        v = g.loc[g.domain == domain, "p_cardiac"]
        ax.hist(v, bins=bins, color=color, alpha=0.6, label=f"{label}, n={len(v)}")
    errors = int((~g["domain_ok"]).sum())
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("P(cardíaco) media de las ventanas de la grabación")
    ax.set_ylabel("grabaciones (escala log)")
    ax.set_title(f"{run} — dominio por grabación (test): {errors} errores de {len(g)}")
    ax.legend(fontsize=9, loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "domain_separation.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="cs_base")
    parser.add_argument("--card-ref", default="abl_card")
    parser.add_argument("--pulm-ref", default="spr_inv")
    args = parser.parse_args()
    out = EVAL_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)

    run_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / args.run
    w = load_window_probs(run_dir).rename(columns={"prob": "p_murmur"}).query("role == 'test'")
    refs = pd.concat([load_window_probs(UNIFIED_MODEL_REPORTS_DIR / "runs" / args.card_ref).assign(run=args.card_ref),
                      load_window_probs(SPRSOUND_REPORTS_DIR / "runs" / args.pulm_ref).assign(run=args.pulm_ref)])
    refs = refs.rename(columns={"prob": "p_murmur"}).query("role == 'test'")
    run_eval = EVAL_DIR / f"{args.run}_B"

    learning_curves(args.run, run_dir, out)
    roc_windows({"modelo de 6 clases": w.assign(run=args.run), "ablación": refs}, out)
    confusion_windows(w, args.run, out)
    confusion_system(run_eval, args.run, out)
    recording_metrics(args.run, args.card_ref, args.pulm_ref, out)
    domain_separation(run_eval, args.run, out)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
