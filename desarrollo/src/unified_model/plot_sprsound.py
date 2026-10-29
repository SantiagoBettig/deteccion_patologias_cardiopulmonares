"""
unified_model/plot_sprsound.py
=================================
Figuras del paso 3 de SPRSound (avances/23): la rama pulmonar chica entrenada
solo con SPRSound (run_kfold.py --data sprsound), con abl_pulm (ICBHI →
ICBHI) de referencia. Lee los history_fold*.csv y window_probs_fold*.csv de
cada corrida y reports/sprsound/eval/sprsound_by_release.csv
(evaluate_sprsound.py).

Salida: reports/sprsound/figures/
    learning_curves.png   loss train/val y AUC de selección en val por época, por fold
    roc_windows.png       ROC por ventana de crackle y wheeze (test de todos los folds juntos)
    confusion.png         matriz de confusión de las 4 clases pulmonares (argmax), por fila
    auc_by_release.png    AUC normal vs anormal por grabación, por año de SPRSound

Uso:
    python unified_model/plot_sprsound.py [--runs spr_inv spr_sqrt] [--ref abl_pulm]
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
from common.paths import SPRSOUND_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR
from unified_model.evaluate_unified import K, load_window_probs

PULM = ["normal", "crackle", "wheeze", "both"]   # clases 2-5
TESTS = {"SPRSound": "test", "ICBHI (cruzado)": "test_icbhi"}


def learning_curves(runs: dict[str, Path], out: Path) -> None:
    fig, axes = plt.subplots(2, len(runs), figsize=(6 * len(runs), 7), sharex=True, squeeze=False)
    colors = plt.cm.tab10(np.arange(K))
    for j, (run, d) in enumerate(runs.items()):
        for f in range(K):
            h = pd.read_csv(d / f"history_fold{f}.csv")
            best = h.loc[h["select"].idxmax()]
            axes[0, j].plot(h["epoch"], h["train_loss"], color=colors[f], ls="--", lw=1)
            axes[0, j].plot(h["epoch"], h["val_loss"], color=colors[f], lw=1.5, label=f"fold {f}")
            axes[1, j].plot(h["epoch"], h["select"], color=colors[f], lw=1.5)
            axes[1, j].scatter(best["epoch"], best["select"], color=colors[f], s=40, zorder=3)
        axes[0, j].set_title(f"{run} — loss (línea: val, guiones: train)")
        axes[1, j].set_title(f"{run} — AUC de selección en val (punto: checkpoint)")
        axes[1, j].set_xlabel("época")
        axes[0, j].legend(fontsize=8)
        for ax in axes[:, j]:
            ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "learning_curves.png", dpi=110)
    plt.close(fig)


def roc_windows(probs: dict[str, pd.DataFrame], ref: str, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, task, pos, col in ((axes[0], "crackle", [3, 5], "p_crackle"), (axes[1], "wheeze", [4, 5], "p_wheeze")):
        for run, w in probs.items():
            roles = {"ICBHI": "test"} if run == ref else TESTS
            for test, role in roles.items():
                t = w[w.role == role]
                y = t["label"].isin(pos)
                fpr, tpr, _ = roc_curve(y, t[col])
                ls = "-" if role == "test" and run != ref else (":" if run == ref else "--")
                ax.plot(fpr, tpr, ls=ls, lw=1.5, label=f"{run} → {test} ({roc_auc_score(y, t[col]):.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.set_title(f"ROC por ventana — {task}")
        ax.set_xlabel("1 − especificidad")
        ax.set_ylabel("sensibilidad")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "roc_windows.png", dpi=110)
    plt.close(fig)


def confusion(probs: dict[str, pd.DataFrame], ref: str, out: Path) -> None:
    panels = [(run, test, role) for run in probs if run != ref for test, role in TESTS.items()]
    panels.append((ref, "ICBHI", "test"))
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.8))
    for ax, (run, test, role) in zip(axes, panels):
        t = probs[run][probs[run].role == role]
        p = t[["p_pulm_normal", "p_pulm_crackle", "p_pulm_wheeze", "p_pulm_both"]].to_numpy()
        cm = confusion_matrix(t["label"] - 2, p.argmax(1), labels=range(4), normalize="true")
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
        ax.set_xticks(range(4), PULM, rotation=45, fontsize=8)
        ax.set_yticks(range(4), PULM, fontsize=8)
        ax.set_title(f"{run} → {test}", fontsize=9)
        ax.set_xlabel("predicha", fontsize=8)
    axes[0].set_ylabel("real")
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=110)
    plt.close(fig)


def auc_by_release(out: Path) -> None:
    rel = pd.read_csv(SPRSOUND_REPORTS_DIR / "eval" / "sprsound_by_release.csv")
    tab = rel.pivot(index="release", columns="model", values="auc_normal_vs_abnormal")
    n = rel.groupby("release")[["n_recordings", "abnormal_frac"]].first()
    fig, ax = plt.subplots(figsize=(8, 4))
    tab.plot.bar(ax=ax, rot=0)
    ax.set_xticks(range(len(tab)), [f"{r}\nn={n.loc[r, 'n_recordings']}, anorm. {n.loc[r, 'abnormal_frac']:.0%}"
                                    for r in tab.index], fontsize=7)
    ax.axhline(0.5, color="grey", lw=0.8)
    ax.set_ylim(0.4, 1)
    ax.set_ylabel("AUC normal vs anormal (grabación)")
    ax.set_title("SPRSound test por año (con el sesgo de ventanas normales descartadas)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "auc_by_release.png", dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=["spr_inv", "spr_sqrt"])
    parser.add_argument("--ref", default="abl_pulm")
    args = parser.parse_args()
    out = SPRSOUND_REPORTS_DIR / "figures"
    out.mkdir(parents=True, exist_ok=True)

    runs = {r: SPRSOUND_REPORTS_DIR / "runs" / r for r in args.runs}
    probs = {r: load_window_probs(d) for r, d in runs.items()}
    probs[args.ref] = load_window_probs(UNIFIED_MODEL_REPORTS_DIR / "runs" / args.ref)
    probs[args.ref] = probs[args.ref][probs[args.ref].domain == "pulmonary"]

    learning_curves(runs, out)
    roc_windows(probs, args.ref, out)
    confusion(probs, args.ref, out)
    auc_by_release(out)
    print(f"✓ {out}")


if __name__ == "__main__":
    main()
