"""
pulmonary_adventitious_model/run_kfold.py
============================================
Entrena y evalúa el modelo de sonidos adventicios por ciclo (crackle /
wheeze) con validación k-fold por paciente (K=5).

Para cada fold: entrena (train.py), carga el mejor checkpoint, calibra los
dos umbrales sobre validación y predice test. Como cada paciente pasa por
test exactamente una vez, al final se tienen **predicciones out-of-fold
(OOF) para los ~6900 ciclos** — cada una hecha por un modelo que nunca vio
a ese paciente. Sobre esas predicciones se calculan, sin reentrenar:

- el resultado global (matriz de confusión de 4 clases, score ICBHI);
- **el resultado por equipo de grabación y por grupo de edad** — el chequeo
  de atajo de esta rama (análogo a gate_model/diagnostics_source_db.py):
  si el modelo rindiera muy distinto según el equipo o la edad, sería una
  señal de que aprendió algo de la grabación o del paciente y no del sonido
  (ver avances/12, sección 3, para el sesgo equipo/edad del dataset).

Requiere haber corrido antes:
    preprocessing/make_pulmonary_cycles.py
    splits/make_pulmonary_kfold.py

Guarda en reports/pulmonary_adventitious/<run_name>/:
    per_fold.csv, summary.csv          — métricas de test por fold y media ± std
    oof_predictions.csv                — probabilidades y predicción por ciclo
    confusion_matrix_4class.png        — OOF, las 4 clases ICBHI
    metrics_by_equipment.csv / metrics_by_age_group.csv
    training_log_fold{i}.csv, training_curves.png

Uso:
    python pulmonary_adventitious_model/run_kfold.py
    python pulmonary_adventitious_model/run_kfold.py --folds 0 --epochs 2   # prueba rápida
    python pulmonary_adventitious_model/run_kfold.py --pooling avg --run-name v1_logmel   # reproduce v1
    python pulmonary_adventitious_model/run_kfold.py --feature mfcc --run-name v2_mfcc
    python pulmonary_adventitious_model/run_kfold.py --epochs 40 --select auc --run-name v3_auc_40ep
    python pulmonary_adventitious_model/run_kfold.py --epochs 40 --select auc --cycles-variant hp100 --run-name v4_hp100
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    PULMONARY_KFOLD_DIR, PULMONARY_ADV_CHECKPOINTS_DIR, PULMONARY_ADV_REPORTS_DIR,
    pulmonary_cycles_paths,
)
from pulmonary_adventitious_model.dataset import CycleDataset, LABEL_COLUMNS
from pulmonary_adventitious_model.model import AdventitiousCNN, POOLING_TYPES
from pulmonary_adventitious_model.metrics import (
    CLASS_NAMES, calibrate_thresholds, full_metrics, predict_4class, to_4class,
)
from pulmonary_adventitious_model.train import BATCH_SIZE, DROPOUT, EPOCHS, predict_probs, train_fold

K = 5
TARGET_SEC = 4.0  # ~p88 de la duración de ciclo (mediana 2.5 s, p90 4.1 s); ver make_pulmonary_cycles.py
SUMMARY_METRICS = ["icbhi_score", "se", "sp", "accuracy_4class",
                   "crackle_auc", "crackle_precision", "crackle_recall", "crackle_f1",
                   "wheeze_auc", "wheeze_precision", "wheeze_recall", "wheeze_f1"]


def _age_group(age) -> str:
    if pd.isna(age):
        return "desconocida"
    return "pediátrico (<18)" if age < 18 else "adulto"


def _group_metrics(oof: pd.DataFrame, column: str) -> pd.DataFrame:
    """Métricas OOF por grupo, usando para cada ciclo los umbrales de su fold."""
    rows = []
    for value, g in oof.groupby(column):
        labels = g[LABEL_COLUMNS].to_numpy()
        true4 = to_4class(labels[:, 0], labels[:, 1])
        pred4 = g["pred_4class"].to_numpy()
        normal = true4 == 0
        se = (pred4[~normal] == true4[~normal]).mean() if (~normal).any() else np.nan
        sp = (pred4[normal] == 0).mean() if normal.any() else np.nan
        rows.append({
            column: value, "pacientes": g["subject_id"].nunique(), "ciclos": len(g),
            "pct_crackle": 100 * labels[:, 0].mean(), "pct_wheeze": 100 * labels[:, 1].mean(),
            "se": se, "sp": sp, "icbhi_score": np.nanmean([se, sp]) if normal.any() and (~normal).any() else np.nan,
        })
    return pd.DataFrame(rows)


def _plot_training_curves(histories: list[pd.DataFrame], out_png: Path) -> None:
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(12, 4))
    for i, h in enumerate(histories):
        ax_loss.plot(h["epoch"], h["train_loss"], color=f"C{i}", label=f"fold {i} train")
        ax_loss.plot(h["epoch"], h["val_loss"], color=f"C{i}", linestyle="--", label=f"fold {i} val")
        ax_score.plot(h["epoch"], h["val_icbhi_score"], color=f"C{i}", label=f"fold {i}")
        if "val_mean_auc" in h:
            ax_score.plot(h["epoch"], h["val_mean_auc"], color=f"C{i}", linestyle="--")
    ax_loss.set_title("Loss por época (— train, -- val)")
    ax_loss.set_xlabel("Época")
    ax_score.set_title("Validación: — score ICBHI (umbrales calibrados), -- AUC medio")
    ax_score.set_xlabel("Época")
    ax_score.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def run_kfold(folds: list[int], epochs: int, feature_type: str, pooling: str, run_name: str,
              select_metric: str = "icbhi", cycles_variant: str | None = None) -> None:
    PULMONARY_CYCLES_DIR, PULMONARY_CYCLES_INDEX_CSV = pulmonary_cycles_paths(cycles_variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    torch.manual_seed(42)
    np.random.seed(42)

    cycles = pd.read_csv(PULMONARY_CYCLES_INDEX_CSV)
    reports_dir = PULMONARY_ADV_REPORTS_DIR / run_name
    checkpoints_dir = PULMONARY_ADV_CHECKPOINTS_DIR / run_name
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_kwargs = {"target_sec": TARGET_SEC, "feature_type": feature_type}
    model_kwargs = {"pooling": pooling}
    wav_cache: dict = {}  # compartida entre folds: el audio no cambia, solo el split

    per_fold, oof_parts, histories = [], [], []
    for i in folds:
        print(f"\n{'=' * 60}\n  FOLD {i}  ({run_name})\n{'=' * 60}")
        split_map = pd.read_csv(PULMONARY_KFOLD_DIR / f"fold{i}.csv")
        df = cycles.merge(split_map, on="subject_id", how="inner")
        train_df, val_df, test_df = (df[df["split"] == s] for s in ("train", "val", "test"))

        ckpt = checkpoints_dir / f"fold{i}_best.pt"
        history = train_fold(train_df, val_df, PULMONARY_CYCLES_DIR, ckpt, device,
                             epochs=epochs, wav_cache=wav_cache,
                             dataset_kwargs=dataset_kwargs, model_kwargs=model_kwargs,
                             select_metric=select_metric, log_prefix=f"  [fold {i}] ")
        h = pd.DataFrame(history)
        h.to_csv(reports_dir / f"training_log_fold{i}.csv", index=False)
        histories.append(h)

        model = AdventitiousCNN(dropout=DROPOUT, **model_kwargs).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))

        def loader(split_df):
            ds = CycleDataset(split_df, PULMONARY_CYCLES_DIR, augment=False,
                              wav_cache=wav_cache, **dataset_kwargs)
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

        val_labels, val_probs = predict_probs(model, loader(val_df), device)
        thresholds, val_score = calibrate_thresholds(val_labels, val_probs)
        test_labels, test_probs = predict_probs(model, loader(test_df), device)

        m = full_metrics(test_labels, test_probs, thresholds)
        m.update({"fold": i, "val_icbhi_score": val_score,
                  "test_pacientes": test_df["subject_id"].nunique(), "test_ciclos": len(test_df)})
        per_fold.append(m)
        print(f"  Fold {i} — TEST score={m['icbhi_score']:.4f} (Se={m['se']:.3f} Sp={m['sp']:.3f}) | "
              f"AUC crackle={m['crackle_auc']:.3f} wheeze={m['wheeze_auc']:.3f} | "
              f"umbrales={thresholds}")

        part = test_df[["cycle_id", "subject_id", "pathology_label", "equipment",
                        "age_years", "crackle", "wheeze"]].copy()
        part["fold"] = i
        part["p_crackle"], part["p_wheeze"] = test_probs[:, 0], test_probs[:, 1]
        part["pred_4class"] = predict_4class(test_probs, thresholds)
        oof_parts.append(part)

    # ── Reportes ──────────────────────────────────────────────────────────────
    per_fold_df = pd.DataFrame(per_fold)
    per_fold_df.to_csv(reports_dir / "per_fold.csv", index=False)
    summary = pd.DataFrame([{"metrica": k, "media": per_fold_df[k].mean(), "std": per_fold_df[k].std()}
                            for k in SUMMARY_METRICS])
    summary.to_csv(reports_dir / "summary.csv", index=False)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof["true_4class"] = to_4class(oof["crackle"].to_numpy(), oof["wheeze"].to_numpy())
    oof["age_group"] = oof["age_years"].map(_age_group)
    oof.to_csv(reports_dir / "oof_predictions.csv", index=False)

    by_equipment = _group_metrics(oof, "equipment")
    by_age = _group_metrics(oof, "age_group")
    by_equipment.to_csv(reports_dir / "metrics_by_equipment.csv", index=False)
    by_age.to_csv(reports_dir / "metrics_by_age_group.csv", index=False)

    cm = confusion_matrix(oof["true_4class"], oof["pred_4class"], labels=range(4))
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
    ax.set_title(f"Sonidos adventicios — OOF ({len(folds)} folds, {run_name})")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix_4class.png", dpi=120)
    plt.close(fig)
    _plot_training_curves(histories, reports_dir / "training_curves.png")

    pd.set_option("display.float_format", "{:.3f}".format)
    print(f"\n{'=' * 60}\n  RESUMEN K-FOLD ({run_name}, folds={folds})\n{'=' * 60}")
    print(summary.to_string(index=False))
    print("\nMatriz de confusión OOF (filas=real, columnas=predicho):")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
    print("\nPor equipo:")
    print(by_equipment.to_string(index=False))
    print("\nPor grupo de edad:")
    print(by_age.to_string(index=False))
    print(f"\n✓ Reportes en: {reports_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--feature", choices=["logmel", "mfcc"], default="logmel")
    parser.add_argument("--pooling", choices=list(POOLING_TYPES), default="time_max")
    parser.add_argument("--select", choices=["icbhi", "auc"], default="icbhi",
                        help="Criterio para elegir el mejor checkpoint en validación")
    parser.add_argument("--cycles-variant", default=None,
                        help="Variante de preprocesamiento de los ciclos (ej. hp100), "
                             "ver preprocessing/make_pulmonary_cycles.py")
    parser.add_argument("--run-name", default=None,
                        help="Carpeta de reportes/checkpoints (default: v2_<feature>)")
    args = parser.parse_args()
    run_kfold(args.folds, args.epochs, args.feature, args.pooling,
              args.run_name or f"v2_{args.feature}", select_metric=args.select,
              cycles_variant=args.cycles_variant)
