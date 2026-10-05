"""
pulmonary_adventitious_model/run_frozen_probe.py
===================================================
Etapa 1 del transfer learning (ver avances/15): entrena un clasificador
chico sobre los embeddings de la CNN6 preentrenada y congelada (generados
por extract_cnn6_embeddings.py), con el MISMO protocolo que v4 para que la
comparación sea directa:

- mismos 5 folds por paciente (data/unified_v2/pulmonary_kfold/);
- BCEWithLogitsLoss con pos_weight natural por etiqueta (de train, por fold);
- mejor época = mayor AUC medio (crackle, wheeze) en validación;
- los dos umbrales se calibran una vez sobre validación (score ICBHI);
- predicciones out-of-fold (OOF) y desglose por equipo y edad.

Pregunta que responde: ¿lo que la CNN6 aprendió de AudioSet sirve, tal cual,
para detectar crackles/wheezes en ciclos de estetoscopio? Si el clasificador
chico iguala o supera a v4 (entrenada desde cero), vale la pena invertir en
el fine-tuning (etapa 2); si queda muy por debajo, no.

Dos clasificadores:
    "linear": 512 -> 2 (dos regresiones logísticas; el "linear probe" de la
              literatura: mide si la información está directamente en el
              embedding).
    "mlp":    512 -> 128 (ReLU, dropout 0.3) -> 2.
Cada dimensión del embedding se estandariza con media/desvío de train de
cada fold (nunca de val/test).

Uso:
    python pulmonary_adventitious_model/run_frozen_probe.py --frontend ours64
    python pulmonary_adventitious_model/run_frozen_probe.py --frontend panns32k

Guarda en reports/pulmonary_adventitious/v5_cnn6_frozen_<frontend>_<head>/ los
mismos archivos que run_kfold.py (summary.csv, per_fold.csv,
oof_predictions.csv, confusion_matrix_4class.png, metrics_by_*.csv).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PULMONARY_KFOLD_DIR, PULMONARY_ADV_REPORTS_DIR, pulmonary_cycles_paths
from pulmonary_adventitious_model.dataset import LABEL_COLUMNS
from pulmonary_adventitious_model.extract_cnn6_embeddings import CYCLES_VARIANT, embeddings_paths
from pulmonary_adventitious_model.metrics import (
    CLASS_NAMES, calibrate_thresholds, full_metrics, predict_4class, to_4class,
)
from pulmonary_adventitious_model.run_kfold import K, SUMMARY_METRICS, _age_group, _group_metrics

EPOCHS = 100
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HEADS = ("linear", "mlp")


def build_head(kind: str) -> nn.Module:
    if kind == "linear":
        return nn.Linear(512, 2)
    return nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))


def mean_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean([roc_auc_score(labels[:, j], probs[:, j]) for j in range(2)]))


def predict(model, x: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x)).numpy()


def run_fold(kind, x_tr, y_tr, x_va, y_va, seed: int):
    torch.manual_seed(seed)
    model = build_head(kind)
    pos_weight = torch.tensor((len(y_tr) - y_tr.sum(0)) / y_tr.sum(0), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    xt, yt, xv = torch.from_numpy(x_tr), torch.from_numpy(y_tr), torch.from_numpy(x_va)

    best_auc, best_state, best_epoch = -1.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt))
        for start in range(0, len(xt), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            loss = criterion(model(xt[idx]), yt[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        auc = mean_auc(y_va, predict(model, xv))
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_epoch, best_auc


def run_probe(frontend: str) -> None:
    emb_npy, ids_csv = embeddings_paths(frontend)
    emb = np.load(emb_npy)
    ids = pd.read_csv(ids_csv)["cycle_id"]
    cycles = pd.read_csv(pulmonary_cycles_paths(CYCLES_VARIANT)[1]).set_index("cycle_id").loc[ids].reset_index()
    print(f"Embeddings {frontend}: {emb.shape}")

    for kind in HEADS:
        run_name = f"v5_cnn6_frozen_{frontend}_{kind}"
        reports_dir = PULMONARY_ADV_REPORTS_DIR / run_name
        reports_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}\n  {run_name}\n{'=' * 60}")

        per_fold, oof_parts = [], []
        for i in range(K):
            split = cycles[["subject_id"]].merge(
                pd.read_csv(PULMONARY_KFOLD_DIR / f"fold{i}.csv"), on="subject_id", how="left")["split"].to_numpy()
            tr, va, te = (split == "train"), (split == "val"), (split == "test")
            mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-6       # solo train
            x = ((emb - mu) / sd).astype(np.float32)
            y = cycles[LABEL_COLUMNS].to_numpy(dtype=np.float32)

            model, best_epoch, val_auc = run_fold(kind, x[tr], y[tr], x[va], y[va], seed=42 + i)
            val_probs = predict(model, torch.from_numpy(x[va]))
            thresholds, val_score = calibrate_thresholds(y[va], val_probs)
            test_probs = predict(model, torch.from_numpy(x[te]))

            m = full_metrics(y[te], test_probs, thresholds)
            m.update({"fold": i, "best_epoch": best_epoch, "val_mean_auc": val_auc,
                      "val_icbhi_score": val_score})
            per_fold.append(m)
            print(f"  Fold {i} — TEST score={m['icbhi_score']:.4f} (Se={m['se']:.3f} Sp={m['sp']:.3f}) | "
                  f"AUC crackle={m['crackle_auc']:.3f} wheeze={m['wheeze_auc']:.3f} | "
                  f"mejor época {best_epoch} (val AUC {val_auc:.3f})")

            part = cycles.loc[te, ["cycle_id", "subject_id", "pathology_label", "equipment",
                                   "age_years", "crackle", "wheeze"]].copy()
            part["fold"] = i
            part["p_crackle"], part["p_wheeze"] = test_probs[:, 0], test_probs[:, 1]
            part["pred_4class"] = predict_4class(test_probs, thresholds)
            oof_parts.append(part)

        per_fold_df = pd.DataFrame(per_fold)
        per_fold_df.to_csv(reports_dir / "per_fold.csv", index=False)
        summary = pd.DataFrame([{"metrica": k, "media": per_fold_df[k].mean(), "std": per_fold_df[k].std()}
                                for k in SUMMARY_METRICS])
        summary.to_csv(reports_dir / "summary.csv", index=False)

        oof = pd.concat(oof_parts, ignore_index=True)
        oof["true_4class"] = to_4class(oof["crackle"].to_numpy(), oof["wheeze"].to_numpy())
        oof["age_group"] = oof["age_years"].map(_age_group)
        oof.to_csv(reports_dir / "oof_predictions.csv", index=False)
        by_equipment, by_age = _group_metrics(oof, "equipment"), _group_metrics(oof, "age_group")
        by_equipment.to_csv(reports_dir / "metrics_by_equipment.csv", index=False)
        by_age.to_csv(reports_dir / "metrics_by_age_group.csv", index=False)

        cm = confusion_matrix(oof["true_4class"], oof["pred_4class"], labels=range(4))
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(ax=ax, colorbar=False)
        ax.set_title(f"Sonidos adventicios — OOF ({run_name})")
        fig.tight_layout()
        fig.savefig(reports_dir / "confusion_matrix_4class.png", dpi=120)
        plt.close(fig)

        pd.set_option("display.float_format", "{:.3f}".format)
        print(f"\n  RESUMEN {run_name}")
        print(summary.to_string(index=False))
        print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())
        print(by_equipment[["equipment", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
        print(by_age[["age_group", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend", choices=["panns32k", "ours64"], required=True)
    run_probe(parser.parse_args().frontend)
