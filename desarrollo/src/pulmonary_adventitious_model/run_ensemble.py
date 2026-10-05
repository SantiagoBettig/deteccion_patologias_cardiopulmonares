"""
pulmonary_adventitious_model/run_ensemble.py
===============================================
Ensamble de dos modelos de naturaleza muy distinta (ver avances/16):

    A = v4: CNN de 3 bloques entrenada desde cero (checkpoints de
        run_kfold.py, run v4_hp100).
    B = CNN6 de PANNs congelada + MLP (run_frozen_probe.py, entrada
        panns32k) — se reentrena acá el MLP con la misma semilla (segundos),
        porque hacen falta sus probabilidades de VALIDACIÓN, que
        run_frozen_probe.py no guarda.

Hipótesis: si A y B se equivocan en ciclos distintos, promediar sus
probabilidades mejora a los dos.

Para cada fold (los mismos 5 folds por paciente):
    1. Probabilidades de val y test de A y de B.
    2. Ensamble = promedio simple, p = (p_A + p_B) / 2 — sin pesos
       ajustados, para no sobreajustar a los 16 pacientes de val.
       Como referencia secundaria, también el peso w de p = w·p_A + (1-w)·p_B
       que maximiza el AUC medio de val (grilla 0.0-1.0).
    3. Umbrales calibrados sobre val (igual que siempre) y métricas de test.

Además reporta la correlación entre las probabilidades de A y B en test: si
es muy alta, los modelos "piensan" parecido y el ensamble no puede aportar
mucho.

Uso:
    python pulmonary_adventitious_model/run_ensemble.py
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    PULMONARY_KFOLD_DIR, PULMONARY_ADV_CHECKPOINTS_DIR, PULMONARY_ADV_REPORTS_DIR,
    pulmonary_cycles_paths,
)
from pulmonary_adventitious_model.dataset import CycleDataset, LABEL_COLUMNS
from pulmonary_adventitious_model.model import AdventitiousCNN
from pulmonary_adventitious_model.metrics import calibrate_thresholds, full_metrics, predict_4class, to_4class
from pulmonary_adventitious_model.train import BATCH_SIZE, DROPOUT, predict_probs
from pulmonary_adventitious_model.run_kfold import K, SUMMARY_METRICS, TARGET_SEC, _age_group, _group_metrics
from pulmonary_adventitious_model.extract_cnn6_embeddings import embeddings_paths
from pulmonary_adventitious_model.run_frozen_probe import mean_auc, predict as predict_head, run_fold

V4_RUN = "v4_hp100"
CYCLES_VARIANT = "hp100"
CNN6_FRONTEND = "panns32k"
WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 1)


def main() -> None:
    torch.manual_seed(42)
    cycles_dir, index_csv = pulmonary_cycles_paths(CYCLES_VARIANT)
    cycles = pd.read_csv(index_csv)
    emb = np.load(embeddings_paths(CNN6_FRONTEND)[0])
    emb_ids = pd.read_csv(embeddings_paths(CNN6_FRONTEND)[1])["cycle_id"]
    assert (emb_ids.to_numpy() == cycles["cycle_id"].to_numpy()).all(), "orden de embeddings != índice"
    y_all = cycles[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    wav_cache: dict = {}
    reports_dir = PULMONARY_ADV_REPORTS_DIR / "v6_ensemble_v4_cnn6"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows, oof_parts, corr = [], [], []
    for i in range(K):
        split = cycles[["subject_id"]].merge(
            pd.read_csv(PULMONARY_KFOLD_DIR / f"fold{i}.csv"), on="subject_id", how="left")["split"].to_numpy()
        va, te, tr = split == "val", split == "test", split == "train"

        # ── A: v4 ────────────────────────────────────────────────────────────
        model_a = AdventitiousCNN(dropout=DROPOUT, pooling="time_max")
        model_a.load_state_dict(torch.load(PULMONARY_ADV_CHECKPOINTS_DIR / V4_RUN / f"fold{i}_best.pt",
                                           map_location="cpu"))

        def probs_a(mask):
            ds = CycleDataset(cycles[mask], cycles_dir, target_sec=TARGET_SEC, feature_type="logmel",
                              augment=False, wav_cache=wav_cache)
            return predict_probs(model_a, DataLoader(ds, batch_size=BATCH_SIZE), "cpu")[1]

        pa_va, pa_te = probs_a(va), probs_a(te)

        # ── B: CNN6 congelada + MLP (misma receta y semilla que run_frozen_probe) ──
        mu, sd = emb[tr].mean(0), emb[tr].std(0) + 1e-6
        x = torch.from_numpy(((emb - mu) / sd).astype(np.float32))
        model_b, _, _ = run_fold("mlp", x[tr].numpy(), y_all[tr], x[va].numpy(), y_all[va], seed=42 + i)
        pb_va, pb_te = predict_head(model_b, x[va]), predict_head(model_b, x[te])

        y_va, y_te = y_all[va], y_all[te]
        corr.append({"fold": i, **{f"corr_{name}": np.corrcoef(pa_te[:, j], pb_te[:, j])[0, 1]
                                   for j, name in enumerate(LABEL_COLUMNS)}})

        # peso w ajustado sobre val (referencia secundaria)
        w_best = max(WEIGHT_GRID, key=lambda w: mean_auc(y_va, w * pa_va + (1 - w) * pb_va))

        candidates = {
            "A_v4": (pa_va, pa_te),
            "B_cnn6_congelada": (pb_va, pb_te),
            "ensamble_promedio": ((pa_va + pb_va) / 2, (pa_te + pb_te) / 2),
            "ensamble_peso_val": (w_best * pa_va + (1 - w_best) * pb_va, w_best * pa_te + (1 - w_best) * pb_te),
        }
        for name, (p_va, p_te) in candidates.items():
            thresholds, _ = calibrate_thresholds(y_va, p_va)
            m = full_metrics(y_te, p_te, thresholds)
            m.update({"modelo": name, "fold": i, "w_v4": w_best if name == "ensamble_peso_val" else np.nan})
            rows.append(m)
            if name == "ensamble_promedio":
                part = cycles.loc[te, ["cycle_id", "subject_id", "equipment", "age_years",
                                       "crackle", "wheeze"]].copy()
                part["pred_4class"] = predict_4class(p_te, thresholds)
                part["fold"] = i
                part["p_crackle"], part["p_wheeze"] = p_te[:, 0], p_te[:, 1]
                oof_parts.append(part)
        print(f"Fold {i}: " + " | ".join(
            f"{r['modelo']}={r['icbhi_score']:.3f}" for r in rows[-4:]) + f" | w_v4(val)={w_best}")

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(reports_dir / "per_fold.csv", index=False)
    summary = per_fold.groupby("modelo", sort=False)[SUMMARY_METRICS].agg(["mean", "std"])
    summary.to_csv(reports_dir / "summary.csv")
    corr_df = pd.DataFrame(corr)
    corr_df.to_csv(reports_dir / "correlation_v4_cnn6.csv", index=False)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof["age_group"] = oof["age_years"].map(_age_group)
    oof.to_csv(reports_dir / "oof_predictions_ensamble_promedio.csv", index=False)
    by_equipment, by_age = _group_metrics(oof, "equipment"), _group_metrics(oof, "age_group")
    by_equipment.to_csv(reports_dir / "metrics_by_equipment.csv", index=False)
    by_age.to_csv(reports_dir / "metrics_by_age_group.csv", index=False)
    true4 = to_4class(oof["crackle"].to_numpy(), oof["wheeze"].to_numpy())
    cm = pd.crosstab(pd.Series(true4, name="real"), pd.Series(oof["pred_4class"].to_numpy(), name="pred"))

    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.width", 200)
    cols = ["icbhi_score", "se", "sp", "crackle_auc", "wheeze_auc", "crackle_f1", "wheeze_f1"]
    print("\nMedia ± std (5 folds, test):")
    print(pd.DataFrame({c: summary[c]["mean"].map("{:.3f}".format) + " ± " + summary[c]["std"].map("{:.3f}".format)
                        for c in cols}).to_string())
    print(f"\nPesos w_v4 elegidos en val: {per_fold.loc[per_fold.modelo == 'ensamble_peso_val', 'w_v4'].tolist()}")
    print("\nCorrelación de probabilidades v4 vs CNN6 (test):")
    print(corr_df.to_string(index=False))
    print("\nMatriz de confusión OOF — ensamble promedio (0 normal, 1 crackle, 2 wheeze, 3 both):")
    print(cm.to_string())
    print(by_equipment[["equipment", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
    print(by_age[["age_group", "pacientes", "se", "sp", "icbhi_score"]].to_string(index=False))
    print(f"\n✓ Reportes en: {reports_dir}")


if __name__ == "__main__":
    main()
