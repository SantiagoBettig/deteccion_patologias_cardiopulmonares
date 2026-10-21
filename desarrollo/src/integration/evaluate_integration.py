"""
integration/evaluate_integration.py
======================================
Pasos 2-5 de la integración (avances/18 §5), a partir de las probabilidades
por ventana de integration/window_inference.py (no corre ningún modelo).

1. Agregación ventana -> grabación. Reglas candidatas, para cada probabilidad
   (murmullo; crackle y wheeze):
       mean  probabilidad media de las ventanas
       p75   percentil 75
       max   máximo (acumuló falsos positivos en murmullo, avances/10 §14)
       frac  proporción de ventanas positivas, con el umbral POR VENTANA ya
             calibrado de cada fold (per_fold.csv de cada modelo)
   Para cada fold, el umbral de grabación de cada regla se calibra sobre las
   grabaciones de los pacientes de VALIDACIÓN de ese fold (murmullo: máximo
   F2; pulmonar: máximo score ICBHI por grabación, umbrales crackle/wheeze en
   conjunto). La regla "auto" elige, en cada fold, la de mejor resultado en
   validación. Test nunca participa de ninguna elección.

2. Capa 2: métricas por grabación de cada rama sobre todos los pacientes
   (cada uno en test exactamente una vez), media ± desvío entre folds y
   desglose por edad / equipo / ubicación.

3. Comparación con la segmentación oráculo: el mismo procedimiento sobre las
   ventanas de 4 latidos (.tsv de CirCor) y los ciclos anotados (ICBHI), en
   las mismas grabaciones.

4. Capa 1: pipeline de punta a punta sobre los pacientes de test del gate
   (split único): gate v5 por grabación (probabilidad media de sus ventanas
   de 5 s > 0.5) -> rama elegida -> resultado. Cada grabación usa la rama
   del fold donde su paciente estaba en test, con la regla "auto" de ese
   fold. Se compara contra las mismas grabaciones con ruteo perfecto.

Salida: reports/integration/ (CSV + matrices de confusión).

Uso:
    python integration/evaluate_integration.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score, roc_auc_score)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (INTEGRATION_PROBS_DIR, INTEGRATION_REPORTS_DIR, MASTER_CSV_V2,
                          PULMONARY_V8B_DIR, REPORTS_DIR)
from pulmonary_adventitious_model.metrics import (CLASS_NAMES, calibrate_thresholds, full_metrics,
                                                  predict_4class, to_4class)
from pulmonary_adventitious_model.run_kfold import _age_group

K = 5
RULES = ["mean", "p75", "max", "frac"]
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)
GATE_THRESHOLD = 0.5
MURMUR_PER_FOLD_CSV = REPORTS_DIR / "cardiac_murmur" / "kfold" / "per_fold.csv"
KEYS = ["file_id", "subject_id", "fold", "role"]


# ── Datos ─────────────────────────────────────────────────────────────────────
def load_recordings() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CSV_V2)
    for c in ("has_crackles", "has_wheezes"):
        df[c] = df[c].map({True: 1, False: 0, "True": 1, "False": 0})
    df["murmur"] = (df["pathology_label"] == "heart_murmur").astype(int)
    df["age_group"] = df["age_years"].map(_age_group)
    return df


def window_thresholds() -> dict[str, dict[int, float]]:
    """Umbral por ventana calibrado en validación de cada fold, por modelo."""
    m = pd.read_csv(MURMUR_PER_FOLD_CSV).set_index("fold")
    p = pd.read_csv(PULMONARY_V8B_DIR / "per_fold.csv").set_index("fold")
    return {"prob": m["window_threshold"].to_dict(),
            "p_crackle": p["crackle_threshold"].to_dict(),
            "p_wheeze": p["wheeze_threshold"].to_dict()}


def aggregate(windows: pd.DataFrame, cols: list[str], wthr: dict) -> pd.DataFrame:
    """Una fila por (grabación, fold, rol); columnas "<regla>__<col>"."""
    w = windows.copy()
    for c in cols:
        w[f"pos_{c}"] = (w[c] > w["fold"].map(wthr[c])).astype(float)
    g = w.groupby(KEYS)
    parts = []
    for c in cols:
        parts += [g[c].mean().rename(f"mean__{c}"), g[c].quantile(0.75).rename(f"p75__{c}"),
                  g[c].max().rename(f"max__{c}"), g[f"pos_{c}"].mean().rename(f"frac__{c}")]
    out = pd.concat(parts, axis=1)
    out["n_windows"] = g.size()
    return out.reset_index()


# ── Murmullo ──────────────────────────────────────────────────────────────────
def binary_metrics(y, pred, score=None) -> dict:
    out = {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
           "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
           "f2": fbeta_score(y, pred, beta=2.0, zero_division=0)}
    if score is not None:
        out["auc"] = roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan
    return out


def best_f2_threshold(y, score) -> tuple[float, float]:
    best_t, best = 0.5, -1.0
    for t in GRID:
        f2 = fbeta_score(y, score > t, beta=2.0, zero_division=0)
        if f2 > best:
            best, best_t = f2, float(t)
    return best_t, best


def evaluate_murmur(rec: pd.DataFrame, name: str) -> dict:
    """rec: salida de aggregate + columna murmur. Devuelve tablas por regla
    y predicciones OOF (por grabación) de la regla auto."""
    rows, oof = [], []
    for fold in range(K):
        val, test = rec[(rec.fold == fold) & (rec.role == "val")], rec[(rec.fold == fold) & (rec.role == "test")]
        best_rule, best_val = None, -1.0
        fold_rows = {}
        for rule in RULES:
            col = f"{rule}__prob"
            t, val_f2 = best_f2_threshold(val["murmur"], val[col])
            m = binary_metrics(test["murmur"], test[col] > t, test[col])
            fold_rows[rule] = {"fold": fold, "rule": rule, "threshold": t, "val_f2": val_f2, **m}
            if val_f2 > best_val:
                best_val, best_rule = val_f2, rule
        rows += fold_rows.values()
        auto = dict(fold_rows[best_rule], rule="auto", chosen=best_rule)
        rows.append(auto)
        col = f"{best_rule}__prob"
        part = test[["file_id", "subject_id", "fold", "murmur", "n_windows"]].copy()
        part["score"], part["rule"], part["threshold"] = test[col].to_numpy(), best_rule, auto["threshold"]
        part["pred"] = (part["score"] > auto["threshold"]).astype(int)
        oof.append(part)
    per_fold = pd.DataFrame(rows)
    per_fold.insert(0, "setting", name)
    return {"per_fold": per_fold, "oof": pd.concat(oof, ignore_index=True)}


# ── Pulmonar ──────────────────────────────────────────────────────────────────
def evaluate_pulmonary(rec: pd.DataFrame, name: str) -> dict:
    rows, oof = [], []
    for fold in range(K):
        val, test = rec[(rec.fold == fold) & (rec.role == "val")], rec[(rec.fold == fold) & (rec.role == "test")]
        y_val = val[["has_crackles", "has_wheezes"]].to_numpy()
        y_test = test[["has_crackles", "has_wheezes"]].to_numpy()
        best_rule, best_val = None, -1.0
        fold_rows = {}
        for rule in RULES:
            cols = [f"{rule}__p_crackle", f"{rule}__p_wheeze"]
            thr, val_score = calibrate_thresholds(y_val, val[cols].to_numpy())
            m = full_metrics(y_test, test[cols].to_numpy(), thr)
            fold_rows[rule] = {"fold": fold, "rule": rule, "val_icbhi_score": val_score, **m}
            if val_score > best_val:
                best_val, best_rule = val_score, rule
        rows += fold_rows.values()
        auto = dict(fold_rows[best_rule], rule="auto", chosen=best_rule)
        rows.append(auto)
        cols = [f"{best_rule}__p_crackle", f"{best_rule}__p_wheeze"]
        part = test[["file_id", "subject_id", "fold", "has_crackles", "has_wheezes", "n_windows"]].copy()
        part["s_crackle"], part["s_wheeze"] = test[cols[0]].to_numpy(), test[cols[1]].to_numpy()
        part["rule"] = best_rule
        part["pred_4class"] = predict_4class(test[cols].to_numpy(),
                                             (auto["crackle_threshold"], auto["wheeze_threshold"]))
        part["true_4class"] = to_4class(y_test[:, 0], y_test[:, 1])
        oof.append(part)
    per_fold = pd.DataFrame(rows)
    per_fold.insert(0, "setting", name)
    return {"per_fold": per_fold, "oof": pd.concat(oof, ignore_index=True)}


def icbhi_from_4class(true4, pred4) -> dict:
    normal = true4 == 0
    sp = float((pred4[normal] == 0).mean()) if normal.any() else np.nan
    se = float((pred4[~normal] == true4[~normal]).mean()) if (~normal).any() else np.nan
    return {"se": se, "sp": sp, "icbhi_score": np.nanmean([se, sp])}


# ── Resúmenes ─────────────────────────────────────────────────────────────────
def summarize(per_fold: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    g = per_fold.groupby(["setting", "rule"], sort=False)[metrics]
    mean, std = g.mean(), g.std()
    out = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    return out.reset_index()


def group_table(oof: pd.DataFrame, recs: pd.DataFrame, by: str, task: str) -> pd.DataFrame:
    df = oof.merge(recs[["file_id", by]], on="file_id", how="left")
    rows = []
    for value, g in df.groupby(by, dropna=False):
        row = {by: value, "pacientes": g["subject_id"].nunique(), "grabaciones": len(g)}
        if task == "murmur":
            row["pct_murmullo"] = 100 * g["murmur"].mean()
            row.update(binary_metrics(g["murmur"], g["pred"]))
        else:
            row.update(icbhi_from_4class(g["true_4class"].to_numpy(), g["pred_4class"].to_numpy()))
        rows.append(row)
    return pd.DataFrame(rows)


def save_confusion(true, pred, labels, display, title, path) -> None:
    cm = confusion_matrix(true, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 1.3 * len(labels) + 2))
    ConfusionMatrixDisplay(cm, display_labels=display).plot(ax=ax, colorbar=False, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ── Capa 1 ────────────────────────────────────────────────────────────────────
SYSTEM_CLASSES = ["card: sin murmullo", "card: murmullo", "pulm: normal", "pulm: crackle",
                  "pulm: wheeze", "pulm: both"]


def layer1(recs: pd.DataFrame, murmur_oof: pd.DataFrame, pulm_oof: pd.DataFrame, out_dir: Path) -> dict:
    gate = pd.read_csv(INTEGRATION_PROBS_DIR / "gate.csv")
    gate = gate[gate["split"] == "test"]
    g = gate.groupby("file_id").agg(subject_id=("subject_id", "first"), sound_type=("sound_type", "first"),
                                     gate_mean=("prob_cardiac", "mean"),
                                     gate_vote=("prob_cardiac", lambda p: (p > GATE_THRESHOLD).mean()),
                                     n_windows=("prob_cardiac", "size")).reset_index()
    g["routed"] = np.where(g["gate_mean"] > GATE_THRESHOLD, "cardiac", "pulmonary")
    g["routed_vote"] = np.where(g["gate_vote"] > 0.5, "cardiac", "pulmonary")
    g["gate_ok"] = g["routed"] == g["sound_type"]
    gate_win_acc = ((gate["prob_cardiac"] > GATE_THRESHOLD) == (gate["sound_type"] == "cardiac")).mean()

    # verdad y salida de la rama correcta (ruteo perfecto) para cada grabación
    card = murmur_oof.set_index("file_id")
    pulm = pulm_oof.set_index("file_id")
    true_cls, branch_cls = [], []
    for _, r in g.iterrows():
        if r["sound_type"] == "cardiac":
            true_cls.append(int(card.loc[r["file_id"], "murmur"]))
            branch_cls.append(int(card.loc[r["file_id"], "pred"]))
        else:
            true_cls.append(2 + int(pulm.loc[r["file_id"], "true_4class"]))
            branch_cls.append(2 + int(pulm.loc[r["file_id"], "pred_4class"]))
    g["true_class"], g["branch_class"] = true_cls, branch_cls
    # Una grabación ruteada a la rama equivocada no tiene una salida válida de
    # su dominio: queda como "ruteo erróneo" (clase 6) — error en la matriz.
    g["system_class"] = np.where(g["gate_ok"], g["branch_class"], 6)
    g.to_csv(out_dir / "layer1_recordings.csv", index=False)

    labels_disp = SYSTEM_CLASSES + ["ruteo erróneo"]
    save_confusion(g["true_class"], g["system_class"], list(range(7)), labels_disp,
                   "Sistema completo (capa 1, test del gate)", out_dir / "layer1_confusion_system.png")
    cm = pd.DataFrame(confusion_matrix(g["true_class"], g["system_class"], labels=list(range(7))),
                      index=labels_disp, columns=labels_disp).iloc[:6]
    cm.to_csv(out_dir / "layer1_confusion_system.csv")

    # Métricas por rama: sistema (ruteo del gate) vs. ruteo perfecto, mismas grabaciones.
    rows = []
    c = g[g["sound_type"] == "cardiac"]
    y = (c["true_class"] == 1).astype(int)
    for setting, pred in [("ruteo perfecto", c["branch_class"] == 1),
                          ("sistema (gate v5)", (c["system_class"] == 1))]:
        rows.append({"rama": "murmullo", "setting": setting, "grabaciones": len(c),
                     "pacientes": c["subject_id"].nunique(), **binary_metrics(y, pred.astype(int))})
    p = g[g["sound_type"] == "pulmonary"]
    true4 = p["true_class"].to_numpy() - 2
    for setting, pred4 in [("ruteo perfecto", p["branch_class"].to_numpy() - 2),
                           ("sistema (gate v5)", np.where(p["gate_ok"], p["branch_class"] - 2, -1))]:
        rows.append({"rama": "pulmonar", "setting": setting, "grabaciones": len(p),
                     "pacientes": p["subject_id"].nunique(), **icbhi_from_4class(true4, pred4),
                     "accuracy_4class": float((pred4 == true4).mean())})
    branch = pd.DataFrame(rows)
    branch.to_csv(out_dir / "layer1_branch_metrics.csv", index=False)

    system = {
        "grabaciones": len(g), "pacientes": g["subject_id"].nunique(),
        "gate_accuracy_ventana": gate_win_acc,
        "gate_accuracy_grabacion_media": g["gate_ok"].mean(),
        "gate_accuracy_grabacion_votacion": (g["routed_vote"] == g["sound_type"]).mean(),
        "gate_errores_cardiaco_a_pulmonar": int(((g.sound_type == "cardiac") & ~g.gate_ok).sum()),
        "gate_errores_pulmonar_a_cardiaco": int(((g.sound_type == "pulmonary") & ~g.gate_ok).sum()),
        "accuracy_6clases_sistema": float((g["system_class"] == g["true_class"]).mean()),
        "accuracy_6clases_ruteo_perfecto": float((g["branch_class"] == g["true_class"]).mean()),
    }
    pd.Series(system).to_csv(out_dir / "layer1_system.csv", header=["valor"])
    return {"system": system, "branch": branch, "confusion": cm}


# ── Main ──────────────────────────────────────────────────────────────────────
MURMUR_METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc"]
PULM_METRICS = ["se", "sp", "icbhi_score", "accuracy_4class", "crackle_auc", "crackle_recall",
                "crackle_precision", "wheeze_auc", "wheeze_recall", "wheeze_precision"]


def main() -> None:
    out = INTEGRATION_REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    recs = load_recordings()
    wthr = window_thresholds()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # ── Murmullo: ventanas fijas vs. oráculo ──────────────────────────────────
    lab_m = recs[["file_id", "murmur"]]
    fixed_m = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / "murmur.csv"), ["prob"], wthr).merge(lab_m, on="file_id")
    oracle_m = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / "murmur_oracle.csv"), ["prob"], wthr).merge(lab_m, on="file_id")
    common_m = set(oracle_m["file_id"])
    res_m = {
        "ventanas fijas (todas)": evaluate_murmur(fixed_m, "ventanas fijas (todas)"),
        "ventanas fijas (con .tsv)": evaluate_murmur(fixed_m[fixed_m.file_id.isin(common_m)], "ventanas fijas (con .tsv)"),
        "oráculo 4 latidos": evaluate_murmur(oracle_m, "oráculo 4 latidos"),
    }
    per_fold_m = pd.concat([r["per_fold"] for r in res_m.values()], ignore_index=True)
    per_fold_m.to_csv(out / "murmur_per_fold.csv", index=False)
    sum_m = summarize(per_fold_m, MURMUR_METRICS)
    sum_m.to_csv(out / "murmur_summary.csv", index=False)
    oof_m = res_m["ventanas fijas (todas)"]["oof"]
    oof_m.to_csv(out / "murmur_recording_oof.csv", index=False)

    # nivel paciente (positivo si alguna grabación lo es), OOF
    subj = oof_m.groupby("subject_id").agg(murmur=("murmur", "max"), pred=("pred", "max"))
    subj_m = pd.DataFrame([binary_metrics(subj["murmur"], subj["pred"])], index=["paciente (alguna grabación)"])
    subj_m.to_csv(out / "murmur_patient_level.csv")
    for by in ("age_category", "auscultation_location"):
        group_table(oof_m, recs, by, "murmur").to_csv(out / f"murmur_by_{by}.csv", index=False)
    save_confusion(oof_m["murmur"], oof_m["pred"], [0, 1], ["sin murmullo", "murmullo"],
                   "Murmullo por grabación — ventanas fijas (OOF)", out / "murmur_confusion.png")

    # ── Pulmonar: ventanas fijas vs. oráculo ──────────────────────────────────
    lab_p = recs[["file_id", "has_crackles", "has_wheezes"]]
    cols_p = ["p_crackle", "p_wheeze"]
    fixed_p = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / "pulmonary.csv"), cols_p, wthr).merge(lab_p, on="file_id")
    oracle_p = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / "pulmonary_oracle.csv"), cols_p, wthr).merge(lab_p, on="file_id")
    common_p = set(oracle_p["file_id"])
    res_p = {
        "ventanas fijas (todas)": evaluate_pulmonary(fixed_p, "ventanas fijas (todas)"),
        "ventanas fijas (con .txt)": evaluate_pulmonary(fixed_p[fixed_p.file_id.isin(common_p)], "ventanas fijas (con .txt)"),
        "oráculo ciclos": evaluate_pulmonary(oracle_p, "oráculo ciclos"),
    }
    per_fold_p = pd.concat([r["per_fold"] for r in res_p.values()], ignore_index=True)
    per_fold_p.to_csv(out / "pulmonary_per_fold.csv", index=False)
    sum_p = summarize(per_fold_p, PULM_METRICS)
    sum_p.to_csv(out / "pulmonary_summary.csv", index=False)
    oof_p = res_p["ventanas fijas (todas)"]["oof"]
    oof_p.to_csv(out / "pulmonary_recording_oof.csv", index=False)
    for by in ("equipment", "age_group"):
        group_table(oof_p, recs, by, "pulmonary").to_csv(out / f"pulmonary_by_{by}.csv", index=False)
    save_confusion(oof_p["true_4class"], oof_p["pred_4class"], list(range(4)), CLASS_NAMES,
                   "Pulmonar por grabación — ventanas fijas (OOF)", out / "pulmonary_confusion.png")

    # ── Chequeo: nivel ventana del oráculo (debe parecerse a per_fold.csv) ────
    om = pd.read_csv(INTEGRATION_PROBS_DIR / "murmur_oracle.csv").merge(lab_m, on="file_id")
    om = om[om.role == "test"]
    om_chk = pd.DataFrame([{"fold": f, **binary_metrics(d["murmur"], d["prob"] > wthr["prob"][f])}
                           for f, d in om.groupby("fold")])
    om_chk.to_csv(out / "check_murmur_oracle_window_level.csv", index=False)

    # ── Capa 1 ────────────────────────────────────────────────────────────────
    l1 = layer1(recs, oof_m, oof_p, out)

    # ── Consola ───────────────────────────────────────────────────────────────
    print("\n== Chequeo murmullo oráculo, nivel ventana (test) ==")
    print(om_chk.round(3).to_string(index=False))
    print(f"  F2 medio {om_chk.f2.mean():.3f} (per_fold.csv de k-fold: 0.737)")
    print("\n== Murmullo por grabación (media ± desvío en 5 folds) ==")
    print(sum_m.to_string(index=False))
    print("  regla elegida por fold:", per_fold_m.query("rule == 'auto'").groupby("setting", sort=False)["chosen"].apply(list).to_dict())
    print("\n  Nivel paciente (OOF):")
    print(subj_m.round(3).to_string())
    print("\n== Pulmonar por grabación (media ± desvío en 5 folds) ==")
    print(sum_p.to_string(index=False))
    print("  regla elegida por fold:", per_fold_p.query("rule == 'auto'").groupby("setting", sort=False)["chosen"].apply(list).to_dict())
    print("\n== Capa 1 — sistema completo sobre el test del gate ==")
    for k, v in l1["system"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(l1["branch"].round(3).to_string(index=False))
    print(l1["confusion"].to_string())
    print(f"\n✓ Reportes en: {out}")


if __name__ == "__main__":
    main()
