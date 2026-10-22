"""
integration/evaluate_integration.py
======================================
Pasos 2-5 de la integración (avances/18 §5, avances/19-20), a partir de las
probabilidades por ventana de integration/window_inference.py (no corre
ningún modelo).

1. Agregación ventana -> grabación. Reglas candidatas, para cada probabilidad
   (murmullo; crackle y wheeze):
       mean  probabilidad media de las ventanas
       p75   percentil 75
       max   máximo (acumuló falsos positivos en murmullo, avances/10 §14)
       frac  proporción de ventanas positivas, con el umbral POR VENTANA ya
             calibrado de cada fold (per_fold.csv de cada modelo)
   El umbral de grabación se calibra con máximo F2 (murmullo) o máximo
   score ICBHI por grabación, umbrales crackle/wheeze en conjunto
   (pulmonar). Estrategias de calibración (avances/20 §2):
       fold_auto     validación del fold (16 pacientes en pulmonar); regla
                     elegida en cada fold. Es la de avances/19.
       fixed_rule    igual, pero una sola regla para todos los folds (la de
                     mejor score medio de validación).
       pooled        validación de TODOS los folds juntos, menos los
                     pacientes que son test del fold que se evalúa (~100
                     pacientes en pulmonar). Cada probabilidad viene de un
                     modelo que no vio a ese paciente; test nunca entra.
       pooled_fixed  pooled + una sola regla para todos los folds.
   Test nunca participa de ninguna elección.

2. Capa 2: métricas por grabación de cada rama sobre todos los pacientes
   (cada uno en test exactamente una vez), media ± desvío entre folds y
   desglose por edad / equipo / ubicación.

3. Comparación con la segmentación oráculo: el mismo procedimiento sobre las
   ventanas de 4 latidos (.tsv de CirCor) y los ciclos anotados (ICBHI).

4. Capa 1: pipeline de punta a punta sobre los pacientes de test del gate
   (split único): gate v5 por grabación (probabilidad media de sus ventanas
   de 5 s > 0.5) -> rama elegida -> resultado, con la rama del fold donde
   el paciente estaba en test. Se compara contra ruteo perfecto.

5. Barrido de duración de ventana (si existen los CSV con sufijo _<d>s de
   window_inference.py --suffix): la duración se elige por el score de
   VALIDACIÓN, no por test.

Salida: reports/integration/ (CSV + matrices de confusión).

Uso:
    python integration/evaluate_integration.py [--strategy pooled_fixed]
        [--murmur-probs murmur.csv] [--pulmonary-probs pulmonary.csv]
"""

import argparse

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
STRATEGIES = ["fold_auto", "fixed_rule", "pooled", "pooled_fixed"]
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


def aggregate(windows: pd.DataFrame, cols: list[str], wthr: dict, keys=KEYS) -> pd.DataFrame:
    """Una fila por grabación (y fold/rol, si los hay); columnas "<regla>__<col>".
    wthr[col] es un dict fold -> umbral por ventana (para "frac")."""
    w = windows.copy()
    for c in cols:
        w[f"pos_{c}"] = (w[c] > w["fold"].map(wthr[c])).astype(float)
    g = w.groupby(keys)
    parts = []
    for c in cols:
        parts += [g[c].mean().rename(f"mean__{c}"), g[c].quantile(0.75).rename(f"p75__{c}"),
                  g[c].max().rename(f"max__{c}"), g[f"pos_{c}"].mean().rename(f"frac__{c}")]
    out = pd.concat(parts, axis=1)
    out["n_windows"] = g.size()
    return out.reset_index()


# ── Métricas y calibración por tarea ──────────────────────────────────────────
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


def icbhi_from_4class(true4, pred4) -> dict:
    normal = true4 == 0
    sp = float((pred4[normal] == 0).mean()) if normal.any() else np.nan
    se = float((pred4[~normal] == true4[~normal]).mean()) if (~normal).any() else np.nan
    return {"se": se, "sp": sp, "icbhi_score": np.nanmean([se, sp])}


class MurmurTask:
    name = "murmur"
    labels = ["murmur"]

    @staticmethod
    def cols(rule):
        return [f"{rule}__prob"]

    def calibrate(self, cal, rule):
        t, score = best_f2_threshold(cal["murmur"], cal[self.cols(rule)[0]])
        return (t,), score

    def metrics(self, df, rule, thr):
        s = df[self.cols(rule)[0]]
        return {"threshold": thr[0], **binary_metrics(df["murmur"], s > thr[0], s)}

    def predictions(self, df, rule, thr):
        part = df[["file_id", "subject_id", "fold", "murmur", "n_windows"]].copy()
        part["score"] = df[self.cols(rule)[0]].to_numpy()
        part["rule"], part["threshold"] = rule, thr[0]
        part["pred"] = (part["score"] > thr[0]).astype(int)
        return part


class PulmonaryTask:
    name = "pulmonary"
    labels = ["has_crackles", "has_wheezes"]

    @staticmethod
    def cols(rule):
        return [f"{rule}__p_crackle", f"{rule}__p_wheeze"]

    def calibrate(self, cal, rule):
        return calibrate_thresholds(cal[self.labels].to_numpy(), cal[self.cols(rule)].to_numpy())

    def metrics(self, df, rule, thr):
        return full_metrics(df[self.labels].to_numpy(), df[self.cols(rule)].to_numpy(), thr)

    def predictions(self, df, rule, thr):
        y = df[self.labels].to_numpy()
        part = df[["file_id", "subject_id", "fold", "has_crackles", "has_wheezes", "n_windows"]].copy()
        part["s_crackle"], part["s_wheeze"] = df[self.cols(rule)[0]].to_numpy(), df[self.cols(rule)[1]].to_numpy()
        part["rule"], part["crackle_threshold"], part["wheeze_threshold"] = rule, thr[0], thr[1]
        part["pred_4class"] = predict_4class(df[self.cols(rule)].to_numpy(), thr)
        part["true_4class"] = to_4class(y[:, 0], y[:, 1])
        return part


MURMUR, PULMONARY = MurmurTask(), PulmonaryTask()


def calibration_set(rec: pd.DataFrame, fold: int, strategy: str) -> pd.DataFrame:
    if strategy in ("fold_auto", "fixed_rule"):
        return rec[(rec.fold == fold) & (rec.role == "val")]
    test_subjects = set(rec.loc[(rec.fold == fold) & (rec.role == "test"), "subject_id"])
    return rec[(rec.role == "val") & ~rec.subject_id.isin(test_subjects)]


def calibrate_folds(rec: pd.DataFrame, task, strategy: str) -> tuple[dict, dict]:
    """calib[(fold, regla)] = (umbrales, score de calibración);
    chosen[fold] = regla elegida. Solo usa grabaciones de validación."""
    calib = {}
    for fold in range(K):
        cal = calibration_set(rec, fold, strategy)
        for rule in RULES:
            calib[fold, rule] = task.calibrate(cal, rule)
    if strategy in ("fixed_rule", "pooled_fixed"):
        mean_val = {r: np.mean([calib[f, r][1] for f in range(K)]) for r in RULES}
        chosen = {f: max(RULES, key=mean_val.get) for f in range(K)}
    else:
        chosen = {f: max(RULES, key=lambda r: calib[f, r][1]) for f in range(K)}
    return calib, chosen


def evaluate_branch(rec: pd.DataFrame, task, strategy: str, setting: str) -> dict:
    """rec: salida de aggregate + etiquetas. Devuelve métricas de test por
    fold para cada regla y para la elegida ("auto"), y las predicciones OOF
    por grabación de la regla elegida."""
    calib, chosen = calibrate_folds(rec, task, strategy)

    rows, oof = [], []
    for fold in range(K):
        test = rec[(rec.fold == fold) & (rec.role == "test")]
        for rule in RULES + ["auto"]:
            r = chosen[fold] if rule == "auto" else rule
            thr, val_score = calib[fold, r]
            rows.append({"setting": setting, "strategy": strategy, "fold": fold, "rule": rule,
                         "chosen": r, "val_score": val_score, **task.metrics(test, r, thr)})
        oof.append(task.predictions(test, chosen[fold], calib[fold, chosen[fold]][0]))
    return {"per_fold": pd.DataFrame(rows), "oof": pd.concat(oof, ignore_index=True)}


# ── Resúmenes ─────────────────────────────────────────────────────────────────
def summarize(per_fold: pd.DataFrame, metrics: list[str], by=("setting", "strategy", "rule")) -> pd.DataFrame:
    g = per_fold.groupby(list(by), sort=False)[["val_score"] + metrics]
    mean, std = g.mean(), g.std()
    out = mean.round(3).astype(str) + " ± " + std.round(3).astype(str)
    out["chosen"] = per_fold.groupby(list(by), sort=False)["chosen"].agg(lambda s: "/".join(s))
    return out.reset_index()


def group_table(oof: pd.DataFrame, recs: pd.DataFrame, by: str, task) -> pd.DataFrame:
    df = oof.merge(recs[["file_id", by]], on="file_id", how="left")
    rows = []
    for value, g in df.groupby(by, dropna=False):
        row = {by: value, "pacientes": g["subject_id"].nunique(), "grabaciones": len(g)}
        if task is MURMUR:
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


def gate_by_recording(gate_windows: pd.DataFrame) -> pd.DataFrame:
    g = gate_windows.groupby("file_id").agg(
        sound_type=("sound_type", "first"), gate_mean=("prob_cardiac", "mean"),
        gate_vote=("prob_cardiac", lambda p: (p > GATE_THRESHOLD).mean()),
        n_windows=("prob_cardiac", "size")).reset_index()
    g["routed"] = np.where(g["gate_mean"] > GATE_THRESHOLD, "cardiac", "pulmonary")
    g["routed_vote"] = np.where(g["gate_vote"] > 0.5, "cardiac", "pulmonary")
    g["gate_ok"] = g["routed"] == g["sound_type"]
    return g


def layer1(murmur_oof: pd.DataFrame, pulm_oof: pd.DataFrame, out_dir: Path) -> dict:
    gate = pd.read_csv(INTEGRATION_PROBS_DIR / "gate.csv")
    gate = gate[gate["split"] == "test"]
    g = gate_by_recording(gate).merge(gate.groupby("file_id")["subject_id"].first().reset_index(), on="file_id")
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


# ── Barrido de duración ───────────────────────────────────────────────────────
def duration_sweep(recs, wthr, out_dir: Path) -> pd.DataFrame | None:
    files = {MURMUR: sorted(INTEGRATION_PROBS_DIR.glob("murmur_*s.csv")) + [INTEGRATION_PROBS_DIR / "murmur.csv"],
             PULMONARY: sorted(INTEGRATION_PROBS_DIR.glob("pulmonary_*s.csv")) + [INTEGRATION_PROBS_DIR / "pulmonary.csv"]}
    default_sec = {MURMUR: "2.5", PULMONARY: "4"}
    rows = []
    for task, paths in files.items():
        if len(paths) < 2:
            continue
        cols = ["prob"] if task is MURMUR else ["p_crackle", "p_wheeze"]
        labels = recs[["file_id"] + task.labels]
        for path in paths:
            sec = path.stem.split("_")[-1][:-1] if path.stem.endswith("s") else default_sec[task]
            rec = aggregate(pd.read_csv(path), cols, wthr).merge(labels, on="file_id")
            for strategy in STRATEGIES:
                pf = evaluate_branch(rec, task, strategy, f"{sec} s")["per_fold"]
                a = pf[pf.rule == "auto"]
                metric = "f2" if task is MURMUR else "icbhi_score"
                auc = ["auc"] if task is MURMUR else ["crackle_auc", "wheeze_auc"]
                rows.append({"rama": task.name, "ventana_s": float(sec), "strategy": strategy,
                             "val_score": a["val_score"].mean(), f"test_{metric}": a[metric].mean(),
                             f"test_{metric}_std": a[metric].std(),
                             **{f"test_{m}": a[m].mean() for m in auc},
                             "reglas": "/".join(a["chosen"])})
    if not rows:
        return None
    sweep = pd.DataFrame(rows).sort_values(["rama", "strategy", "ventana_s"])
    sweep.to_csv(out_dir / "duration_sweep.csv", index=False)
    return sweep


# ── Main ──────────────────────────────────────────────────────────────────────
MURMUR_METRICS = ["accuracy", "precision", "recall", "f1", "f2", "auc"]
PULM_METRICS = ["se", "sp", "icbhi_score", "accuracy_4class", "crackle_auc", "crackle_recall",
                "crackle_precision", "wheeze_auc", "wheeze_recall", "wheeze_precision"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default="pooled_fixed", choices=STRATEGIES,
                        help="calibración usada para las predicciones OOF, desgloses y capa 1")
    parser.add_argument("--murmur-probs", default="murmur.csv")
    parser.add_argument("--pulmonary-probs", default="pulmonary.csv")
    args = parser.parse_args()

    out = INTEGRATION_REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    recs = load_recordings()
    wthr = window_thresholds()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_colwidth", 30)

    results = {}
    for task, probs, oracle, cols, subset_name in [
        (MURMUR, args.murmur_probs, "murmur_oracle.csv", ["prob"], "con .tsv"),
        (PULMONARY, args.pulmonary_probs, "pulmonary_oracle.csv", ["p_crackle", "p_wheeze"], "con .txt"),
    ]:
        labels = recs[["file_id"] + task.labels]
        fixed = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / probs), cols, wthr).merge(labels, on="file_id")
        orac = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / oracle), cols, wthr).merge(labels, on="file_id")
        settings = {"ventanas fijas": fixed,
                    f"ventanas fijas ({subset_name})": fixed[fixed.file_id.isin(set(orac.file_id))],
                    "oráculo": orac}
        res = {(s, st): evaluate_branch(d, task, st, s) for s, d in settings.items() for st in STRATEGIES}
        per_fold = pd.concat([r["per_fold"] for r in res.values()], ignore_index=True)
        per_fold.to_csv(out / f"{task.name}_per_fold.csv", index=False)
        metrics = MURMUR_METRICS if task is MURMUR else PULM_METRICS
        summary = summarize(per_fold, metrics)
        summary.to_csv(out / f"{task.name}_summary.csv", index=False)
        oof = res["ventanas fijas", args.strategy]["oof"]
        oof.to_csv(out / f"{task.name}_recording_oof.csv", index=False)
        results[task] = {"summary": summary, "oof": oof}

    oof_m, oof_p = results[MURMUR]["oof"], results[PULMONARY]["oof"]
    subj = oof_m.groupby("subject_id").agg(murmur=("murmur", "max"), pred=("pred", "max"))
    subj_m = pd.DataFrame([binary_metrics(subj["murmur"], subj["pred"])], index=["paciente (alguna grabación)"])
    subj_m.to_csv(out / "murmur_patient_level.csv")
    for by in ("age_category", "auscultation_location"):
        group_table(oof_m, recs, by, MURMUR).to_csv(out / f"murmur_by_{by}.csv", index=False)
    for by in ("equipment", "age_group"):
        group_table(oof_p, recs, by, PULMONARY).to_csv(out / f"pulmonary_by_{by}.csv", index=False)
    save_confusion(oof_m["murmur"], oof_m["pred"], [0, 1], ["sin murmullo", "murmullo"],
                   "Murmullo por grabación — ventanas fijas (OOF)", out / "murmur_confusion.png")
    save_confusion(oof_p["true_4class"], oof_p["pred_4class"], list(range(4)), CLASS_NAMES,
                   "Pulmonar por grabación — ventanas fijas (OOF)", out / "pulmonary_confusion.png")

    # Chequeo: nivel ventana del oráculo de murmullo (debe parecerse a per_fold.csv)
    om = pd.read_csv(INTEGRATION_PROBS_DIR / "murmur_oracle.csv").merge(recs[["file_id", "murmur"]], on="file_id")
    om = om[om.role == "test"]
    om_chk = pd.DataFrame([{"fold": f, **binary_metrics(d["murmur"], d["prob"] > wthr["prob"][f])}
                           for f, d in om.groupby("fold")])
    om_chk.to_csv(out / "check_murmur_oracle_window_level.csv", index=False)

    l1 = layer1(oof_m, oof_p, out)
    sweep = duration_sweep(recs, wthr, out)

    # ── Consola ───────────────────────────────────────────────────────────────
    print(f"Estrategia para OOF / desgloses / capa 1: {args.strategy} | "
          f"murmullo: {args.murmur_probs} | pulmonar: {args.pulmonary_probs}")
    print(f"\n== Chequeo murmullo oráculo, nivel ventana: F2 medio {om_chk.f2.mean():.3f} (per_fold.csv: 0.737)")
    for task, key in [(MURMUR, ["recall", "precision", "f2", "auc"]),
                      (PULMONARY, ["se", "sp", "icbhi_score", "crackle_auc", "wheeze_auc"])]:
        s = results[task]["summary"]
        print(f"\n== {task.name} por grabación, regla elegida (media ± desvío en 5 folds) ==")
        print(s[s.rule == "auto"][["setting", "strategy", "val_score"] + key + ["chosen"]].to_string(index=False))
    print("\n  Murmullo nivel paciente (OOF):")
    print(subj_m.round(3).to_string())
    print("\n== Capa 1 — sistema completo sobre el test del gate ==")
    for k, v in l1["system"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(l1["branch"].round(3).to_string(index=False))
    if sweep is not None:
        print("\n== Barrido de duración de ventana (elegir por val_score) ==")
        print(sweep.round(3).to_string(index=False))
    print(f"\n✓ Reportes en: {out}")


if __name__ == "__main__":
    main()
