"""
unified_model/evaluate_unified.py
====================================
Evaluación por grabación del modelo único sin gate (avances/21 §4), con el
MISMO procedimiento que integration/evaluate_integration.py, a partir de las
probabilidades por ventana que guarda unified_model/run_kfold.py
(window_probs_fold*.csv: cada grabación aparece como val en un fold y como
test en otro, con el modelo de ese fold). No corre ningún modelo.

Probabilidades por ventana (dataset.derived_probs): P(murmullo) =
card_murmur; P(crackle) = crackle + both; P(wheeze) = wheeze + both;
P(cardíaco) = card_normal + card_murmur.

1. Umbral por ventana de cada fold (lo necesita la regla "frac"), calibrado
   en las ventanas de validación del fold: máximo F2 para murmullo (ventanas
   cardíacas) y máximo score ICBHI en conjunto para crackle/wheeze (ventanas
   pulmonares, etiqueta por superposición). Es lo que hacía cada rama con su
   per_fold.csv.

2. Capa 2 (ramas sobre todos los pacientes): las grabaciones de cada
   dominio VERDADERO, agregadas con las reglas mean/p75/max/frac y
   calibradas con las estrategias de evaluate_integration (default del
   resumen: pooled_fixed). Comparable directamente con
   reports/integration/{murmur,pulmonary}_summary.csv ("ventanas fijas").

3. Dominio por grabación: P(cardíaco) media de las ventanas > 0.5 (la
   misma regla que el gate v5). Accuracy de dominio en todos los pacientes y
   en los de test del gate.

4. Capa 1 (sistema de punta a punta, 6 clases): para cada grabación, el
   dominio predicho y, si es el correcto, la salida de la tarea de ese
   dominio con la regla y los umbrales de la capa 2 (OOF de test). Si el
   dominio es incorrecto queda como "ruteo erróneo", igual que en
   evaluate_integration.layer1. Se informa sobre los 161 pacientes de test
   del gate (comparable con reports/integration/layer1_*.csv) y sobre todos
   los pacientes (el modelo único no necesita un split aparte para el gate).

Salida: reports/unified_model/runs/<run>/eval/

Uso:
    python unified_model/evaluate_unified.py --run-name v1_base [--strategy pooled_fixed]
"""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import INTEGRATION_PROBS_DIR, INTEGRATION_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import (K, MURMUR, MURMUR_METRICS, PULM_METRICS, PULMONARY, STRATEGIES,
                                              SYSTEM_CLASSES, aggregate, best_f2_threshold, binary_metrics,
                                              evaluate_branch, icbhi_from_4class, load_recordings,
                                              save_confusion, summarize)
from pulmonary_adventitious_model.metrics import calibrate_thresholds

DOMAIN_THRESHOLD = 0.5


def load_window_probs(run_dir: Path) -> pd.DataFrame:
    paths = sorted(run_dir.glob("window_probs_fold*.csv"))
    w = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert sorted(w["fold"].unique()) == list(range(K)), f"faltan folds: {sorted(w['fold'].unique())}"
    return w.rename(columns={"p_murmur": "prob"})   # nombre que usa MurmurTask


def window_thresholds(w: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Umbral por ventana de cada fold, calibrado en sus ventanas de val. En
    una ablación por dominio (un solo dominio en w) el otro queda en NaN."""
    thr = {"prob": {}, "p_crackle": {}, "p_wheeze": {}}
    rows = []
    for f in range(K):
        val = w[(w.fold == f) & (w.role == "val")]
        c, p = val[val.domain == "cardiac"], val[val.domain == "pulmonary"]
        thr["prob"][f], f2 = best_f2_threshold(c["label"] == 1, c["prob"]) if len(c) else (np.nan, np.nan)
        labels = np.stack([p["label"].isin([3, 5]), p["label"].isin([4, 5])], axis=1).astype(int)
        (thr["p_crackle"][f], thr["p_wheeze"][f]), score = (
            calibrate_thresholds(labels, p[["p_crackle", "p_wheeze"]].to_numpy()) if len(p) else ((np.nan, np.nan), np.nan))
        rows.append({"fold": f, "murmur_threshold": thr["prob"][f], "val_f2": f2,
                     "crackle_threshold": thr["p_crackle"][f], "wheeze_threshold": thr["p_wheeze"][f],
                     "val_icbhi_score": score})
    return thr, pd.DataFrame(rows)


def domain_by_recording(w: pd.DataFrame) -> pd.DataFrame:
    """Una fila por (grabación, fold, rol): P(cardíaco) media y dominio predicho."""
    g = w.groupby(["file_id", "subject_id", "fold", "role"]).agg(
        domain=("domain", "first"), p_cardiac=("p_cardiac", "mean")).reset_index()
    g["pred_domain"] = np.where(g["p_cardiac"] > DOMAIN_THRESHOLD, "cardiac", "pulmonary")
    g["domain_ok"] = g["pred_domain"] == g["domain"]
    return g


def layer1(dom_test: pd.DataFrame, oof_m: pd.DataFrame, oof_p: pd.DataFrame, files: set | None,
           title: str, out_dir: Path, tag: str) -> dict:
    g = dom_test if files is None else dom_test[dom_test.file_id.isin(files)]
    card, pulm = oof_m.set_index("file_id"), oof_p.set_index("file_id")
    true_cls, task_cls = [], []
    for _, r in g.iterrows():
        if r["domain"] == "cardiac":
            true_cls.append(int(card.loc[r["file_id"], "murmur"]))
            task_cls.append(int(card.loc[r["file_id"], "pred"]))
        else:
            true_cls.append(2 + int(pulm.loc[r["file_id"], "true_4class"]))
            task_cls.append(2 + int(pulm.loc[r["file_id"], "pred_4class"]))
    g = g.assign(true_class=true_cls, task_class=task_cls)
    g["system_class"] = np.where(g["domain_ok"], g["task_class"], 6)
    g.to_csv(out_dir / f"layer1_{tag}_recordings.csv", index=False)
    save_confusion(g["true_class"], g["system_class"], list(range(7)), SYSTEM_CLASSES + ["dominio erróneo"],
                   title, out_dir / f"layer1_{tag}_confusion.png")

    c, p = g[g.domain == "cardiac"], g[g.domain == "pulmonary"]
    true4 = p["true_class"].to_numpy() - 2
    pred4 = np.where(p["domain_ok"], p["task_class"] - 2, -1)
    return {
        "grabaciones": len(g), "pacientes": g["subject_id"].nunique(),
        "accuracy_dominio": g["domain_ok"].mean(),
        "errores_cardiaco_a_pulmonar": int(((g.domain == "cardiac") & ~g.domain_ok).sum()),
        "errores_pulmonar_a_cardiaco": int(((g.domain == "pulmonary") & ~g.domain_ok).sum()),
        "accuracy_6clases_sistema": float((g["system_class"] == g["true_class"]).mean()),
        "accuracy_6clases_dominio_perfecto": float((g["task_class"] == g["true_class"]).mean()),
        "murmullo_f2_sistema": binary_metrics((c.true_class == 1).astype(int), (c.system_class == 1).astype(int))["f2"],
        "pulmonar_score_sistema": icbhi_from_4class(true4, pred4)["icbhi_score"],
    }


def reference_rows(task) -> pd.DataFrame:
    """Fila de la capa 2 del pipeline con gate (ventanas fijas, regla elegida)."""
    s = pd.read_csv(INTEGRATION_REPORTS_DIR / f"{task.name}_summary.csv")
    return s[(s.setting == "ventanas fijas") & (s.rule == "auto")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--strategy", default="pooled_fixed", choices=STRATEGIES)
    args = parser.parse_args()

    run_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / args.run_name
    out = run_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    recs = load_recordings()
    w = load_window_probs(run_dir)
    wthr, wthr_df = window_thresholds(w)
    wthr_df.to_csv(out / "window_thresholds.csv", index=False)

    # ── Capa 2 ───────────────────────────────────────────────────────────────
    results = {}
    for task, domain, cols in [(MURMUR, "cardiac", ["prob"]), (PULMONARY, "pulmonary", ["p_crackle", "p_wheeze"])]:
        rec = aggregate(w[w.domain == domain], cols, wthr).merge(recs[["file_id"] + task.labels], on="file_id")
        res = {st: evaluate_branch(rec, task, st, "modelo único") for st in STRATEGIES}
        per_fold = pd.concat([r["per_fold"] for r in res.values()], ignore_index=True)
        per_fold.to_csv(out / f"{task.name}_per_fold.csv", index=False)
        summary = summarize(per_fold, MURMUR_METRICS if task is MURMUR else PULM_METRICS)
        summary.to_csv(out / f"{task.name}_summary.csv", index=False)
        oof = res[args.strategy]["oof"]
        oof.to_csv(out / f"{task.name}_recording_oof.csv", index=False)
        results[task] = {"summary": summary, "oof": oof}
    oof_m, oof_p = results[MURMUR]["oof"], results[PULMONARY]["oof"]

    # ── Dominio y capa 1 ─────────────────────────────────────────────────────
    dom = domain_by_recording(w)
    dom_test = dom[dom.role == "test"]
    gate = pd.read_csv(INTEGRATION_PROBS_DIR / "gate.csv", usecols=["file_id", "split"])
    gate_test_files = set(gate.loc[gate.split == "test", "file_id"])
    l1 = pd.DataFrame({
        "test del gate (161 pacientes)": layer1(dom_test, oof_m, oof_p, gate_test_files,
                                                "Modelo único (capa 1, test del gate)", out, "gate_test"),
        "todos los pacientes": layer1(dom_test, oof_m, oof_p, None,
                                      "Modelo único (capa 1, todos los pacientes)", out, "all"),
    })
    ref = pd.read_csv(INTEGRATION_REPORTS_DIR / "layer1_system.csv", index_col=0)["valor"]
    l1["pipeline con gate (test del gate)"] = pd.Series({
        "grabaciones": ref["grabaciones"], "pacientes": ref["pacientes"],
        "accuracy_dominio": ref["gate_accuracy_grabacion_media"],
        "errores_cardiaco_a_pulmonar": ref["gate_errores_cardiaco_a_pulmonar"],
        "errores_pulmonar_a_cardiaco": ref["gate_errores_pulmonar_a_cardiaco"],
        "accuracy_6clases_sistema": ref["accuracy_6clases_sistema"],
        "accuracy_6clases_dominio_perfecto": ref["accuracy_6clases_ruteo_perfecto"]})
    l1.to_csv(out / "layer1_system.csv")

    # ── Consola ──────────────────────────────────────────────────────────────
    print(f"Run {args.run_name} | estrategia para OOF y capa 1: {args.strategy}")
    print(f"\nUmbrales por ventana (val de cada fold):\n{wthr_df.round(3).to_string(index=False)}")
    for task, key in [(MURMUR, ["recall", "precision", "f2", "auc"]),
                      (PULMONARY, ["se", "sp", "icbhi_score", "crackle_auc", "wheeze_auc"])]:
        s = results[task]["summary"]
        both = pd.concat([s[s.rule == "auto"], reference_rows(task).assign(setting="pipeline con gate")])
        print(f"\n== Capa 2 — {task.name} por grabación, regla elegida (media ± desvío en 5 folds) ==")
        print(both[["setting", "strategy", "val_score"] + key + ["chosen"]].to_string(index=False))
    print(f"\n== Dominio por grabación (todas, rol test): accuracy {dom_test['domain_ok'].mean():.4f}")
    print(f"\n== Capa 1 ==\n{l1.round(4).to_string()}")
    print(f"\n✓ Reportes en: {out}")


if __name__ == "__main__":
    main()
