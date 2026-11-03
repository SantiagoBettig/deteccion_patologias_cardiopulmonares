"""
unified_model/evaluate_circor_spr.py
=======================================
Evaluación por grabación del modelo de 6 clases CirCor + SPRSound
(avances/24, run_kfold.py --data circor_spr), frente a las ablaciones por
dominio con la misma receta: abl_card (solo CirCor) y spr_inv (solo
SPRSound). Mismo procedimiento que evaluate_unified (capa 2 con reglas de
agregación y estrategias de calibración de evaluate_integration) con TODAS
las ventanas de cada grabación:

- CirCor: el índice ya tiene todas las ventanas de cada grabación, así que
  se usan las probabilidades que guarda run_kfold (window_probs_fold*.csv).
- SPRSound: el índice deja afuera ventanas (avances/23 §4.1), así que las
  grabaciones completas de los pacientes de val y test de cada fold pasan
  por el checkpoint del fold (evaluate_full_recordings.infer, con cache).

Los umbrales por ventana (para la regla "frac") se calibran en las ventanas
de val del índice, como en evaluate_unified.

Para cada modelo:
1. Capa 2: murmullo por grabación (CirCor) y crackle/wheeze por grabación
   (SPRSound, score ICBHI de 4 clases).
2. AUC normal contra anormal por grabación en SPRSound (media de
   1 − P(pulm_normal)), como en avances/23.
3. Solo el modelo de 6 clases: dominio por grabación (P(cardíaco) media >
   0.5) y capa 1 sobre todos los pacientes (accuracy de 6 clases del
   sistema). El test del gate de avances/22 no aplica (era de ICBHI).

El modelo de 6 clases se evalúa con los dos criterios de checkpoint: B
(select_domain, el principal) y A (select, sensibilidad).

Con --gate-run (avances/25), además el pipeline separado: dominio por
grabación del gate k-fold (gate_model/run_kfold_circor_spr.py, mismos
folds; P(cardíaco) media de las ventanas > 0.5, la regla del gate v5) y las
ramas de las ablaciones con sus predicciones OOF de capa 2. La capa 1 se
compara con la del modelo de 6 clases sobre las mismas grabaciones, con la
diferencia de accuracy y un intervalo bootstrap por paciente (las
grabaciones de un paciente no son independientes).

Salida: reports/unified_model/circor_spr/

Uso:
    python unified_model/evaluate_circor_spr.py [--run cs_base] [--card-ref abl_card] [--pulm-ref spr_inv] [--gate-run kfold_cs] [--base-run cs_base]
"""

import argparse

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_REPORTS_DIR, SPRSOUND_RECORDINGS_CSV, SPRSOUND_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import (MURMUR, MURMUR_METRICS, PULM_METRICS, PULMONARY, STRATEGIES,
                                              aggregate, evaluate_branch, load_recordings, summarize)
from unified_model.evaluate_full_recordings import infer, test_sets
from unified_model.evaluate_sprsound import auc, recording_scores
from unified_model.evaluate_unified import domain_by_recording, layer1, load_window_probs, window_thresholds

OUT_DIR = UNIFIED_MODEL_REPORTS_DIR / "circor_spr"
FULL_COLS = ("p_pulm_normal", "p_crackle", "p_wheeze", "p_cardiac", "p_murmur")
N_BOOT = 2000
SEED = 42


def layer2(task, windows: pd.DataFrame, cols: list, wthr: dict, labels: pd.DataFrame, setting: str,
           out: Path, strategy: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec = aggregate(windows, cols, wthr).merge(labels[["file_id"] + task.labels], on="file_id")
    res = {st: evaluate_branch(rec, task, st, setting) for st in STRATEGIES}
    per_fold = pd.concat([r["per_fold"] for r in res.values()], ignore_index=True)
    per_fold.to_csv(out / f"{task.name}_per_fold.csv", index=False)
    summary = summarize(per_fold, MURMUR_METRICS if task is MURMUR else PULM_METRICS)
    summary.to_csv(out / f"{task.name}_summary.csv", index=False)
    oof = res[strategy]["oof"]
    oof.to_csv(out / f"{task.name}_recording_oof.csv", index=False)
    return summary[(summary.rule == "auto") & (summary.strategy == strategy)], oof


def evaluate_model(setting: str, run: str, run_dir: Path, criterion: str | None, sets: dict,
                   recs: pd.DataFrame, spr: pd.DataFrame, strategy: str) -> dict:
    out = OUT_DIR / setting.replace(" ", "_").replace("(", "").replace(")", "")
    out.mkdir(parents=True, exist_ok=True)
    w = load_window_probs(run_dir, criterion)
    wthr, wthr_df = window_thresholds(w)
    wthr_df.dropna(axis=1, how="all").to_csv(out / "window_thresholds.csv", index=False)
    domains = set(w["domain"])
    res = {"setting": setting}

    card = w[(w.domain == "cardiac") & w.role.isin(["val", "test"])]
    if "cardiac" in domains:
        res["murmur"], res["oof_m"] = layer2(MURMUR, card, ["prob"], wthr, recs, setting, out, strategy)
    if "pulmonary" in domains:
        cols = FULL_COLS if "cardiac" in domains else FULL_COLS[:3]
        pulm = infer(run_dir, run, "SPRSound", sets, criterion, cols).assign(domain="pulmonary")
        res["pulmonary"], res["oof_p"] = layer2(PULMONARY, pulm, ["p_crackle", "p_wheeze"], wthr, spr,
                                                setting, out, strategy)
        sc = recording_scores(pulm[pulm.role == "test"], spr)
        per_fold = [auc(g["abnormal"], g["score"]) for _, g in sc.groupby("fold")]
        res["rec_auc"] = {"setting": setting, "auc_normal_anormal_pooled": auc(sc["abnormal"], sc["score"]),
                          "auc_normal_anormal_por_fold": f"{np.mean(per_fold):.3f} ± {np.std(per_fold, ddof=1):.3f}"}
    if domains == {"cardiac", "pulmonary"}:
        both = pd.concat([card.rename(columns={"prob": "p_murmur"}), pulm], ignore_index=True)
        dom_test = domain_by_recording(both).query("role == 'test'")
        dom_test.to_csv(out / "domain_by_recording.csv", index=False)
        res["layer1"] = layer1(dom_test, res["oof_m"], res["oof_p"], None,
                               f"{setting} (capa 1, todos los pacientes)", out, "all")
    return res


def separate_pipeline(gate_run: str, oof_m: pd.DataFrame, oof_p: pd.DataFrame) -> dict:
    """Capa 1 del pipeline separado: dominio del gate k-fold + ramas de las ablaciones."""
    out = OUT_DIR / f"pipeline_{gate_run}"
    out.mkdir(parents=True, exist_ok=True)
    gate_dir = GATE_REPORTS_DIR / "runs" / gate_run
    w = pd.concat([pd.read_csv(p) for p in sorted(gate_dir.glob("window_probs_fold?.csv"))], ignore_index=True)
    dom_test = domain_by_recording(w.rename(columns={"prob_cardiac": "p_cardiac"})).query("role == 'test'")
    dom_test.to_csv(out / "domain_by_recording.csv", index=False)
    return layer1(dom_test, oof_m, oof_p, None, "Pipeline separado (capa 1, todos los pacientes)", out, "all")


def paired_difference(a_dir: Path, b_dir: Path) -> dict:
    """Accuracy de 6 clases del sistema, a − b, sobre las mismas grabaciones:
    diferencia, grabaciones discordantes e IC 95% bootstrap por paciente."""
    a = pd.read_csv(a_dir / "layer1_all_recordings.csv")
    b = pd.read_csv(b_dir / "layer1_all_recordings.csv")
    m = a.merge(b, on=["file_id", "subject_id"], suffixes=("_a", "_b"))
    assert len(m) == len(a) == len(b), "las capas 1 no cubren las mismas grabaciones"
    m["ok_a"] = m["system_class_a"] == m["true_class_a"]
    m["ok_b"] = m["system_class_b"] == m["true_class_b"]
    per_subj = m.groupby("subject_id")[["ok_a", "ok_b"]].agg(["sum", "count"])
    ok_a, ok_b, n = per_subj[("ok_a", "sum")].to_numpy(), per_subj[("ok_b", "sum")].to_numpy(), per_subj[("ok_a", "count")].to_numpy()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(n), size=(N_BOOT, len(n)))
    boot = (ok_a[idx].sum(1) - ok_b[idx].sum(1)) / n[idx].sum(1)
    return {"accuracy_a": m["ok_a"].mean(), "accuracy_b": m["ok_b"].mean(),
            "diferencia_a_menos_b": m["ok_a"].mean() - m["ok_b"].mean(),
            "ic95_bootstrap_paciente": f"[{np.percentile(boot, 2.5):.4f}, {np.percentile(boot, 97.5):.4f}]",
            "solo_a_acierta": int((m.ok_a & ~m.ok_b).sum()), "solo_b_acierta": int((~m.ok_a & m.ok_b).sum()),
            "grabaciones": len(m), "pacientes": len(n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="cs_base")
    parser.add_argument("--card-ref", default="abl_card")
    parser.add_argument("--pulm-ref", default="spr_inv")
    parser.add_argument("--strategy", default="pooled_fixed", choices=STRATEGIES)
    parser.add_argument("--base-run", help="corrida de 6 clases contra la que comparar --run con la diferencia apareada (avances/25)")
    parser.add_argument("--gate-run", help="gate k-fold (reports/gate/runs/<run>) para el pipeline separado")
    args = parser.parse_args()
    global OUT_DIR
    if args.base_run:   # no pisa las tablas de cs_base; base y ablaciones se reevalúan con los mismos cachés
        OUT_DIR = OUT_DIR / f"{args.run}_vs_{args.base_run}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    sets = test_sets()
    recs, spr = load_recordings(), pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    run_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / args.run
    models = [(f"{args.run} (B)", args.run, run_dir, None),
              (f"{args.run} (A)", args.run, run_dir, "select"),
              (args.card_ref, args.card_ref, UNIFIED_MODEL_REPORTS_DIR / "runs" / args.card_ref, None),
              (args.pulm_ref, args.pulm_ref, SPRSOUND_REPORTS_DIR / "runs" / args.pulm_ref, None)]
    if args.base_run:
        base_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / args.base_run
        models[2:2] = [(f"{args.base_run} (B)", args.base_run, base_dir, None)]
    results = [evaluate_model(*m, sets, recs, spr, args.strategy) for m in models]

    for task, key in [("murmur", ["recall", "precision", "f2", "auc"]),
                      ("pulmonary", ["se", "sp", "icbhi_score", "crackle_auc", "wheeze_auc"])]:
        t = pd.concat([r[task] for r in results if task in r])[["setting", "strategy", "val_score"] + key + ["chosen"]]
        t.to_csv(OUT_DIR / f"{task}_comparison.csv", index=False)
        print(f"\n== Capa 2 — {task} por grabación, regla elegida (media ± desvío en 5 folds) ==")
        print(t.to_string(index=False))
    rec_auc = pd.DataFrame([r["rec_auc"] for r in results if "rec_auc" in r])
    rec_auc.to_csv(OUT_DIR / "sprsound_recording_auc.csv", index=False)
    print(f"\n== SPRSound — AUC normal contra anormal por grabación ==\n{rec_auc.round(3).to_string(index=False)}")
    l1 = pd.DataFrame({r["setting"]: r["layer1"] for r in results if "layer1" in r})
    if args.gate_run:
        card, pulm = results[-2], results[-1]
        l1[f"pipeline separado ({args.gate_run} + {args.card_ref} + {args.pulm_ref})"] = separate_pipeline(
            args.gate_run, card["oof_m"], pulm["oof_p"])
    l1.to_csv(OUT_DIR / "layer1_system.csv")
    print(f"\n== Capa 1 — sistema de 6 clases, todos los pacientes ==\n{l1.round(4).to_string()}")
    if args.gate_run:
        diff = pd.Series(paired_difference(OUT_DIR / f"{args.run}_B", OUT_DIR / f"pipeline_{args.gate_run}"))
        diff.to_csv(OUT_DIR / "layer1_paired_difference.csv", header=["valor"])
        print(f"\n== Capa 1 — a: {args.run} (B), b: pipeline separado ==\n{diff.to_string()}")
    if args.base_run:
        diff = pd.Series(paired_difference(OUT_DIR / f"{args.run}_B", OUT_DIR / f"{args.base_run}_B"))
        diff.to_csv(OUT_DIR / "layer1_paired_difference.csv", header=["valor"])
        print(f"\n== Capa 1 — a: {args.run} (B), b: {args.base_run} (B) ==\n{diff.to_string()}")
    print(f"\n✓ Reportes en: {OUT_DIR}")


if __name__ == "__main__":
    main()
