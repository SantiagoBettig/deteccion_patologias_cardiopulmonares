"""
unified_model/evaluate_ablation.py
=====================================
Evaluación de la ablación por dominio (avances/22): la receta de la v1_base
entrenada solo con las ventanas cardíacas (--card-run) o solo con las
pulmonares (--pulm-run), con run_kfold.py --domain. No corre ningún modelo.

1. Capa 2 de cada rama, con el mismo procedimiento que evaluate_unified
   (umbrales por ventana calibrados en val, reglas de agregación,
   estrategias de calibración), comparada contra el modelo único
   (--unified-run) y contra la rama del pipeline actual (MurmurCNN / CNN6,
   reports/integration).

2. Capa 1 con el gate v5 (integration/evaluate_integration.layer1, test del
   gate): el pipeline separado con las ramas de la ablación — ambas, o
   cada una combinada con la rama actual de la otra tarea — frente al
   pipeline actual y al modelo único. Es la comparación entre metodologías
   a igual red: solo cambia integrado contra separado.

Salida: reports/unified_model/ablation/

Uso:
    python unified_model/evaluate_ablation.py [--card-run abl_card] [--pulm-run abl_pulm] [--unified-run v1_base]
"""

import argparse

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import INTEGRATION_REPORTS_DIR, UNIFIED_MODEL_REPORTS_DIR
from integration.evaluate_integration import (MURMUR, MURMUR_METRICS, PULM_METRICS, PULMONARY, STRATEGIES,
                                              aggregate, evaluate_branch, layer1, load_recordings, summarize)
from unified_model.evaluate_unified import load_window_probs, reference_rows, window_thresholds


def branch_layer2(run: str, task, domain: str, cols: list, recs: pd.DataFrame, out: Path, strategy: str):
    w = load_window_probs(UNIFIED_MODEL_REPORTS_DIR / "runs" / run)
    assert set(w["domain"]) == {domain}, f"{run} no es una ablación {domain}"
    wthr, wthr_df = window_thresholds(w)
    wthr_df.dropna(axis=1, how="all").to_csv(out / f"{task.name}_window_thresholds.csv", index=False)
    rec = aggregate(w, cols, wthr).merge(recs[["file_id"] + task.labels], on="file_id")
    res = {st: evaluate_branch(rec, task, st, f"ablación ({run})") for st in STRATEGIES}
    per_fold = pd.concat([r["per_fold"] for r in res.values()], ignore_index=True)
    per_fold.to_csv(out / f"{task.name}_per_fold.csv", index=False)
    summary = summarize(per_fold, MURMUR_METRICS if task is MURMUR else PULM_METRICS)
    summary.to_csv(out / f"{task.name}_summary.csv", index=False)
    oof = res[strategy]["oof"]
    oof.to_csv(out / f"{task.name}_recording_oof.csv", index=False)
    return summary, oof


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card-run", default="abl_card")
    parser.add_argument("--pulm-run", default="abl_pulm")
    parser.add_argument("--unified-run", default="v1_base")
    parser.add_argument("--strategy", default="pooled_fixed", choices=STRATEGIES)
    args = parser.parse_args()

    out = UNIFIED_MODEL_REPORTS_DIR / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    recs = load_recordings()
    unified_eval = UNIFIED_MODEL_REPORTS_DIR / "runs" / args.unified_run / "eval"

    # ── Capa 2 ───────────────────────────────────────────────────────────────
    oof, rows = {}, []
    for task, run, domain, cols, key in [
            (MURMUR, args.card_run, "cardiac", ["prob"], ["recall", "precision", "f2", "auc"]),
            (PULMONARY, args.pulm_run, "pulmonary", ["p_crackle", "p_wheeze"],
             ["se", "sp", "icbhi_score", "crackle_auc", "wheeze_auc"])]:
        summary, oof[task] = branch_layer2(run, task, domain, cols, recs, out, args.strategy)
        uni = pd.read_csv(unified_eval / f"{task.name}_summary.csv").assign(setting=f"modelo único ({args.unified_run})")
        ref = reference_rows(task).assign(setting="rama del pipeline actual")
        both = pd.concat([d[(d.rule == "auto") & (d.strategy == args.strategy)] for d in (summary, uni, ref)])
        both = both[["setting", "strategy", "val_score"] + key + ["chosen"]]
        both.to_csv(out / f"{task.name}_comparison.csv", index=False)
        rows.append((task, both))

    # ── Capa 1 con el gate v5 ────────────────────────────────────────────────
    current = {t: pd.read_csv(INTEGRATION_REPORTS_DIR / f"{t.name}_recording_oof.csv") for t in (MURMUR, PULMONARY)}
    combos = {
        "gate v5 + ramas de la ablación": (oof[MURMUR], oof[PULMONARY]),
        "gate v5 + murmullo ablación + pulmonar CNN6": (oof[MURMUR], current[PULMONARY]),
        "gate v5 + MurmurCNN + pulmonar ablación": (current[MURMUR], oof[PULMONARY]),
        "gate v5 + MurmurCNN + CNN6 (pipeline actual)": (current[MURMUR], current[PULMONARY]),
    }
    l1 = {}
    for k, (name, (om, op)) in enumerate(combos.items()):
        d = out / f"layer1_combo{k}"
        d.mkdir(exist_ok=True)
        r = layer1(om, op, d)
        b = r["branch"].set_index(["rama", "setting"])
        l1[name] = {"accuracy_6clases_sistema": r["system"]["accuracy_6clases_sistema"],
                    "accuracy_6clases_ruteo_perfecto": r["system"]["accuracy_6clases_ruteo_perfecto"],
                    "murmullo_f2_sistema": b.loc[("murmullo", "sistema (gate v5)"), "f2"],
                    "pulmonar_score_sistema": b.loc[("pulmonar", "sistema (gate v5)"), "icbhi_score"]}
    uni_l1 = pd.read_csv(unified_eval / "layer1_system.csv", index_col=0)["test del gate (161 pacientes)"]
    l1[f"modelo único ({args.unified_run})"] = {
        "accuracy_6clases_sistema": uni_l1["accuracy_6clases_sistema"],
        "accuracy_6clases_ruteo_perfecto": uni_l1["accuracy_6clases_dominio_perfecto"],
        "murmullo_f2_sistema": uni_l1["murmullo_f2_sistema"],
        "pulmonar_score_sistema": uni_l1["pulmonar_score_sistema"]}
    l1 = pd.DataFrame(l1).T
    l1.to_csv(out / "layer1_comparison.csv")

    # ── Consola ──────────────────────────────────────────────────────────────
    print(f"Ablación: cardíaca {args.card_run}, pulmonar {args.pulm_run} | estrategia {args.strategy}")
    for task, both in rows:
        print(f"\n== Capa 2 — {task.name} por grabación, regla elegida (media ± desvío en 5 folds) ==")
        print(both.to_string(index=False))
    print(f"\n== Capa 1 — test del gate (161 pacientes) ==\n{l1.round(4).to_string()}")
    print(f"\n✓ Reportes en: {out}")


if __name__ == "__main__":
    main()
