"""
unified_model/import_colab_run.py
====================================
Trae una corrida de run_cnn6_gpu.py (Colab) al repositorio, en los lugares
y con los nombres de run_kfold, para evaluarla con las herramientas de
siempre (filter_variants.py, evaluate_circor_spr.py, plot_circor_spr.py).

- CSV y JSON -> reports/unified_model/runs/<run>/
- Checkpoints -> unified_model/checkpoints_<run>/ (no versionados)
- full_spr_fold{i}[_select].csv -> las caches de
  evaluate_full_recordings.infer en reports/sprsound/full_recording/:
  <run>_SPRSound_windows.csv y <run>_select_SPRSound_windows.csv con los 5
  folds, y <run>_folds03_SPRSound_windows.csv con los folds del filtro
  (0 y 3). infer lee la cache y no necesita reconstruir el modelo (la CNN6
  no la arma run_kfold.build_model).

Uso (src = la carpeta de la corrida bajada de Drive):
    python unified_model/import_colab_run.py --src <carpeta>/cnn6_panns32k
"""

import argparse
import shutil

import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import UNIFIED_MODEL_REPORTS_DIR, UNIFIED_MODEL_SRC_DIR
from unified_model.evaluate_full_recordings import OUT_DIR as FULL_DIR
from unified_model.filter_variants import FOLDS as FILTER_FOLDS
from unified_model.run_kfold import K


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--run-name", help="Por defecto, el nombre de la carpeta")
    args = parser.parse_args()
    run = args.run_name or args.src.name
    run_dir, ckpt_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / run, UNIFIED_MODEL_SRC_DIR / f"checkpoints_{run}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    FULL_DIR.mkdir(parents=True, exist_ok=True)

    for f in sorted(args.src.iterdir()):
        if f.name.startswith("full_spr_") or f.name.endswith("_last.pt"):
            continue
        if f.suffix in (".csv", ".json"):
            shutil.copy2(f, run_dir / f.name)
        elif f.suffix == ".pt":
            shutil.copy2(f, ckpt_dir / f.name)

    done = sorted(int(p.stem.removeprefix("result_fold")) for p in args.src.glob("result_fold*.json"))
    print(f"{run}: folds terminados {done} -> {run_dir}")
    caches = []
    if done == list(range(K)):
        for suffix, tag in (("", run), ("_select", f"{run}_select")):
            caches.append((tag, [args.src / f"full_spr_fold{i}{suffix}.csv" for i in range(K)]))
    if set(FILTER_FOLDS) <= set(done):
        tag = f"{run}_folds" + "".join(map(str, FILTER_FOLDS))
        caches.append((tag, [args.src / f"full_spr_fold{i}.csv" for i in FILTER_FOLDS]))
    for tag, paths in caches:
        path = FULL_DIR / f"{tag}_SPRSound_windows.csv"
        pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).to_csv(path, index=False)
        print(f"  cache {path.name}")
    if not caches:
        print("  sin caches: faltan los folds 0 y 3 o los 5 folds")


if __name__ == "__main__":
    main()
