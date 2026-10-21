"""
pulmonary_adventitious_model/export_val_probs.py
===================================================
Reconstruye las probabilidades de VALIDACIÓN (por fold) de modelos ya
entrenados que no las guardaron, a partir de sus checkpoints — insumo del
estudio de calibración de umbrales (calibration_study.py, avances/17).

Solo inferencia, sin reentrenar:
- "v4_hp100": CNN de 3 bloques (checkpoints de run_kfold.py), sobre los WAV
  de los ciclos con pasa-altos.
- corridas de run_finetune_gpu.py ("v8_cnn6_ft_*"): CNN6 ajustada, sobre el
  log-Mel del paquete de Colab (se descomprime en data/ si hace falta). Los
  checkpoints están en la misma carpeta de reportes.

Escribe val_probs_fold{i}.csv en la carpeta de reportes de cada corrida, con
el mismo formato que guarda run_finetune_gpu.py. Como control, recalcula
también las probabilidades de test y las compara con oof_predictions.csv
(deben coincidir salvo redondeo: en Colab se usó precisión mixta).

Uso:
    python pulmonary_adventitious_model/export_val_probs.py --runs v4_hp100 v8_cnn6_ft_A
"""

import argparse
import zipfile

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    DATA_DIR, PULMONARY_ADV_CHECKPOINTS_DIR, PULMONARY_ADV_REPORTS_DIR, PULMONARY_KFOLD_DIR,
    pulmonary_cycles_paths,
)
from pulmonary_adventitious_model.dataset import CycleDataset
from pulmonary_adventitious_model.model import AdventitiousCNN
from pulmonary_adventitious_model.pack_for_colab import PACKAGE_ZIP
from pulmonary_adventitious_model.panns_cnn6 import load_pretrained_cnn6
from pulmonary_adventitious_model.run_finetune_gpu import FineTuneCnn6, LogmelBank, predict as predict_ft
from pulmonary_adventitious_model.run_kfold import K, TARGET_SEC
from pulmonary_adventitious_model.train import BATCH_SIZE, DROPOUT, predict_probs

PACKAGE_DIR = DATA_DIR / "colab_pulmonary_package"


def _ensure_package() -> Path:
    if not (PACKAGE_DIR / "logmel_panns32k.npy").exists():
        print(f"Descomprimiendo {PACKAGE_ZIP.name} en {PACKAGE_DIR} ...")
        zipfile.ZipFile(PACKAGE_ZIP).extractall(PACKAGE_DIR)
    return PACKAGE_DIR


def export(run: str) -> None:
    reports_dir = PULMONARY_ADV_REPORTS_DIR / run
    oof = pd.read_csv(reports_dir / "oof_predictions.csv")
    torch.set_flush_denormal(True)

    if run.startswith("v8_cnn6_ft_"):
        pkg = _ensure_package()
        cycles = pd.read_csv(pkg / "cycles_index.csv")
        bank = LogmelBank(np.load(pkg / "logmel_panns32k.npy"), cycles["logmel_start"].to_numpy(),
                          cycles["logmel_length"].to_numpy(), torch.device("cpu"))

        def probs(i, idx):
            model = FineTuneCnn6(load_pretrained_cnn6(pkg / "Cnn6_mAP=0.343.pth"), first_trainable=0)
            model.load_state_dict(torch.load(reports_dir / f"fold{i}_best.pt", map_location="cpu"))
            return predict_ft(model, bank, idx, amp=False)
    elif run == "v4_hp100":
        cycles_dir, index_csv = pulmonary_cycles_paths("hp100")
        cycles = pd.read_csv(index_csv)
        cache: dict = {}

        def probs(i, idx):
            model = AdventitiousCNN(dropout=DROPOUT, pooling="time_max")
            model.load_state_dict(torch.load(PULMONARY_ADV_CHECKPOINTS_DIR / run / f"fold{i}_best.pt",
                                             map_location="cpu"))
            ds = CycleDataset(cycles.iloc[idx], cycles_dir, target_sec=TARGET_SEC, feature_type="logmel",
                              augment=False, wav_cache=cache)
            return predict_probs(model, DataLoader(ds, batch_size=BATCH_SIZE), "cpu")[1]
    else:
        raise ValueError(f"Corrida no soportada: {run}")

    for i in range(K):
        split = cycles[["subject_id"]].merge(
            pd.read_csv(PULMONARY_KFOLD_DIR / f"fold{i}.csv"), on="subject_id", how="left")["split"].to_numpy()
        idx_va, idx_te = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
        p_va, p_te = probs(i, idx_va), probs(i, idx_te)

        part = cycles.iloc[idx_va][["cycle_id", "subject_id", "crackle", "wheeze"]].copy()
        part["fold"] = i
        part["p_crackle"], part["p_wheeze"] = p_va[:, 0], p_va[:, 1]
        part.to_csv(reports_dir / f"val_probs_fold{i}.csv", index=False)

        ref = oof[oof.fold == i].set_index("cycle_id").loc[cycles.iloc[idx_te]["cycle_id"], ["p_crackle", "p_wheeze"]]
        diff = np.abs(ref.to_numpy() - p_te).max()
        print(f"  {run} fold {i}: {len(idx_va)} ciclos de val exportados | control test: "
              f"diferencia máx. con oof_predictions = {diff:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True)
    for run in parser.parse_args().runs:
        export(run)
