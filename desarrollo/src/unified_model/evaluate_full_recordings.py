"""
unified_model/evaluate_full_recordings.py
============================================
AUC normal vs anormal por grabación con TODAS las ventanas de cada grabación
(avances/23), como en uso real: el modelo no sabe dónde están las
anotaciones.

En evaluate_sprsound.py el puntaje de la grabación promedia solo las
ventanas del índice. En SPRSound el índice deja afuera las ventanas sin
eventos y, en las grabaciones anormales, las normales (sus huecos tienen
anomalías sin marcar), así que el puntaje de una grabación anormal se
calcula solo sobre sus ventanas anormales: el AUC queda inflado. En ICBHI
los ciclos cubren casi toda la grabación y la diferencia es chica, pero se
recalcula igual para comparar con la misma vara.

Para cada corrida y fold, pasa por el checkpoint del fold todas las
grabaciones de los pacientes de val y test de ese fold, cortadas con
integration.window_inference.fixed_window_starts (ventana 4 s, paso 2 s,
la última alineada al final). Las grabaciones de menos de 4 s (unas pocas
de SPRSound) se rellenan con ceros hasta 4 s. El puntaje es el de
evaluate_sprsound.recording_scores: media de 1 − P(pulm_normal).

Tests, para cada corrida (las de reports/sprsound/runs y --ref, abl_pulm,
de reports/unified_model/runs): SPRSound (pacientes de test de su fold),
ICBHI (pacientes de test del fold unificado) y "conjunto" (los dos juntos,
como una sola base). Para abl_pulm, SPRSound es el test cruzado inverso.

Salida: reports/sprsound/full_recording/
    <run>_<test>_windows.csv   probabilidades por ventana (cache: si existe no se recalcula)
    summary.csv                AUC por fold y pooled, junto al del índice (evaluate_sprsound)

Uso:
    python unified_model/evaluate_full_recordings.py [--runs spr_inv spr_sqrt] [--ref abl_pulm]
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV, SPRSOUND_KFOLD_DIR,
                          SPRSOUND_RECORDINGS_CSV, SPRSOUND_REPORTS_DIR, UNIFIED_AUDIO_CACHE_NPY,
                          UNIFIED_AUDIO_OFFSETS_CSV, UNIFIED_KFOLD_DIR, UNIFIED_MODEL_REPORTS_DIR, UNIFIED_MODEL_SRC_DIR, unified_windows_index)
from integration.evaluate_integration import load_recordings
from integration.window_inference import SR, fixed_window_starts
from unified_model.dataset import derived_probs, open_audio_cache
from unified_model.evaluate_sprsound import auc, recording_scores
from unified_model.run_kfold import K, build_model, predict

WIN_SEC = 4.0
OUT_DIR = SPRSOUND_REPORTS_DIR / "full_recording"


class FullRecordingDataset(Dataset):
    """Todas las ventanas fijas de las grabaciones dadas (file_id ->
    offset/n_samples en la cache), rellenadas con ceros si son más cortas."""

    def __init__(self, recs: pd.DataFrame, audio: np.ndarray, offsets: pd.DataFrame):
        win, hop = int(WIN_SEC * SR), int(WIN_SEC * SR / 2)
        self.audio, self.win = audio, win
        rows = []
        for r in recs.itertuples():
            off, n = offsets.loc[r.file_id, ["offset", "n_samples"]]
            for w, s in enumerate(fixed_window_starts(int(n), win, hop)):
                rows.append({"file_id": r.file_id, "subject_id": r.subject_id, "window_idx": w,
                             "start_sec": s / SR, "start": int(off) + s, "length": min(win, int(n) - s)})
        self.df = pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        s, n = self.df.at[idx, "start"], self.df.at[idx, "length"]
        y = np.zeros(self.win, dtype=np.float32)
        y[:n] = self.audio[s:s + n].astype(np.float32) / 32768.0
        return torch.from_numpy(y), 0


def collate(batch):
    ys, labels = zip(*batch)
    return torch.stack(ys), torch.tensor(labels)


def test_sets() -> dict[str, tuple]:
    """test -> (grabaciones file_id/subject_id, etiquetas, audio, offsets, carpeta de folds)."""
    spr = pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    icb = (pd.read_csv(unified_windows_index(WIN_SEC)).query("domain == 'pulmonary'")
           [["file_id", "subject_id"]].drop_duplicates())
    out = {}
    for name, recs, labels, (cache_npy, offsets_csv), fold_dir in (
            ("SPRSound", spr[["file_id", "subject_id"]], spr,
             (SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV), SPRSOUND_KFOLD_DIR),
            ("ICBHI", icb, load_recordings(), (UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV), UNIFIED_KFOLD_DIR)):
        audio, _ = open_audio_cache(cache_npy, offsets_csv)
        offsets = pd.read_csv(offsets_csv).set_index("file_id")
        out[name] = (recs, labels, audio, offsets, fold_dir)
    return out


def infer(run_dir: Path, run: str, test: str, sets: dict) -> pd.DataFrame:
    path = OUT_DIR / f"{run}_{test}_windows.csv"
    if path.exists():
        return pd.read_csv(path)
    args = argparse.Namespace(**json.loads((run_dir / "config.json").read_text()))
    recs, _, audio, offsets, fold_dir = sets[test]
    parts = []
    for i in range(K):
        model = build_model(args)
        model.load_state_dict(torch.load(UNIFIED_MODEL_SRC_DIR / f"checkpoints_{run}" / f"fold{i}_best.pt",
                                         map_location="cpu"))
        fold = pd.read_csv(fold_dir / f"fold{i}.csv")
        for role in ("val", "test"):
            subj = set(fold.loc[fold["split"] == role, "subject_id"])
            ds = FullRecordingDataset(recs[recs["subject_id"].isin(subj)], audio, offsets)
            probs = predict(model, DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate))
            part = ds.df[["file_id", "subject_id", "window_idx", "start_sec"]].assign(fold=i, role=role)
            for k in ("p_pulm_normal", "p_crackle", "p_wheeze"):
                part[k] = derived_probs(probs)[k]
            parts.append(part)
        print(f"  {run} → {test}: fold {i} ({sum(len(p) for p in parts)} ventanas)", flush=True)
    w = pd.concat(parts, ignore_index=True)
    w.to_csv(path, index=False)
    return w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=["spr_inv", "spr_sqrt"])
    parser.add_argument("--ref", default="abl_pulm")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sets = test_sets()

    runs = {args.ref: UNIFIED_MODEL_REPORTS_DIR / "runs" / args.ref}
    runs.update({r: SPRSOUND_REPORTS_DIR / "runs" / r for r in args.runs})
    both_labels = pd.concat([sets[t][1][["file_id", "has_crackles", "has_wheezes"]] for t in ("SPRSound", "ICBHI")])
    index = pd.read_csv(SPRSOUND_REPORTS_DIR / "eval" / "summary.csv").set_index(["model", "test"])
    rows = []
    for run, run_dir in runs.items():
        ws = {t: infer(run_dir, run, t, sets) for t in ("SPRSound", "ICBHI")}
        ws["conjunto"] = pd.concat(ws.values(), ignore_index=True)
        for test, w in ws.items():
            labels = both_labels if test == "conjunto" else sets[test][1]
            sc = recording_scores(w[w.role == "test"], labels)
            per_fold = [auc(g["abnormal"], g["score"]) for _, g in sc.groupby("fold")]
            key = (run, test)
            rows.append({"model": run, "test": test, "n_recordings": len(sc),
                         "n_windows": int((w.role == "test").sum()),
                         "auc_full_pooled": auc(sc["abnormal"], sc["score"]),
                         "auc_full_per_fold": f"{np.mean(per_fold):.3f} ± {np.std(per_fold, ddof=1):.3f}",
                         "auc_index_pooled": index.at[key, "auc_normal_vs_abnormal_pooled"] if key in index.index else np.nan,
                         "auc_index_per_fold": index.at[key, "auc_normal_vs_abnormal"] if key in index.index else ""})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(3).to_string(index=False))
    print(f"\n✓ {OUT_DIR}")


if __name__ == "__main__":
    main()
