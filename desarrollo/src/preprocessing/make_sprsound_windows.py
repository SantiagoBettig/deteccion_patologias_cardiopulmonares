"""
preprocessing/make_sprsound_windows.py
=========================================
Índice de ventanas de SPRSound con la etiqueta pulmonar del modelo único
(clases 2-5), en el mismo formato que make_unified_windows.py, para entrenar
con unified_model/run_kfold.py --data sprsound.

Lee los JSON de data/raw/SPRSound/BioCAS20xx (los de Classification/ y
Detection/ tienen los mismos eventos, ver eda/explore_sprsound.py) y aplica
las decisiones del paso 2 (eda/sprsound_preprocessing_check.py):

- Fuera: grabaciones "Poor Quality" (no tienen eventos) y las 390 sin número
  de paciente en el nombre (no se pueden asignar a un fold sin fuga).
- Cada evento hace de "ciclo" en la regla de superposición de
  make_unified_windows.label_pulmonary (≥ --min-overlap s o ≥ 50% del
  evento). crackle = Fine/Coarse Crackle y Wheeze+Crackle; wheeze = Wheeze,
  Rhonchi (continuo, como en ICBHI) y Wheeze+Crackle.
- Las ventanas que tocan un Stridor con la misma regla se descartan.
- Las ventanas sin ningún evento se descartan (como en ICBHI), y también las
  de grabaciones de menos de --win-sec (el dataset apila largos fijos).
- Ventanas normales solo de grabaciones Normal: en las anormales se anotó
  solo una parte de las respiraciones y los huecos tienen anomalías sin
  marcar. La etiqueta de grabación de 2024-2025 (no viene en el JSON) es la
  que implican los eventos, que coincide al 100% en los años que la tienen.

El largo en muestras es el de audio_ops.preprocess a 4 kHz (8 kHz nativos ->
la mitad), así que las ventanas coinciden con el cache de
make_unified_audio_cache.py --data sprsound.

Salidas:
    data/sprsound/recordings.csv      una fila por grabación usada (con has_crackles/has_wheezes)
    data/sprsound/windows_index_4s.csv

Uso:
    python preprocessing/make_sprsound_windows.py [--win-sec 4] [--min-overlap 1.0]
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import SPRSOUND_DIR, SPRSOUND_RAW_DIR, SPRSOUND_RECORDINGS_CSV, sprsound_windows_index
from integration.window_inference import SR, fixed_window_starts
from preprocessing.make_unified_windows import CLASSES, label_pulmonary

SPLITS = {   # carpeta de JSON -> carpeta de WAV
    "train2022": ("BioCAS2022/train2022_json", "BioCAS2022/train2022_wav"),
    "test2022_inter": ("BioCAS2022/test2022_json/inter_test_json", "BioCAS2022/test2022_wav"),
    "test2022_intra": ("BioCAS2022/test2022_json/intra_test_json", "BioCAS2022/test2022_wav"),
    "test2023": ("BioCAS2023/test2023_json", "BioCAS2023/test2023_wav"),
    "test2024": ("BioCAS2024/test2024_json", "BioCAS2024/test2024_wav"),
    "test2025": ("BioCAS2025/test2025_json", "BioCAS2025/test2025_wav"),
}
NAME_RE = re.compile(r"^(?P<patient>\d*)_(?P<age>[\d.]+)_(?P<gender>[01])_(?P<loc>p\d)_(?P<rec>\d+)$")
CRACKLE_TYPES = {"Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"}
WHEEZE_TYPES = {"Wheeze", "Rhonchi", "Wheeze+Crackle"}
FLAGS = {"crackle": CRACKLE_TYPES, "wheeze": WHEEZE_TYPES, "stridor": {"Stridor"}}


def load_split(split: str) -> tuple[list[dict], list[dict]]:
    jdir, wdir = SPLITS[split]
    recs, evs = [], []
    for jp in sorted((SPRSOUND_RAW_DIR / jdir).glob("*.json")):
        m = NAME_RE.match(jp.stem)
        d = json.loads(jp.read_text(encoding="utf-8"))
        events = [(int(e["start"]) / 1000, int(e["end"]) / 1000, e["type"]) for e in d.get("event_annotation", [])]
        types = {t for *_, t in events}
        has_c, has_w = bool(types & CRACKLE_TYPES), bool(types & (WHEEZE_TYPES | {"Stridor"}))
        implied = {(0, 0): "Normal", (1, 0): "DAS", (0, 1): "CAS", (1, 1): "CAS & DAS"}[(has_c, has_w)]
        record = d.get("record_annotation") or implied
        wav = Path(wdir) / f"{jp.stem}.wav"
        recs.append({"file_id": f"{jp.stem}.wav", "audio_path": wav.as_posix(),
                     "subject_id": f"SPR_{m['patient']}" if m["patient"] else "",
                     "age_years": float(m["age"]), "sex": "M" if m["gender"] == "0" else "F",
                     "auscultation_location": m["loc"], "release": split, "record_annotation": record,
                     "has_crackles": int(bool(types & CRACKLE_TYPES)), "has_wheezes": int(bool(types & WHEEZE_TYPES)),
                     "n_frames": sf.info(str(SPRSOUND_RAW_DIR / wav)).frames, "n_events": len(events)})
        evs += [{"file_id": f"{jp.stem}.wav", "start_sec": s, "duration_sec": e - s, "type": t}
                for s, e, t in events]
    return recs, evs


def windows_for(recs: pd.DataFrame, evs: pd.DataFrame, win: int, hop: int) -> pd.DataFrame:
    """Superposición máxima de cada ventana con eventos de cada tipo (mismas
    columnas que make_unified_windows.pulmonary_windows)."""
    by_file = {f: g for f, g in evs.groupby("file_id")}
    rows = []
    for _, r in recs.iterrows():
        n = int(np.ceil(r["n_frames"] * SR / 8000))   # largo tras preprocess (8 kHz -> 4 kHz)
        c = by_file[r["file_id"]]
        cs, dur = c["start_sec"].to_numpy(), c["duration_sec"].to_numpy()
        ce = cs + dur
        masks = {k: c["type"].isin(v).to_numpy() for k, v in FLAGS.items()}
        for w, s in enumerate(fixed_window_starts(n, win, hop)):
            a, b = s / SR, (s + min(win, n - s)) / SR
            ov = np.clip(np.minimum(b, ce) - np.maximum(a, cs), 0, None)
            row = {"file_id": r["file_id"], "subject_id": r["subject_id"], "domain": "pulmonary",
                   "window_idx": w, "start_sample": s, "n_samples": min(win, n - s),
                   "annotated_sec": float(ov.sum())}
            for col, m in masks.items():
                row[f"{col}_ov_sec"] = float(ov[m].max()) if m.any() else 0.0
                row[f"{col}_ov_frac"] = float((ov[m] / dur[m]).max()) if m.any() else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-sec", type=float, default=4.0)
    parser.add_argument("--min-overlap", type=float, default=1.0)
    args = parser.parse_args()
    win, hop = int(args.win_sec * SR), int(args.win_sec * SR / 2)

    recs, evs = [], []
    for split in SPLITS:
        r, e = load_split(split)
        recs += r
        evs += e
    recs, evs = pd.DataFrame(recs), pd.DataFrame(evs)
    print(f"Grabaciones: {len(recs)}")
    drop_q = recs["record_annotation"] == "Poor Quality"
    drop_id = recs["subject_id"] == ""
    print(f"  fuera: Poor Quality {drop_q.sum()}, sin número de paciente {(drop_id & ~drop_q).sum()}")
    recs = recs[~drop_q & ~drop_id].reset_index(drop=True)
    evs = evs[evs["file_id"].isin(recs["file_id"])]

    w = windows_for(recs, evs, win, hop)
    steps = {"ventanas": len(w)}
    # unified_model/dataset.py apila ventanas de largo fijo (en ICBHI no hay
    # grabaciones de menos de 4 s; en SPRSound unas pocas, desde 0.3 s)
    short = w["n_samples"] < win
    steps["de grabaciones más cortas que la ventana (fuera)"] = int(short.sum())
    w = w[~short]
    n = len(w)
    w = w[w["annotated_sec"] > 0].copy()
    steps["sin ningún evento (fuera)"] = n - len(w)
    w["label"], w["borderline"] = label_pulmonary(w, args.min_overlap)
    stridor = (w["stridor_ov_sec"] >= args.min_overlap) | (w["stridor_ov_frac"] >= 0.5)
    steps["tocan estridor (fuera)"] = int(stridor.sum())
    w = w[~stridor]
    abnormal_rec = w["file_id"].map(recs.set_index("file_id")["record_annotation"]) != "Normal"
    normal_in_abn = (w["label"] == 2) & abnormal_rec
    steps["normales de grabaciones anormales (fuera)"] = int(normal_in_abn.sum())
    w = w[~normal_in_abn].copy()
    steps["quedan"] = len(w)
    print("\n".join(f"  {k}: {v}" for k, v in steps.items()))

    w["label_name"] = w["label"].map(lambda c: CLASSES[c])
    w["start_sec"] = w["start_sample"] / SR
    w = w.drop(columns=[c for c in w.columns if c.startswith("stridor_")]).reset_index(drop=True)
    recs = recs[recs["file_id"].isin(w["file_id"])].reset_index(drop=True)

    SPRSOUND_DIR.mkdir(parents=True, exist_ok=True)
    recs.to_csv(SPRSOUND_RECORDINGS_CSV, index=False)
    path = sprsound_windows_index(args.win_sec)
    w.to_csv(path, index=False)
    summ = w.groupby("label_name").agg(ventanas=("label", "size"), pacientes=("subject_id", "nunique"))
    print(f"\n{summ.reindex(CLASSES[2:]).to_string()}")
    print(f"\n✓ {SPRSOUND_RECORDINGS_CSV} — {len(recs)} grabaciones, {recs['subject_id'].nunique()} pacientes")
    print(f"✓ {path} — {len(w)} ventanas")


if __name__ == "__main__":
    main()
