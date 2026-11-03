"""
unified_model/pack_cnn6_colab.py
===================================
Empaqueta en UN solo .zip lo que necesita el fine-tuning de la CNN6 de 6
clases CirCor + SPRSound en Google Colab (run_cnn6_gpu.py, notebook
colab_cnn6_circor_spr.ipynb), avances/25 §8 paso 2. Mismo motivo que
pulmonary_adventitious_model/pack_for_colab.py: un solo archivo que Colab
copia una vez al disco local, en lugar de leer miles de archivos de Drive.

A diferencia del paquete pulmonar, se empaqueta el AUDIO (4 kHz, int16, el
mismo de las caches de run_kfold --data circor_spr) y no los log-Mel: el
espectrograma se calcula en la GPU, así se pueden probar los dos frontends
(panns32k y ours64) con el mismo paquete y mantener la augmentation de
cs_wide sobre la onda. Pesa menos que los log-Mel (8 KB/s contra 12.8 KB/s).

Contenido del zip:
    audio_4k_int16.npy     grabaciones cardíacas de CirCor (las del índice
                           unificado) + todas las de SPRSound, concatenadas
    recordings.csv         file_id, subject_id, domain, offset, n_samples
    windows.csv            ventanas del índice (las de run_kfold): file_id,
                           subject_id, domain, window_idx, start_sec,
                           start_sample, label
    full_windows_spr.csv   todas las ventanas de cada grabación de SPRSound,
                           como evaluate_full_recordings.FullRecordingDataset
                           (start_sample dentro de la grabación, length < 4 s
                           si es más corta: se rellena con ceros)
    folds/fold{0..4}.csv   subject_id, split: los folds unificados para
                           CirCor y los de SPRSound, juntos
    Cnn6_mAP=0.343.pth     pesos preentrenados de PANNs (CC-BY-4.0)
    manifest.json          conteos y controles

Uso:
    python unified_model/pack_cnn6_colab.py
    -> data/colab_cnn6_circor_spr.zip (subir ese archivo a Google Drive)
"""

import json
import shutil
import time
import zipfile

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (DATA_DIR, PANNS_CNN6_CKPT, SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV,
                          SPRSOUND_KFOLD_DIR, SPRSOUND_RECORDINGS_CSV, UNIFIED_AUDIO_OFFSETS_CSV, UNIFIED_KFOLD_DIR,
                          sprsound_windows_index, unified_windows_index)
from unified_model.dataset import open_audio_cache
from unified_model.evaluate_full_recordings import WIN_SEC, FullRecordingDataset
from unified_model.run_kfold import K

PACKAGE_ZIP = DATA_DIR / "colab_cnn6_circor_spr.zip"
WINDOW_COLS = ["file_id", "subject_id", "domain", "window_idx", "start_sec", "start_sample", "label"]


def pack() -> None:
    t0 = time.perf_counter()
    card_w = pd.read_csv(unified_windows_index(WIN_SEC)).query("domain == 'cardiac'")
    spr_w = pd.read_csv(sprsound_windows_index(WIN_SEC))
    windows = pd.concat([card_w, spr_w], ignore_index=True)
    assert (windows["n_samples"] == int(WIN_SEC * 4000)).all(), "hay ventanas del índice de menos de 4 s"

    # Audio: las grabaciones cardíacas del índice unificado, después todo SPRSound.
    card_audio, _ = open_audio_cache()
    card_off = pd.read_csv(UNIFIED_AUDIO_OFFSETS_CSV).set_index("file_id")
    card_off = card_off.loc[card_w["file_id"].unique()]
    spr_audio, _ = open_audio_cache(SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV)
    spr_off = pd.read_csv(SPRSOUND_AUDIO_OFFSETS_CSV).set_index("file_id")
    n_card = int(card_off["n_samples"].sum())
    audio = np.empty(n_card + len(spr_audio), dtype=np.int16)
    pos, rec_rows = 0, []
    subj = windows.drop_duplicates("file_id").set_index("file_id")["subject_id"]
    for fid, r in card_off.iterrows():
        audio[pos:pos + r.n_samples] = card_audio[r.offset:r.offset + r.n_samples]
        rec_rows.append({"file_id": fid, "subject_id": subj[fid], "domain": "cardiac", "offset": pos,
                         "n_samples": int(r.n_samples)})
        pos += int(r.n_samples)
    audio[pos:] = spr_audio
    spr_recs = pd.read_csv(SPRSOUND_RECORDINGS_CSV)[["file_id", "subject_id"]]
    spr_off = spr_off.join(spr_recs.set_index("file_id"), how="inner")
    for fid, r in spr_off.iterrows():
        rec_rows.append({"file_id": fid, "subject_id": r.subject_id, "domain": "pulmonary",
                         "offset": pos + int(r.offset), "n_samples": int(r.n_samples)})
    recordings = pd.DataFrame(rec_rows)
    assert recordings["file_id"].is_unique and set(windows["file_id"]) <= set(recordings["file_id"])

    # Todas las ventanas de SPRSound, con la misma función que la evaluación local.
    full = FullRecordingDataset(spr_recs, spr_audio, pd.read_csv(SPRSOUND_AUDIO_OFFSETS_CSV).set_index("file_id")).df
    full["start_sample"] = full["start"] - full["file_id"].map(spr_off["offset"])
    full = full[["file_id", "subject_id", "window_idx", "start_sec", "start_sample", "length"]]

    # Control: cada ventana del índice sale igual del audio nuevo que de su cache.
    rec = recordings.set_index("file_id")
    rng = np.random.default_rng(0)
    for src, (w, a, off) in {"cardiac": (card_w, card_audio, card_off),
                             "pulmonary": (spr_w, spr_audio, spr_off)}.items():
        for r in w.iloc[rng.choice(len(w), 200, replace=False)].itertuples():
            old = off.at[r.file_id, "offset"] + r.start_sample
            new = rec.at[r.file_id, "offset"] + r.start_sample
            assert np.array_equal(a[old:old + r.n_samples], audio[new:new + r.n_samples]), (src, r.file_id)

    folds = []
    for i in range(K):
        f = pd.concat([pd.read_csv(UNIFIED_KFOLD_DIR / f"fold{i}.csv")[["subject_id", "split"]],
                       pd.read_csv(SPRSOUND_KFOLD_DIR / f"fold{i}.csv")[["subject_id", "split"]]])
        f = f[f["subject_id"].isin(recordings["subject_id"])]
        assert f["subject_id"].is_unique and set(recordings["subject_id"]) <= set(f["subject_id"])
        folds.append(f)

    staging = DATA_DIR / "colab_cnn6_circor_spr"
    if staging.exists():
        shutil.rmtree(staging)
    (staging / "folds").mkdir(parents=True)
    np.save(staging / "audio_4k_int16.npy", audio)
    recordings.to_csv(staging / "recordings.csv", index=False)
    windows[WINDOW_COLS].to_csv(staging / "windows.csv", index=False)
    full.to_csv(staging / "full_windows_spr.csv", index=False)
    for i, f in enumerate(folds):
        f.to_csv(staging / "folds" / f"fold{i}.csv", index=False)
    shutil.copy2(PANNS_CNN6_CKPT, staging / PANNS_CNN6_CKPT.name)
    manifest = {
        "audio_samples": int(len(audio)), "audio_hours": round(len(audio) / 4000 / 3600, 2),
        "recordings": recordings["domain"].value_counts().to_dict(),
        "subjects": recordings.groupby("domain")["subject_id"].nunique().to_dict(),
        "index_windows": windows["domain"].value_counts().to_dict(),
        "index_windows_by_label": windows["label"].value_counts().sort_index().to_dict(),
        "full_windows_spr": len(full), "full_windows_spr_short": int((full["length"] < WIN_SEC * 4000).sum()),
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, default=int))

    # Sin compresión: el int16 casi no comprime y así se descomprime al instante.
    with zipfile.ZipFile(PACKAGE_ZIP, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in sorted(staging.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(staging))
    shutil.rmtree(staging)
    print(json.dumps(manifest, indent=2, default=int))
    print(f"✓ {PACKAGE_ZIP} — {PACKAGE_ZIP.stat().st_size / 1e6:.0f} MB | {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    pack()
