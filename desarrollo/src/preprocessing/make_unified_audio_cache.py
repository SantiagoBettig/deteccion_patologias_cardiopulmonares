"""
preprocessing/make_unified_audio_cache.py
============================================
Cache del audio preprocesado para el modelo único sin gate: las 4083
grabaciones pasadas una sola vez por audio_ops.preprocess (pasa-bajos,
4 kHz, RMS; SIN pasa-altos, ver make_unified_windows.py), concatenadas en
un solo .npy de int16 (PCM 16 bits, como los ciclos y ventanas que ya se
guardaban en WAV). El dataset lo abre con memory map y recorta cada ventana
con (offset de la grabación + start_sample) sin cargar el resto en RAM.

Con --data sprsound hace lo mismo con las grabaciones de SPRSound que usa
preprocessing/make_sprsound_windows.py (data/sprsound/recordings.csv).

Salidas:
    data/unified_model/audio_4k_int16.npy   (~720 MB)
    data/unified_model/audio_4k_offsets.csv (file_id, offset, n_samples)
    --data sprsound: data/sprsound/audio_4k_{int16.npy,offsets.csv}

Uso:
    python preprocessing/make_unified_audio_cache.py [--data unified|sprsound]
"""

import argparse
import time

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (AUDIO_DIR_V2, MASTER_CSV_V2, SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV,
                          SPRSOUND_RAW_DIR, SPRSOUND_RECORDINGS_CSV, UNIFIED_AUDIO_CACHE_NPY,
                          UNIFIED_AUDIO_OFFSETS_CSV)
from preprocessing.audio_ops import preprocess

SR = 4000


def sources(data: str) -> tuple[list[tuple[str, Path]], Path, Path]:
    """[(file_id, ruta del WAV)], .npy y .csv de offsets de salida."""
    if data == "sprsound":
        recs = pd.read_csv(SPRSOUND_RECORDINGS_CSV)
        files = [(f, SPRSOUND_RAW_DIR / p) for f, p in zip(recs["file_id"], recs["audio_path"])]
        return files, SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV
    files = [(f, AUDIO_DIR_V2 / f) for f in pd.read_csv(MASTER_CSV_V2)["file_id"]]
    return files, UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["unified", "sprsound"], default="unified")
    files, cache_npy, offsets_csv = sources(parser.parse_args().data)
    t0 = time.perf_counter()
    chunks, rows, offset = [], [], 0
    for i, (f, path) in enumerate(files, 1):
        y, _ = preprocess(path, target_sr=SR)
        chunks.append(np.round(np.clip(y, -1.0, 1.0) * 32767).astype(np.int16))
        rows.append({"file_id": f, "offset": offset, "n_samples": len(y)})
        offset += len(y)
        if i % 500 == 0:
            print(f"  {i}/{len(files)} ({time.perf_counter() - t0:.0f} s)", flush=True)

    cache_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_npy, np.concatenate(chunks))
    pd.DataFrame(rows).to_csv(offsets_csv, index=False)
    print(f"✓ {cache_npy} — {offset} muestras ({offset / SR / 3600:.1f} h), "
          f"{time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
