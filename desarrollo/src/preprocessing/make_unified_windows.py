"""
preprocessing/make_unified_windows.py
========================================
Índice de ventanas fijas del modelo único sin gate (avances/21 §3.2-3.3),
con la etiqueta de 6 clases de cada ventana. No guarda audio: cada fila es
(file_id, start_sample, n_samples) sobre la grabación preprocesada con
audio_ops.preprocess SIN pasa-altos (el modelo ve los dos dominios y el
pasa-altos de 100 Hz borraría S1/S2) a 4 kHz. El dataset de entrenamiento
recorta el audio a partir de este índice.

Ventaneo: el mismo de integration/window_inference.py (fixed_window_starts):
paso = mitad de la ventana, la última alineada al final, y una grabación más
corta que la ventana entra entera. Así las ventanas de entrenamiento son las
mismas que se usan al predecir.

Clases (CLASSES):
    0 card_normal   cardíaca sin murmullo (normal_heart o abnormal_heart_unspecified)
    1 card_murmur   cardíaca con murmullo (pathology_label == heart_murmur)
    2 pulm_normal
    3 pulm_crackle
    4 pulm_wheeze
    5 pulm_both

Cardíacas: etiqueta de la grabación (no hay ciclos anotados).
Pulmonares: regla de superposición con los ciclos anotados de ICBHI
(data/pulmonary/cycles_index.csv). Una ventana tiene crackle si se superpone
con algún ciclo con crackle en al menos --min-overlap segundos O en al menos
la mitad de ese ciclo (para que los ciclos más cortos que el umbral también
cuenten); ídem wheeze; las dos -> both. Las ventanas que no tocan ningún
ciclo anotado se descartan (no sabemos qué tienen). Se marcan como
"borderline" las ventanas normales que tocan un ciclo anormal por debajo
del umbral.

Además compara los umbrales OVERLAP_COMPARE (cuántas ventanas cambian de
clase) para justificar el elegido.

Salidas:
    data/unified_model/windows_index_<win>s.csv
    reports/unified_model/windows_<win>s_by_fold.csv
    reports/unified_model/overlap_threshold_comparison_<win>s.csv

Uso:
    python preprocessing/make_unified_windows.py [--win-sec 4] [--min-overlap 1.0]
"""

import argparse

import numpy as np
import pandas as pd
import soundfile as sf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    AUDIO_DIR_V2, MASTER_CSV_V2, PULMONARY_CYCLES_INDEX_CSV, UNIFIED_KFOLD_DIR,
    UNIFIED_MODEL_REPORTS_DIR, unified_windows_index,
)
from integration.window_inference import SR, fixed_window_starts

K = 5
CLASSES = ["card_normal", "card_murmur", "pulm_normal", "pulm_crackle", "pulm_wheeze", "pulm_both"]
OVERLAP_COMPARE = (0.5, 1.0)


def preprocessed_length(path: Path, target_sr: int = SR) -> int:
    """Largo en muestras tras audio_ops.preprocess, sin cargar el audio:
    el filtrado no cambia el largo y librosa.resample devuelve
    ceil(n · target/orig) muestras."""
    info = sf.info(str(path))
    if info.samplerate == target_sr:
        return info.frames
    return int(np.ceil(info.frames * target_sr / info.samplerate))


def cardiac_windows(recs: pd.DataFrame, win: int, hop: int) -> pd.DataFrame:
    rows = []
    for _, r in recs.iterrows():
        n = preprocessed_length(AUDIO_DIR_V2 / r["file_id"])
        label = 1 if r["pathology_label"] == "heart_murmur" else 0
        for w, s in enumerate(fixed_window_starts(n, win, hop)):
            rows.append({"file_id": r["file_id"], "subject_id": r["subject_id"], "domain": "cardiac",
                         "window_idx": w, "start_sample": s, "n_samples": min(win, n - s), "label": label})
    return pd.DataFrame(rows)


def pulmonary_windows(recs: pd.DataFrame, cycles: pd.DataFrame, win: int, hop: int) -> pd.DataFrame:
    """Una fila por ventana con la superposición máxima (en s y como
    fracción del ciclo) contra ciclos con crackle y con wheeze, y el total
    superpuesto con ciclos anotados. La etiqueta se decide después
    (label_pulmonary) para poder comparar umbrales sin recalcular."""
    by_file = {f: g for f, g in cycles.groupby("file_id")}
    rows = []
    for _, r in recs.iterrows():
        n = preprocessed_length(AUDIO_DIR_V2 / r["file_id"])
        c = by_file.get(r["file_id"])
        cs, ce = c["start_sec"].to_numpy(), (c["start_sec"] + c["duration_sec"]).to_numpy()
        dur = c["duration_sec"].to_numpy()
        for w, s in enumerate(fixed_window_starts(n, win, hop)):
            a, b = s / SR, (s + min(win, n - s)) / SR
            ov = np.clip(np.minimum(b, ce) - np.maximum(a, cs), 0, None)
            row = {"file_id": r["file_id"], "subject_id": r["subject_id"], "domain": "pulmonary",
                   "window_idx": w, "start_sample": s, "n_samples": min(win, n - s),
                   "annotated_sec": float(ov.sum())}
            for col in ("crackle", "wheeze"):
                m = c[col].to_numpy() == 1
                row[f"{col}_ov_sec"] = float(ov[m].max()) if m.any() else 0.0
                row[f"{col}_ov_frac"] = float((ov[m] / dur[m]).max()) if m.any() else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def label_pulmonary(df: pd.DataFrame, min_overlap: float) -> tuple[pd.Series, pd.Series]:
    """(clase 2-5, borderline) para cada ventana pulmonar."""
    has = {}
    touches = {}
    for col in ("crackle", "wheeze"):
        has[col] = (df[f"{col}_ov_sec"] >= min_overlap) | (df[f"{col}_ov_frac"] >= 0.5)
        touches[col] = df[f"{col}_ov_sec"] > 0
    label = 2 + has["crackle"].astype(int) + 2 * has["wheeze"].astype(int)   # 2 normal, 3 crackle, 4 wheeze, 5 both
    borderline = (label == 2) & (touches["crackle"] | touches["wheeze"])
    return label, borderline


def fold_split_table(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(K):
        f = pd.read_csv(UNIFIED_KFOLD_DIR / f"fold{i}.csv")[["subject_id", "split"]]
        t = windows.merge(f, on="subject_id").groupby(["split", "label"]).size().unstack(fill_value=0)
        t.columns = [CLASSES[c] for c in t.columns]
        t.insert(0, "fold", i)
        rows.append(t.reset_index())
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--win-sec", type=float, default=4.0)
    parser.add_argument("--min-overlap", type=float, default=1.0, help="segundos (ver docstring)")
    args = parser.parse_args()
    win, hop = int(args.win_sec * SR), int(args.win_sec * SR / 2)

    master = pd.read_csv(MASTER_CSV_V2)
    cycles = pd.read_csv(PULMONARY_CYCLES_INDEX_CSV)
    card = cardiac_windows(master[master["sound_type"] == "cardiac"], win, hop)
    pulm = pulmonary_windows(master[master["sound_type"] == "pulmonary"], cycles, win, hop)
    print(f"Ventanas cardíacas: {len(card)} ({card['file_id'].nunique()} grabaciones)")
    print(f"Ventanas pulmonares: {len(pulm)} ({pulm['file_id'].nunique()} grabaciones)")

    unannotated = pulm["annotated_sec"] == 0
    print(f"Pulmonares sin ningún ciclo anotado (se descartan): {unannotated.sum()}")
    pulm = pulm[~unannotated].copy()

    # Comparación de umbrales
    UNIFIED_MODEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = {t: label_pulmonary(pulm, t) for t in OVERLAP_COMPARE + (args.min_overlap,)}
    comp = pd.DataFrame({f"{t:g}s": labels[t][0].map(lambda c: CLASSES[c]).value_counts() for t in OVERLAP_COMPARE})
    comp.loc["borderline (normal que toca un ciclo anormal)"] = [labels[t][1].sum() for t in OVERLAP_COMPARE]
    lo, hi = OVERLAP_COMPARE
    changed = labels[lo][0] != labels[hi][0]
    comp.loc[f"cambian de clase entre {lo:g}s y {hi:g}s"] = [changed.sum(), changed.sum()]
    comp_path = UNIFIED_MODEL_REPORTS_DIR / f"overlap_threshold_comparison_{args.win_sec:g}s.csv"
    comp.to_csv(comp_path)
    print(f"\nComparación de umbrales de superposición:\n{comp.to_string()}")
    print("Transiciones (filas: umbral bajo, columnas: umbral alto):")
    print(pd.crosstab(labels[lo][0][changed].map(lambda c: CLASSES[c]),
                      labels[hi][0][changed].map(lambda c: CLASSES[c])).to_string())

    pulm["label"], pulm["borderline"] = labels[args.min_overlap]
    windows = pd.concat([card, pulm], ignore_index=True)
    windows["label_name"] = windows["label"].map(lambda c: CLASSES[c])
    windows["start_sec"] = windows["start_sample"] / SR

    index_path = unified_windows_index(args.win_sec)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    windows.to_csv(index_path, index=False)
    print(f"\n✓ {index_path} — {len(windows)} ventanas, umbral {args.min_overlap:g} s")
    print(windows["label_name"].value_counts().reindex(CLASSES).to_string())

    by_fold = fold_split_table(windows)
    by_fold_path = UNIFIED_MODEL_REPORTS_DIR / f"windows_{args.win_sec:g}s_by_fold.csv"
    by_fold.to_csv(by_fold_path, index=False)
    print(f"\n{by_fold.to_string(index=False)}\n✓ {by_fold_path}")


if __name__ == "__main__":
    main()
