"""
integration/window_inference.py
==================================
Paso 1 de la integración (avances/18 §5): pasa el audio crudo del dataset
unificado por los modelos ya entrenados, cortado en ventanas FIJAS (sin las
anotaciones de ciclos), y guarda la probabilidad de cada ventana. Las reglas
de agregación a nivel grabación se prueban después
(integration/evaluate_integration.py) sin volver a correr los modelos.

Cada rama usa el mismo preprocesamiento que vio su modelo al entrenar:

- gate (v5): audio_ops.preprocess -> ventanas de 5 s, paso 2.5 s, con el
  mismo corte que preprocessing/make_gate_windows.py -> MFCC (20) z-score.
  Un solo checkpoint (split único); se guardan todas las grabaciones con su
  split, y la capa 1 usa solo las de test.
- murmullo (v1 k-fold): audio_ops.preprocess (sin pasa-altos) -> ventanas de
  --murmur-sec (2.5 s) -> MFCC (20 coef., n_mels 64, n_fft 512, hop 160)
  z-score por ventana.
- pulmonar (CNN6 variante B): preprocess(..., highpass_cutoff=100) ->
  ventanas de --pulmonary-sec (4 s) -> remuestreo a 32 kHz -> log-Mel de
  PANNs ("panns32k") -> recorte centrado a 400 frames.

Ventanas de las ramas: paso = mitad de la ventana (50% de solapamiento). Si
la última no llega al final de la grabación, se agrega una alineada al final
(en vez de rellenar con ceros, que ninguno de los dos modelos vio al
entrenar). Una grabación más corta que la ventana entra entera, como una
sola ventana (los dos modelos aceptan duración variable).

Ramas k-fold: cada grabación pasa por DOS modelos — el del fold donde su
paciente es validación (para calibrar la agregación) y el del fold donde es
test (para evaluar). Ninguno de los dos la vio al entrenar.

Además, para la comparación con la "segmentación oráculo" (avances/18 §4.4):
- murmur_oracle: las ventanas de 4 latidos ya recortadas de los .tsv
  (data/cardiac/windows), por los mismos modelos y roles.
- pulmonary_oracle: los ciclos anotados de ICBHI, recortados del audio
  preprocesado igual que preprocessing/make_pulmonary_cycles.py. Sirve
  también de chequeo: debe reproducir oof_predictions.csv de v8_cnn6_ft_B.

Salida: reports/integration/window_probs/<rama>.csv

Uso:
    python integration/window_inference.py --branch gate
    python integration/window_inference.py --branch murmur [--murmur-sec 2.5]
    python integration/window_inference.py --branch pulmonary [--pulmonary-sec 4]
    python integration/window_inference.py --branch murmur_oracle
    python integration/window_inference.py --branch pulmonary_oracle
    # prueba rápida:
    python integration/window_inference.py --branch pulmonary --limit 5
"""

import argparse
import time

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (
    AUDIO_DIR_V2, CARDIAC_MURMUR_KFOLD_CHECKPOINTS_DIR, CARDIAC_MURMUR_KFOLD_DIR,
    CARDIAC_WINDOWS_DIR, CARDIAC_WINDOWS_INDEX_CSV, GATE_V5_CHECKPOINTS_DIR,
    INTEGRATION_PROBS_DIR, MASTER_CSV_V2, PANNS_CNN6_CKPT, PULMONARY_KFOLD_DIR,
    PULMONARY_V8B_DIR, SPLITS_CSV, pulmonary_cycles_paths,
)
from preprocessing.audio_ops import preprocess
from preprocessing.make_gate_windows import _slice_windows as gate_slice_windows
from gate_model.model import GateCNN
from cardiac_murmur_model.model import MurmurCNN
from cardiac_murmur_model.dataset_kfold import build_window_path_index
from pulmonary_adventitious_model.extract_cnn6_embeddings import FRONTENDS, logmel
from pulmonary_adventitious_model.panns_cnn6 import load_pretrained_cnn6
from pulmonary_adventitious_model.run_finetune_gpu import MAX_FRAMES, PAD_DB, VARIANTS, FineTuneCnn6

K = 5
SR = 4000
# Features de murmullo y gate: los defaults de MurmurDataset / GateDataset.
MFCC_KW = {"n_mfcc": 20, "n_mels": 64, "n_fft": 512, "hop_length": 160}
PULMONARY_HIGHPASS = 100.0
PULMONARY_VARIANT = "B"
ROLES = ("val", "test")


# ── Ventaneo y features ───────────────────────────────────────────────────────
def fixed_window_starts(n: int, win: int, hop: int) -> list[int]:
    """Inicios (en muestras) de ventanas de largo win con paso hop. La
    última se alinea al final si la grilla no lo cubre. n < win -> [0]
    (la grabación entera, una sola ventana más corta)."""
    if n <= win:
        return [0]
    starts = list(range(0, n - win + 1, hop))
    if starts[-1] + win < n:
        starts.append(n - win)
    return starts


def _zscore(feat: np.ndarray) -> np.ndarray:
    mean, std = feat.mean(), feat.std()
    return feat - mean if std < 1e-6 else (feat - mean) / std


def mfcc_batch(chunks: list[np.ndarray]) -> torch.Tensor:
    """Ventanas del mismo largo -> (B, 1, 20, frames), igual que
    MurmurDataset / GateDataset sin augmentation."""
    feats = [_zscore(librosa.feature.mfcc(y=c.astype(np.float32), sr=SR, **MFCC_KW)) for c in chunks]
    return torch.tensor(np.stack(feats), dtype=torch.float32)[:, None]


_PANNS = FRONTENDS["panns32k"]
_MEL_BASIS = librosa.filters.mel(sr=_PANNS["sr"], n_fft=_PANNS["n_fft"], n_mels=64,
                                 fmin=_PANNS["fmin"], fmax=_PANNS["fmax"])


def pulmonary_batch(chunks: list[np.ndarray], max_frames: int = MAX_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    """Como LogmelBank.batch(train=False) de run_finetune_gpu.py: recorte
    centrado a MAX_FRAMES y relleno con silencio (PAD_DB), pero SIEMPRE
    hasta MAX_FRAMES y no hasta el más largo del batch. El relleno cambia
    la salida (las convoluciones lo ven en el borde, aunque el pooling lo
    enmascare): en Colab dependía de con quién compartía batch cada ciclo
    (hasta 0.18 de diferencia en un ciclo corto procesado solo). Al
    entrenar, un batch de 64 ciclos casi siempre trae uno de >= 4 s, así
    que rellenar a MAX_FRAMES es lo que vio el modelo.

    max_frames > MAX_FRAMES (ventanas de más de 4 s) evita el recorte: la
    ventana entra entera y el pooling final, que es global, la resume."""
    feats = []
    for c in chunks:
        f = logmel(c.astype(np.float32), SR, _PANNS, _MEL_BASIS)        # (frames, 64)
        if f.shape[0] > max_frames:
            off = (f.shape[0] - max_frames) // 2
            f = f[off:off + max_frames]
        feats.append(f)
    lens = [f.shape[0] for f in feats]
    x = np.full((len(feats), max_frames, 64), PAD_DB, dtype=np.float32)
    for j, f in enumerate(feats):
        x[j, :len(f)] = f
    return torch.from_numpy(x)[:, None], torch.tensor(lens)


# ── Modelos ───────────────────────────────────────────────────────────────────
def load_gate() -> GateCNN:
    m = GateCNN()
    m.load_state_dict(torch.load(GATE_V5_CHECKPOINTS_DIR / "best.pt", map_location="cpu"))
    return m.eval()


def load_murmur_folds() -> list[MurmurCNN]:
    models = []
    for i in range(K):
        m = MurmurCNN()
        m.load_state_dict(torch.load(CARDIAC_MURMUR_KFOLD_CHECKPOINTS_DIR / f"fold{i}_best.pt", map_location="cpu"))
        models.append(m.eval())
    return models


def load_pulmonary_folds() -> list[FineTuneCnn6]:
    first = VARIANTS[PULMONARY_VARIANT]["first_trainable"]
    models = []
    for i in range(K):
        m = FineTuneCnn6(load_pretrained_cnn6(PANNS_CNN6_CKPT), first)
        m.load_state_dict(torch.load(PULMONARY_V8B_DIR / f"fold{i}_best.pt", map_location="cpu"))
        models.append(m.eval())
    return models


def fold_roles(kfold_dir: Path) -> pd.DataFrame:
    """subject_id, fold, role — una fila por (paciente, fold) donde el
    paciente es val o test."""
    rows = []
    for i in range(K):
        f = pd.read_csv(kfold_dir / f"fold{i}.csv")
        f = f[f["split"].isin(ROLES)].rename(columns={"split": "role"})
        f["fold"] = i
        rows.append(f[["subject_id", "fold", "role"]])
    return pd.concat(rows, ignore_index=True)


# ── Ramas ─────────────────────────────────────────────────────────────────────
def run_gate(recs: pd.DataFrame) -> pd.DataFrame:
    model = load_gate()
    splits = pd.read_csv(SPLITS_CSV)
    recs = recs.merge(splits, on="subject_id", how="left")
    rows = []
    with torch.no_grad():
        for _, r in _progress(recs):
            y, _ = preprocess(AUDIO_DIR_V2 / r["file_id"], target_sr=SR)
            chunks = gate_slice_windows(y)
            if not chunks:
                continue
            probs = torch.sigmoid(model(mfcc_batch(chunks))).numpy()
            for w, p in enumerate(probs):
                rows.append({"file_id": r["file_id"], "subject_id": r["subject_id"], "split": r["split"],
                             "sound_type": r["sound_type"], "window_idx": w,
                             "start_sec": w * 2.5, "prob_cardiac": float(p)})
    return pd.DataFrame(rows)


def run_murmur(recs: pd.DataFrame, win_sec: float) -> pd.DataFrame:
    models = load_murmur_folds()
    recs = recs[recs["sound_type"] == "cardiac"].merge(fold_roles(CARDIAC_MURMUR_KFOLD_DIR), on="subject_id")
    win, hop = int(win_sec * SR), int(win_sec * SR / 2)
    rows = []
    with torch.no_grad():
        for file_id, g in _progress_groups(recs):
            y, _ = preprocess(AUDIO_DIR_V2 / file_id, target_sr=SR)
            starts = fixed_window_starts(len(y), win, hop)
            x = mfcc_batch([y[s:s + win] for s in starts])
            for _, r in g.iterrows():
                probs = torch.sigmoid(models[r["fold"]](x)).numpy()
                for w, (s, p) in enumerate(zip(starts, probs)):
                    rows.append({"file_id": file_id, "subject_id": r["subject_id"], "fold": r["fold"],
                                 "role": r["role"], "window_idx": w, "start_sec": s / SR, "prob": float(p)})
    return pd.DataFrame(rows)


def run_pulmonary(recs: pd.DataFrame, win_sec: float) -> pd.DataFrame:
    models = load_pulmonary_folds()
    recs = recs[recs["sound_type"] == "pulmonary"].merge(fold_roles(PULMONARY_KFOLD_DIR), on="subject_id")
    win, hop = int(win_sec * SR), int(win_sec * SR / 2)
    # hasta 4 s, el recorte a MAX_FRAMES de siempre; más largas, enteras (100 frames/s + 1)
    max_frames = MAX_FRAMES if win_sec * 100 <= MAX_FRAMES else int(round(win_sec * 100)) + 1
    rows = []
    with torch.no_grad():
        for file_id, g in _progress_groups(recs):
            y, _ = preprocess(AUDIO_DIR_V2 / file_id, target_sr=SR, highpass_cutoff=PULMONARY_HIGHPASS)
            starts = fixed_window_starts(len(y), win, hop)
            x, lens = pulmonary_batch([y[s:s + win] for s in starts], max_frames)
            for _, r in g.iterrows():
                probs = torch.sigmoid(models[r["fold"]](x, lens)).numpy()
                for w, (s, p) in enumerate(zip(starts, probs)):
                    rows.append({"file_id": file_id, "subject_id": r["subject_id"], "fold": r["fold"],
                                 "role": r["role"], "window_idx": w, "start_sec": s / SR,
                                 "p_crackle": float(p[0]), "p_wheeze": float(p[1])})
    return pd.DataFrame(rows)


def run_murmur_oracle(limit: int | None) -> pd.DataFrame:
    """Ventanas de 4 latidos de los .tsv, una por una (sin el relleno con
    ceros del batch de entrenamiento, que dependía de con quién compartía
    batch cada ventana)."""
    models = load_murmur_folds()
    windows = pd.read_csv(CARDIAC_WINDOWS_INDEX_CSV)[["window_id", "file_id", "subject_id"]]
    if limit is not None:
        windows = windows[windows["file_id"].isin(windows["file_id"].unique()[:limit])]
    windows = windows.merge(fold_roles(CARDIAC_MURMUR_KFOLD_DIR), on="subject_id")
    path_index = build_window_path_index(CARDIAC_WINDOWS_DIR)
    rows = []
    with torch.no_grad():
        for window_id, g in _progress_groups(windows, key="window_id", every=5000):
            y, _ = sf.read(path_index[window_id], dtype="float32")
            x = mfcc_batch([y])
            for _, r in g.iterrows():
                rows.append({"window_id": window_id, "file_id": r["file_id"], "subject_id": r["subject_id"],
                             "fold": r["fold"], "role": r["role"],
                             "prob": float(torch.sigmoid(models[r["fold"]](x))[0])})
    return pd.DataFrame(rows)


def run_pulmonary_oracle(recs: pd.DataFrame) -> pd.DataFrame:
    models = load_pulmonary_folds()
    _, index_csv = pulmonary_cycles_paths("hp100")
    cycles = pd.read_csv(index_csv)
    cycles = cycles[cycles["file_id"].isin(recs["file_id"])]
    roles = fold_roles(PULMONARY_KFOLD_DIR)
    rows = []
    with torch.no_grad():
        for file_id, g in _progress_groups(cycles):
            y, _ = preprocess(AUDIO_DIR_V2 / file_id, target_sr=SR, highpass_cutoff=PULMONARY_HIGHPASS)
            # mismo recorte que make_pulmonary_cycles.py (y su redondeo a PCM 16 bits al guardar)
            chunks = []
            for _, c in g.iterrows():
                a = int(c["start_sec"] * SR)
                chunk = y[a:a + int(round(c["duration_sec"] * SR))]
                chunks.append(np.round(chunk * 32767) / 32768)   # libsndfile: escribe x·32767, lee /32768
            subj = g["subject_id"].iloc[0]
            for _, rr in roles[roles["subject_id"] == subj].iterrows():
                for s in range(0, len(chunks), 64):
                    x, lens = pulmonary_batch(chunks[s:s + 64])
                    probs = torch.sigmoid(models[rr["fold"]](x, lens)).numpy()
                    for (_, c), p in zip(g.iloc[s:s + 64].iterrows(), probs):
                        rows.append({"cycle_id": c["cycle_id"], "file_id": file_id, "subject_id": subj,
                                     "fold": rr["fold"], "role": rr["role"],
                                     "crackle": c["crackle"], "wheeze": c["wheeze"],
                                     "p_crackle": float(p[0]), "p_wheeze": float(p[1])})
    return pd.DataFrame(rows)


# ── Utilidades ────────────────────────────────────────────────────────────────
_T0 = time.perf_counter()


def _log(i: int, n: int) -> None:
    el = time.perf_counter() - _T0
    print(f"  {i}/{n} ({el:.0f} s, faltan ~{el / i * (n - i):.0f} s)", flush=True)


def _progress(df: pd.DataFrame, every: int = 200):
    for i, item in enumerate(df.iterrows(), 1):
        yield item
        if i % every == 0:
            _log(i, len(df))


def _progress_groups(df: pd.DataFrame, key: str = "file_id", every: int = 100):
    groups = df.groupby(key, sort=False)
    for i, item in enumerate(groups, 1):
        yield item
        if i % every == 0:
            _log(i, groups.ngroups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", required=True,
                        choices=["gate", "murmur", "pulmonary", "murmur_oracle", "pulmonary_oracle"])
    parser.add_argument("--murmur-sec", type=float, default=2.5)
    parser.add_argument("--pulmonary-sec", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None, help="solo las primeras N grabaciones")
    parser.add_argument("--suffix", default="", help="agregado al nombre del CSV de salida")
    args = parser.parse_args()

    torch.set_flush_denormal(True)   # bandas vacías del log-Mel de PANNs (ver extract_cnn6_embeddings.py)
    recs = pd.read_csv(MASTER_CSV_V2)[["file_id", "subject_id", "sound_type"]]
    if args.limit is not None and args.branch != "murmur_oracle":
        recs = recs.groupby("sound_type", group_keys=False).head(args.limit)

    if args.branch == "gate":
        out = run_gate(recs)
    elif args.branch == "murmur":
        out = run_murmur(recs, args.murmur_sec)
    elif args.branch == "pulmonary":
        out = run_pulmonary(recs, args.pulmonary_sec)
    elif args.branch == "murmur_oracle":
        out = run_murmur_oracle(args.limit)
    else:
        out = run_pulmonary_oracle(recs)

    INTEGRATION_PROBS_DIR.mkdir(parents=True, exist_ok=True)
    path = INTEGRATION_PROBS_DIR / f"{args.branch}{args.suffix}.csv"
    out.to_csv(path, index=False)
    print(f"✓ {path} — {len(out)} filas, {out['file_id'].nunique()} grabaciones, "
          f"{time.perf_counter() - _T0:.0f} s")


if __name__ == "__main__":
    main()
