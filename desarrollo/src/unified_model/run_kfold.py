"""
unified_model/run_kfold.py
=============================
Entrenamiento k-fold del modelo único sin gate (avances/22): CNN chica
(--width) sobre ventanas fijas de 4 s, con MFCC o log-Mel (--feature) y
pooling promedio o promedio + máximo (--pooling), 6 clases softmax, folds
unificados (splits/make_unified_kfold.py). Piloto 1: MFCC + avg (los
defaults); piloto 2: log-Mel + avg_max. Ajustes posteriores, uno por vez
y elegidos por validación (folds 0 y 3): resolución temporal (--n-fft,
--hop, --time-keep-blocks), balance entre dominios (--class-weights sqrt,
--balance-domains) y SpecAugment (--specaug). Ablación por dominio
(--domain cardiac|pulmonary): la misma receta entrenada solo con las
ventanas de un dominio, con la softmax restringida a sus clases.

--data sprsound (avances/23): la misma receta pulmonar sobre SPRSound, con
su índice, cache y folds por paciente (preprocessing/make_sprsound_windows.py,
splits/make_sprsound_kfold.py). Además del val/test de SPRSound, cada fold
predice las ventanas pulmonares de test de ICBHI del mismo fold (role
"test_icbhi"): test cruzado, el modelo nunca vio ICBHI.

--data joint (avances/23, paso 4): entrena con SPRSound + las ventanas
pulmonares de ICBHI, cada una con sus folds (SPRSound por paciente, ICBHI
los unificados: el test de ICBHI es el mismo de siempre). Pesos por clase
sobre el train conjunto; el checkpoint se elige con el promedio de la
selección en val de cada fuente. Roles: val/test (SPRSound) y
val_icbhi/test_icbhi.

Receta:
- CrossEntropyLoss con pesos por clase inversos a la frecuencia en train
  (--class-weights inv) o su raíz cuadrada (sqrt), normalizados para que el
  peso medio por ventana de train sea 1.
- Adam lr 1e-3, batch 64, hasta --epochs épocas con early stopping
  (--patience épocas sin mejorar en val).
- Checkpoint por la métrica de selección en val: promedio de tres AUC por
  ventana, cada uno dentro de su dominio verdadero — murmullo (ventanas
  cardíacas), crackle y wheeze (ventanas pulmonares) — con las
  probabilidades derivadas de dataset.derived_probs. Es independiente del
  umbral: los umbrales se calibran después por grabación (pooled_fixed). La
  accuracy de dominio y la balanceada en 6 clases se registran pero no
  eligen.

Al terminar cada fold guarda las probabilidades por ventana del mejor
checkpoint para las ventanas de val y de test del fold (mismo formato que
integration/window_inference.py: fold, role), para la evaluación por
grabación.

Salidas (run = --run-name):
    unified_model/checkpoints_<run>/fold{i}_best.pt
    reports/unified_model/runs/<run>/history_fold{i}.csv
    reports/unified_model/runs/<run>/window_probs_fold{i}.csv
    reports/unified_model/runs/<run>/per_fold.csv
    (--data sprsound: reports/sprsound/runs/<run>/)

Uso:
    python unified_model/run_kfold.py --width base --folds 0 --run-name pilot_base
    python unified_model/run_kfold.py --width wide --feature logmel --pooling avg_max --run-name v1_wide
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV, SPRSOUND_KFOLD_DIR,
                          SPRSOUND_REPORTS_DIR, UNIFIED_KFOLD_DIR, UNIFIED_MODEL_REPORTS_DIR,
                          UNIFIED_MODEL_SRC_DIR, sprsound_windows_index, unified_windows_index)
from unified_model.dataset import UnifiedWindowDataset, collate, derived_probs, open_audio_cache
from unified_model.dataset import FEATURES
from unified_model.model import N_CLASSES, POOLING_TYPES, WIDTHS, UnifiedCNN

K = 5
DOMAIN_CLASSES = {"cardiac": (0, 1), "pulmonary": (2, 3, 4, 5)}
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42


def domain_sample_weights(domains: np.ndarray) -> np.ndarray:
    """Probabilidad de muestreo por ventana para que cada dominio sea la
    mitad de lo que se ve en una época (--balance-domains)."""
    w = np.zeros(len(domains))
    for d in np.unique(domains):
        m = domains == d
        w[m] = 0.5 / m.sum()
    return w


def class_weights(labels: np.ndarray, mode: str, sample_w: np.ndarray | None = None) -> torch.Tensor:
    """Pesos de la pérdida. Con sample_w (muestreo balanceado), las
    frecuencias son las que efectivamente ve el modelo, no las del índice."""
    sample_w = np.ones(len(labels)) if sample_w is None else sample_w
    freq = np.bincount(labels, weights=sample_w, minlength=N_CLASSES)
    seen = freq > 0   # en la ablación por dominio faltan las clases del otro
    w = np.zeros(N_CLASSES)
    w[seen] = 1.0 / freq[seen] if mode == "inv" else 1.0 / np.sqrt(freq[seen])
    w = w / ((w[labels] * sample_w).sum() / sample_w.sum())   # peso medio por ventana vista = 1
    return torch.tensor(w, dtype=torch.float32)


def build_model(args) -> UnifiedCNN:
    return UnifiedCNN(width=WIDTHS[args.width], feature=args.feature, pooling=args.pooling,
                      n_fft=args.n_fft, hop_length=args.hop, time_keep_blocks=args.time_keep_blocks,
                      specaug=args.specaug,
                      active_classes=DOMAIN_CLASSES.get(args.domain))


def predict(model, loader) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(torch.softmax(model(x), dim=1).numpy())
    return np.concatenate(out)


def window_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    d = derived_probs(probs)
    card = labels <= 1
    pulm = ~card
    crackle = np.isin(labels, [3, 5])
    wheeze = np.isin(labels, [4, 5])
    nan = float("nan")   # en la ablación por dominio, las métricas del otro quedan en NaN
    m = {
        "auc_murmur": roc_auc_score(labels[card] == 1, d["p_murmur"][card]) if card.any() else nan,
        "auc_crackle": roc_auc_score(crackle[pulm], d["p_crackle"][pulm]) if pulm.any() else nan,
        "auc_wheeze": roc_auc_score(wheeze[pulm], d["p_wheeze"][pulm]) if pulm.any() else nan,
        "domain_acc": float(((d["p_cardiac"] > 0.5) == card).mean()) if card.any() and pulm.any() else nan,
        "bal_acc_6": balanced_accuracy_score(labels, probs.argmax(1)),
    }
    m["select"] = float(np.nanmean([m["auc_murmur"], m["auc_crackle"], m["auc_wheeze"]]))
    return m


def load_sources(args) -> dict[str, tuple]:
    """Fuentes de ventanas: nombre -> (índice, audio, offsets, carpeta de
    folds, entrena). "main" es la del --data; "icbhi" (ventanas pulmonares
    del índice unificado, folds unificados) solo evalúa con --data sprsound
    (test cruzado) y también entrena con --data joint. Sus roles llevan el
    sufijo _icbhi."""
    if args.data == "unified":
        windows = pd.read_csv(unified_windows_index(args.win_sec))
        if args.domain != "both":
            windows = windows[windows["domain"] == args.domain].reset_index(drop=True)
        return {"main": (windows, *open_audio_cache(), UNIFIED_KFOLD_DIR, True)}
    icbhi = pd.read_csv(unified_windows_index(args.win_sec)).query("domain == 'pulmonary'").reset_index(drop=True)
    return {"main": (pd.read_csv(sprsound_windows_index(args.win_sec)),
                     *open_audio_cache(SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV), SPRSOUND_KFOLD_DIR, True),
            "icbhi": (icbhi, *open_audio_cache(), UNIFIED_KFOLD_DIR, args.data == "joint")}


def concat_dataset(parts: list[tuple[str, pd.DataFrame]], sources: dict, augment: bool = False) -> ConcatDataset:
    """Ventanas de varias fuentes, en el orden de parts (el de las predicciones)."""
    return ConcatDataset([UnifiedWindowDataset(df, sources[n][1], sources[n][2], augment=augment) for n, df in parts])


def val_metrics(val_parts: list[tuple[str, pd.DataFrame]], probs: np.ndarray) -> dict:
    """Métricas de la fuente principal; con --data joint, también las de
    ICBHI (prefijo icbhi_) y la selección es el promedio de las dos, para
    que el checkpoint no favorezca a la fuente con más ventanas."""
    out, start = {}, 0
    for name, df in val_parts:
        m = window_metrics(df["label"].to_numpy(), probs[start:start + len(df)])
        start += len(df)
        out.update(m if name == "main" else {f"{name}_{k}": v for k, v in m.items()})
    if len(val_parts) > 1:
        out["select"] = float(np.mean([out["select"]] + [out[f"{n}_select"] for n, _ in val_parts[1:]]))
    return out


def run_fold(i: int, sources: dict, args, ckpt_dir: Path, out_dir: Path) -> dict:
    torch.manual_seed(SEED + i)
    np.random.seed(SEED + i)
    split = {}   # (fuente, split) -> ventanas
    for name, (windows, _, _, fold_dir, _) in sources.items():
        fold = pd.read_csv(fold_dir / f"fold{i}.csv")[["subject_id", "split"]]
        w = windows.merge(fold, on="subject_id")
        for s in ("train", "val", "test"):
            split[name, s] = w[w["split"] == s]
    train_src = [n for n, src in sources.items() if src[4]]
    tr = pd.concat([split[n, "train"] for n in train_src])
    val_parts = [(n, split[n, "val"]) for n in train_src]
    va = pd.concat([df for _, df in val_parts])
    te = pd.concat([split[n, "test"] for n in train_src])

    print(f"Fold {i}: train {len(tr)} | val {len(va)} | test {len(te)} ventanas", flush=True)
    train_ds = concat_dataset([(n, split[n, "train"]) for n in train_src], sources, augment=True)
    if args.balance_domains:
        sample_w = domain_sample_weights(tr["domain"].to_numpy())
        sampler = WeightedRandomSampler(sample_w, num_samples=len(tr), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, collate_fn=collate)
    else:
        sample_w = None
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(concat_dataset(val_parts, sources), batch_size=256, shuffle=False, collate_fn=collate)

    model = build_model(args)
    weights = class_weights(tr["label"].to_numpy(), args.class_weights, sample_w)
    print(f"  pesos por clase: {np.round(weights.numpy(), 2).tolist()}")
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history, best, since_best = [], -1.0, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            n += len(y)
        val_probs = predict(model, val_loader)
        val_loss = float(criterion(torch.log(torch.tensor(val_probs).clamp_min(1e-8)),
                                   torch.tensor(va["label"].to_numpy())).item())
        m = val_metrics(val_parts, val_probs)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, **m,
                        "sec": time.perf_counter() - t0})
        improved = m["select"] > best
        print(f"  [{epoch:02d}] loss {total / n:.4f}/{val_loss:.4f} | AUC murm {m['auc_murmur']:.3f} "
              f"crk {m['auc_crackle']:.3f} whz {m['auc_wheeze']:.3f} | dominio {m['domain_acc']:.3f} "
              f"| bal6 {m['bal_acc_6']:.3f}"
              + (f" | ICBHI crk {m['icbhi_auc_crackle']:.3f} whz {m['icbhi_auc_wheeze']:.3f}" if "icbhi_select" in m else "")
              + f" | {time.perf_counter() - t0:.0f} s{' *' if improved else ''}", flush=True)
        if improved:
            best, since_best = m["select"], 0
            torch.save(model.state_dict(), ckpt_dir / f"fold{i}_best.pt")
        else:
            since_best += 1
            if since_best >= args.patience:
                print(f"  early stopping (mejor época {epoch - since_best})")
                break
    pd.DataFrame(history).to_csv(out_dir / f"history_fold{i}.csv", index=False)

    # Probabilidades del mejor checkpoint en val y test
    model.load_state_dict(torch.load(ckpt_dir / f"fold{i}_best.pt", map_location="cpu"))
    evals = []   # (rol, fuente, ventanas): val solo de las fuentes que entrenan
    for name, src in sources.items():
        suffix = "" if name == "main" else f"_{name}"
        evals += [(f"{s}{suffix}", name, split[name, s]) for s in ("val", "test") if s == "test" or src[4]]
    test_m, rows = {}, []
    for role, name, df in evals:
        probs = predict(model, DataLoader(concat_dataset([(name, df)], sources),
                                          batch_size=256, shuffle=False, collate_fn=collate))
        part = df[["file_id", "subject_id", "domain", "window_idx", "start_sec", "label"]].copy()
        part["fold"], part["role"] = i, role
        for k, v in derived_probs(probs).items():
            part[k] = v
        rows.append(part)
        if role.startswith("test"):
            test_m[role] = window_metrics(df["label"].to_numpy(), probs)
    pd.concat(rows).to_csv(out_dir / f"window_probs_fold{i}.csv", index=False)

    best_row = max(history, key=lambda h: h["select"])
    res = {"fold": i, "best_epoch": best_row["epoch"], "epochs_run": len(history),
           **{f"val_{k}": v for k, v in best_row.items() if k not in ("epoch", "sec")},
           **{f"{role}_{k}": v for role, m in test_m.items() for k, v in m.items()}}
    test_m = test_m["test"]
    print(f"  Fold {i} — mejor época {res['best_epoch']} | val select {res['val_select']:.4f} | "
          f"test: AUC murm {test_m['auc_murmur']:.3f} crk {test_m['auc_crackle']:.3f} "
          f"whz {test_m['auc_wheeze']:.3f} dominio {test_m['domain_acc']:.3f}", flush=True)
    (out_dir / f"result_fold{i}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", choices=list(WIDTHS), default="base")
    parser.add_argument("--feature", choices=FEATURES, default="mfcc")
    parser.add_argument("--pooling", choices=POOLING_TYPES, default="avg")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop", type=int, default=160, help="muestras a 4 kHz (160 = 40 ms)")
    parser.add_argument("--time-keep-blocks", type=int, default=0, choices=[0, 1, 2],
                        help="últimos bloques que reducen solo frecuencia")
    parser.add_argument("--specaug", type=int, default=0, help="máscaras de frecuencia y de tiempo por ejemplo")
    parser.add_argument("--class-weights", choices=["inv", "sqrt"], default="inv")
    parser.add_argument("--balance-domains", action="store_true",
                        help="muestreo con la mitad de cada dominio por época")
    parser.add_argument("--domain", choices=["both", "cardiac", "pulmonary"], default="both",
                        help="ablación: entrenar y evaluar solo con las ventanas de un dominio")
    parser.add_argument("--data", choices=["unified", "sprsound", "joint"], default="unified",
                        help="sprsound: solo SPRSound (implica --domain pulmonary) + test cruzado en ICBHI; "
                             "joint: SPRSound + ICBHI pulmonar en train/val, test de cada uno por separado")
    parser.add_argument("--win-sec", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.data != "unified":
        args.domain = "pulmonary"
        out_dir = SPRSOUND_REPORTS_DIR / "runs" / args.run_name
    else:
        out_dir = UNIFIED_MODEL_REPORTS_DIR / "runs" / args.run_name
    sources = load_sources(args)
    ckpt_dir = UNIFIED_MODEL_SRC_DIR / f"checkpoints_{args.run_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_params = sum(p.numel() for p in build_model(args).parameters())
    # sin "folds": varios procesos pueden correr folds distintos del mismo run
    config = {k: v for k, v in vars(args).items() if k != "folds"}
    (out_dir / "config.json").write_text(json.dumps({**config, "n_params": n_params}, indent=2))
    print(f"Run {args.run_name}: width={args.width}, {args.feature}, pooling {args.pooling} "
          f"({n_params} parámetros), pesos {args.class_weights}, dominio {args.domain}, folds {args.folds}", flush=True)

    for i in args.folds:
        run_fold(i, sources, args, ckpt_dir, out_dir)
    # resumen con TODOS los folds terminados del run (result_fold*.json), no solo los de este proceso
    per_fold = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out_dir.glob("result_fold*.json"))])
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)
    print(f"\nFolds terminados: {per_fold['fold'].tolist()}")
    cols = [c for c in per_fold.columns if c.startswith("test_")]
    print(f"\n{per_fold[cols].agg(['mean', 'std']).T.round(4).to_string()}")


if __name__ == "__main__":
    main()
