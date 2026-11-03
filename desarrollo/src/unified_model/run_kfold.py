"""
unified_model/run_kfold.py
=============================
Entrenamiento k-fold del modelo único sin gate (avances/22): CNN chica
(--width) sobre ventanas fijas de 4 s, con MFCC o log-Mel (--feature) y
pooling promedio o promedio + máximo (--pooling), 6 clases softmax, folds
unificados (splits/make_unified_kfold.py). Piloto 1: MFCC + avg (los
defaults); piloto 2: log-Mel + avg_max. Ajustes posteriores, uno por vez
y elegidos por validación (folds 0 y 3): resolución temporal (--n-fft,
--hop, --time-keep-blocks, --n-fft2), balance entre dominios (--class-weights sqrt,
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

--data circor_spr (avances/24): el modelo de 6 clases con las ventanas
cardíacas del índice unificado (CirCor, folds unificados: el test cardíaco
es el de avances/22) + SPRSound para pulmonar (sus folds). Las dos fuentes
se evalúan juntas, con los mismos roles val/test que --data unified.

Receta:
- CrossEntropyLoss con pesos por clase inversos a la frecuencia en train
  (--class-weights inv) o su raíz cuadrada (sqrt), normalizados para que el
  peso medio por ventana de train sea 1.
- Adam lr 1e-3, batch 64, hasta --epochs épocas con early stopping
  (--patience épocas sin mejorar en val).
- Checkpoint por una métrica de selección en val, con AUC por ventana
  calculados cada uno dentro de su dominio verdadero — murmullo (ventanas
  cardíacas), crackle y wheeze (ventanas pulmonares) — con las
  probabilidades derivadas de dataset.derived_probs. Hay dos criterios y
  se guarda el mejor checkpoint de cada uno; --select elige el principal:
  - "select" (A): promedio de los tres AUC (pulmonar pesa 2/3). Es el de
    avances/22 y el default.
  - "select_domain" (B): promedio por dominio, AUC murmullo y la media de
    crackle y wheeze (cada dominio pesa 1/2). Principal desde avances/24.
  Son independientes del umbral: los umbrales se calibran después por
  grabación (pooled_fixed). La accuracy de dominio y la balanceada en 6
  clases se registran pero no eligen. El early stopping corta cuando
  ningún criterio mejoró en --patience épocas.

Al terminar cada fold guarda las probabilidades por ventana del mejor
checkpoint para las ventanas de val y de test del fold (mismo formato que
integration/window_inference.py: fold, role), para la evaluación por
grabación; las del otro criterio, con el sufijo _<criterio>.

Salidas (run = --run-name; crit = el criterio que no es el principal):
    unified_model/checkpoints_<run>/fold{i}_best.pt, fold{i}_best_<crit>.pt
    reports/unified_model/runs/<run>/history_fold{i}.csv
    reports/unified_model/runs/<run>/window_probs_fold{i}.csv, window_probs_fold{i}_<crit>.csv
    reports/unified_model/runs/<run>/per_fold.csv
    (--data sprsound|joint: reports/sprsound/runs/<run>/)

Uso:
    python unified_model/run_kfold.py --width base --folds 0 --run-name pilot_base
    python unified_model/run_kfold.py --width wide --feature logmel --pooling avg_max --run-name v1_wide
    python unified_model/run_kfold.py --data circor_spr --feature logmel --pooling avg_max --select select_domain --run-name cs_base
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
CRITERIA = ("select", "select_domain")


def domain_sample_weights(domains: np.ndarray) -> np.ndarray:
    """Probabilidad de muestreo por ventana para que cada dominio sea la
    mitad de lo que se ve en una época (--balance-domains)."""
    w = np.zeros(len(domains))
    for d in np.unique(domains):
        m = domains == d
        w[m] = 0.5 / m.sum()
    return w


def domain_class_sample_weights(domains: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Probabilidad de muestreo por ventana con cada dominio la mitad de lo
    que se ve en una época y, dentro de él, todas sus clases por igual
    (--balance-classes, avances/25)."""
    w = np.zeros(len(domains))
    for d in np.unique(domains):
        in_domain = domains == d
        classes = np.unique(labels[in_domain])
        for c in classes:
            m = in_domain & (labels == c)
            w[m] = 0.5 / len(classes) / m.sum()
    return w


def class_weights(labels: np.ndarray, mode: str, sample_w: np.ndarray | None = None) -> torch.Tensor:
    """Pesos de la pérdida. Con sample_w (muestreo balanceado), las
    frecuencias son las que efectivamente ve el modelo, no las del índice.
    mode "none": sin pesos, cuando el muestreo ya balancea las clases."""
    sample_w = np.ones(len(labels)) if sample_w is None else sample_w
    freq = np.bincount(labels, weights=sample_w, minlength=N_CLASSES)
    seen = freq > 0   # en la ablación por dominio faltan las clases del otro
    w = np.zeros(N_CLASSES)
    w[seen] = {"inv": 1.0 / freq[seen], "sqrt": 1.0 / np.sqrt(freq[seen]), "none": 1.0}[mode]
    w = w / ((w[labels] * sample_w).sum() / sample_w.sum())   # peso medio por ventana vista = 1
    return torch.tensor(w, dtype=torch.float32)


def build_model(args) -> UnifiedCNN:
    return UnifiedCNN(width=WIDTHS[args.width], feature=args.feature, pooling=args.pooling,
                      n_fft=args.n_fft, hop_length=args.hop, time_keep_blocks=args.time_keep_blocks,
                      specaug=args.specaug, n_fft2=getattr(args, "n_fft2", 0),
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
    pulm_auc = np.nanmean([m["auc_crackle"], m["auc_wheeze"]]) if pulm.any() else nan
    m["select_domain"] = float(np.nanmean([m["auc_murmur"], pulm_auc]))
    return m


def load_sources(args) -> dict[str, tuple]:
    """Fuentes de ventanas: nombre -> (índice, audio, offsets, carpeta de
    folds, entrena, sufijo). Las fuentes con el mismo sufijo se evalúan
    juntas, con roles <split><sufijo>. "main" es la del --data; "icbhi"
    (ventanas pulmonares del índice unificado, folds unificados) solo evalúa
    con --data sprsound (test cruzado) y también entrena con --data joint,
    con el sufijo _icbhi. Con --data circor_spr, "main" son las ventanas
    cardíacas del índice unificado y "spr" las de SPRSound, sin sufijo."""
    unified = pd.read_csv(unified_windows_index(args.win_sec))
    if args.data == "unified":
        if args.domain != "both":
            unified = unified[unified["domain"] == args.domain].reset_index(drop=True)
        return {"main": (unified, *open_audio_cache(), UNIFIED_KFOLD_DIR, True, "")}
    spr = (pd.read_csv(sprsound_windows_index(args.win_sec)),
           *open_audio_cache(SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV), SPRSOUND_KFOLD_DIR, True)
    if args.data == "circor_spr":
        cardiac = unified.query("domain == 'cardiac'").reset_index(drop=True)
        return {"main": (cardiac, *open_audio_cache(), UNIFIED_KFOLD_DIR, True, ""), "spr": (*spr, "")}
    icbhi = unified.query("domain == 'pulmonary'").reset_index(drop=True)
    return {"main": (*spr, ""),
            "icbhi": (icbhi, *open_audio_cache(), UNIFIED_KFOLD_DIR, args.data == "joint", "_icbhi")}


def concat_dataset(parts: list[tuple[str, pd.DataFrame]], sources: dict, augment: bool = False) -> ConcatDataset:
    """Ventanas de varias fuentes, en el orden de parts (el de las predicciones)."""
    return ConcatDataset([UnifiedWindowDataset(df, sources[n][1], sources[n][2], augment=augment) for n, df in parts])


def val_metrics(val_parts: list[tuple[str, pd.DataFrame]], sources: dict, probs: np.ndarray) -> dict:
    """Métricas por grupo de fuentes (mismo sufijo): las del grupo sin
    sufijo sin prefijo, las demás con prefijo (icbhi_). Con más de un grupo
    (--data joint), cada criterio de selección es el promedio de los
    grupos, para que el checkpoint no favorezca a la fuente con más
    ventanas."""
    groups, start = {}, 0
    for name, df in val_parts:
        labels, parts = groups.setdefault(sources[name][5], ([], []))
        labels.append(df["label"].to_numpy())
        parts.append(probs[start:start + len(df)])
        start += len(df)
    out = {}
    for suffix, (labels, parts) in groups.items():
        m = window_metrics(np.concatenate(labels), np.concatenate(parts))
        out.update({(f"{suffix[1:]}_{k}" if suffix else k): v for k, v in m.items()})
    if len(groups) > 1:
        for c in CRITERIA:
            out[c] = float(np.mean([out[f"{s[1:]}_{c}" if s else c] for s in groups]))
    return out


def run_fold(i: int, sources: dict, args, ckpt_dir: Path, out_dir: Path) -> dict:
    torch.manual_seed(SEED + i)
    np.random.seed(SEED + i)
    split = {}   # (fuente, split) -> ventanas
    for name, (windows, _, _, fold_dir, _, _) in sources.items():
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
    if args.balance_domains or args.balance_classes:
        sample_w = (domain_class_sample_weights(tr["domain"].to_numpy(), tr["label"].to_numpy())
                    if args.balance_classes else domain_sample_weights(tr["domain"].to_numpy()))
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

    def ckpt_path(c: str) -> Path:   # el principal conserva el nombre de siempre
        return ckpt_dir / (f"fold{i}_best.pt" if c == args.select else f"fold{i}_best_{c}.pt")

    history, best, since_best = [], {c: -1.0 for c in CRITERIA}, 0
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
        m = val_metrics(val_parts, sources, val_probs)
        history.append({"epoch": epoch, "train_loss": total / n, "val_loss": val_loss, **m,
                        "sec": time.perf_counter() - t0})
        improved = [c for c in CRITERIA if m[c] > best[c]]
        marks = "".join(" *" if c == args.select else " +" for c in improved)   # *: principal, +: el otro
        print(f"  [{epoch:02d}] loss {total / n:.4f}/{val_loss:.4f} | AUC murm {m['auc_murmur']:.3f} "
              f"crk {m['auc_crackle']:.3f} whz {m['auc_wheeze']:.3f} | dominio {m['domain_acc']:.3f} "
              f"| bal6 {m['bal_acc_6']:.3f} | sel A {m['select']:.3f} B {m['select_domain']:.3f}"
              + (f" | ICBHI crk {m['icbhi_auc_crackle']:.3f} whz {m['icbhi_auc_wheeze']:.3f}" if "icbhi_select" in m else "")
              + f" | {time.perf_counter() - t0:.0f} s{marks}", flush=True)
        for c in improved:
            best[c] = m[c]
            torch.save(model.state_dict(), ckpt_path(c))
        since_best = 0 if improved else since_best + 1
        if since_best >= args.patience:
            print(f"  early stopping (ningún criterio mejoró en {args.patience} épocas)")
            break
    pd.DataFrame(history).to_csv(out_dir / f"history_fold{i}.csv", index=False)

    # Probabilidades del mejor checkpoint de cada criterio en val y test.
    # Las fuentes con el mismo sufijo van juntas; val solo de las que entrenan.
    groups = {}
    for name, src in sources.items():
        groups.setdefault(src[5], []).append(name)
    evals = []   # (rol, [(fuente, ventanas)])
    for suffix, names in groups.items():
        for s in ("val", "test"):
            parts = [(n, split[n, s]) for n in names if s == "test" or sources[n][4]]
            if parts:
                evals.append((f"{s}{suffix}", parts))
    res = {"fold": i, "epochs_run": len(history)}
    for c in (args.select, *[c for c in CRITERIA if c != args.select]):
        model.load_state_dict(torch.load(ckpt_path(c), map_location="cpu"))
        test_m, rows = {}, []
        for role, parts in evals:
            probs = predict(model, DataLoader(concat_dataset(parts, sources),
                                              batch_size=256, shuffle=False, collate_fn=collate))
            df = pd.concat([d for _, d in parts])
            part = df[["file_id", "subject_id", "domain", "window_idx", "start_sec", "label"]].copy()
            part["fold"], part["role"] = i, role
            for k, v in derived_probs(probs).items():
                part[k] = v
            rows.append(part)
            if role.startswith("test"):
                test_m[role] = window_metrics(df["label"].to_numpy(), probs)
        main_crit = c == args.select
        pd.concat(rows).to_csv(out_dir / (f"window_probs_fold{i}.csv" if main_crit
                                          else f"window_probs_fold{i}_{c}.csv"), index=False)
        best_row = max(history, key=lambda h: h[c])
        prefix = "" if main_crit else f"{c}_"   # el principal, con las columnas de siempre
        res.update({f"{prefix}best_epoch": best_row["epoch"],
                    **{f"{prefix}val_{k}": v for k, v in best_row.items() if k not in ("epoch", "sec")},
                    **{f"{prefix}{role}_{k}": v for role, m in test_m.items() for k, v in m.items()}})
        t = test_m["test"]
        print(f"  Fold {i} — {c}{' (principal)' if main_crit else ''}: mejor época {best_row['epoch']} | "
              f"val {best_row[c]:.4f} | test: AUC murm {t['auc_murmur']:.3f} crk {t['auc_crackle']:.3f} "
              f"whz {t['auc_wheeze']:.3f} dominio {t['domain_acc']:.3f}", flush=True)
    (out_dir / f"result_fold{i}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", choices=list(WIDTHS), default="base")
    parser.add_argument("--feature", choices=FEATURES, default="mfcc")
    parser.add_argument("--pooling", choices=POOLING_TYPES, default="avg")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--n-fft2", type=int, default=0, help="segundo canal de entrada con este n_fft (0: uno solo)")
    parser.add_argument("--hop", type=int, default=160, help="muestras a 4 kHz (160 = 40 ms)")
    parser.add_argument("--time-keep-blocks", type=int, default=0, choices=[0, 1, 2],
                        help="últimos bloques que reducen solo frecuencia")
    parser.add_argument("--specaug", type=int, default=0, help="máscaras de frecuencia y de tiempo por ejemplo")
    parser.add_argument("--class-weights", choices=["inv", "sqrt", "none"], default="inv")
    parser.add_argument("--balance-domains", action="store_true",
                        help="muestreo con la mitad de cada dominio por época")
    parser.add_argument("--balance-classes", action="store_true",
                        help="muestreo con la mitad de cada dominio y sus clases por igual (usar con --class-weights none)")
    parser.add_argument("--domain", choices=["both", "cardiac", "pulmonary"], default="both",
                        help="ablación: entrenar y evaluar solo con las ventanas de un dominio")
    parser.add_argument("--data", choices=["unified", "sprsound", "joint", "circor_spr"], default="unified",
                        help="sprsound: solo SPRSound (implica --domain pulmonary) + test cruzado en ICBHI; "
                             "joint: SPRSound + ICBHI pulmonar en train/val, test de cada uno por separado; "
                             "circor_spr: CirCor (cardíaco) + SPRSound (pulmonar), 6 clases")
    parser.add_argument("--select", choices=CRITERIA, default="select",
                        help="criterio principal del checkpoint: select (A, media de los tres AUC) o "
                             "select_domain (B, media por dominio); el otro se guarda igual")
    parser.add_argument("--win-sec", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.data in ("sprsound", "joint"):
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
          f"({n_params} parámetros), pesos {args.class_weights}, datos {args.data}, dominio {args.domain}, "
          f"selección {args.select}, folds {args.folds}", flush=True)

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
