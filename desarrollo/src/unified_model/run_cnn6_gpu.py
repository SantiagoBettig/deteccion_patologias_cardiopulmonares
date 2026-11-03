"""
unified_model/run_cnn6_gpu.py
================================
Fine-tuning de la CNN6 de PANNs como modelo de 6 clases CirCor + SPRSound
(avances/25 §8 paso 2: techo de desempeño con la mejor receta), pensado
para Google Colab (notebook colab_cnn6_circor_spr.ipynb). Lee el paquete de
pack_cnn6_colab.py: el audio a 4 kHz entero en la memoria de la GPU, y el
espectrograma se calcula por batch en la GPU.

Receta de cs_wide (run_kfold --data circor_spr), con la red de la variante
B de avances/17:
- 6 clases softmax, CrossEntropyLoss con pesos inv (run_kfold.class_weights),
  mismos folds, checkpoint por select_domain (B, principal) y select (A),
  early stopping con --patience épocas sin que mejore ningún criterio.
- Augmentation de cs_wide sobre la onda (ganancia ±3 dB + ruido gaussiano,
  std 0.005) y el enmascarado de la variante B sobre el log-Mel (tiempo
  hasta 40 frames, frecuencia hasta 6 bandas, dentro de las que tienen
  contenido).
- CNN6 con bloques 3-4 + fc1 entrenables (lr 1e-4), cabeza nueva 512 -> 6
  (lr 1e-3), BatchNorm con estadísticas de AudioSet congeladas, dropout de
  PANNs (pulmonary_adventitious_model.run_finetune_gpu.FineTuneCnn6).

Frontends (--frontend), los dos con 64 bandas y 100 frames por segundo:
- "panns32k": el de PANNs (32 kHz, STFT 1024 / hop 320, Mel 50-14000 Hz,
  dB absolutos), el de avances/15-17. El audio tiene contenido solo hasta
  2 kHz, así que en lugar de remuestrear se hace la STFT a 4 kHz con 128 /
  hop 40 (mismos bins de 31.25 Hz y mismos frames) y la potencia por 64
  (= 8², el factor de la DFT con 8 veces más muestras), con el banco Mel de
  32 kHz restringido a los bins de 0 a 2 kHz. Control contra librosa
  (remuestreo + STFT): menos de 0.1 dB en las bandas 0-27; las 2 del borde
  de 2 kHz difieren por el filtro antialias del remuestreo, y las de arriba
  quedan en -100 dB (silencio). ~30 bandas con contenido.
- "ours64": 64 bandas Mel de 50 a 2000 Hz sobre los 4 kHz (STFT 256 / hop
  40). Más resolución en bajas frecuencias, que es donde está el murmullo,
  pero no es lo que vio la red en el preentrenamiento.
Sin z-score: bn0 de PANNs normaliza cada banda con las estadísticas de
AudioSet. Sin pasa-altos (el audio es el de cs_wide; el cardíaco lo
necesita).

Pensado para Colab, que corta sesiones: guarda el estado completo al final
de cada época en --out-dir (Drive) y los resultados de cada fold apenas
termina. Volver a correr el mismo comando retoma.

Salidas en <out-dir>/<run>/ (los nombres de run_kfold, para que
import_colab_run.py las lleve a reports/ y evaluate_circor_spr /
filter_variants las lean sin cambios):
    config.json, history_fold{i}.csv, result_fold{i}.json, per_fold.csv
    window_probs_fold{i}.csv, window_probs_fold{i}_select.csv   ventanas del índice, val y test
    full_spr_fold{i}.csv, full_spr_fold{i}_select.csv           todas las ventanas de SPRSound, val y test
    fold{i}_best.pt, fold{i}_best_select.pt

Uso:
    python unified_model/run_cnn6_gpu.py --package-dir <carpeta> --out-dir <carpeta> --frontend panns32k --folds 0 3
    # prueba rápida en CPU:
    python unified_model/run_cnn6_gpu.py --package-dir ... --out-dir ... --frontend ours64 --folds 0 --epochs 1 --limit 150
"""

import argparse
import json
import time

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pulmonary_adventitious_model.panns_cnn6 import load_pretrained_cnn6
from pulmonary_adventitious_model.run_finetune_gpu import FineTuneCnn6
from unified_model.dataset import derived_probs
from unified_model.evaluate_circor_spr import FULL_COLS
from unified_model.model import N_CLASSES
from unified_model.run_kfold import CRITERIA, K, SEED, class_weights, window_metrics

SR = 4000
WIN = 4 * SR
FRONTENDS = {
    "panns32k": {"n_fft": 128, "hop": 40, "power_scale": 64.0, "content_bands": 30,
                 "mel": lambda: librosa.filters.mel(sr=32000, n_fft=1024, n_mels=64, fmin=50, fmax=14000)[:, :64]},
    "ours64":   {"n_fft": 256, "hop": 40, "power_scale": 1.0, "content_bands": 64,
                 "mel": lambda: librosa.filters.mel(sr=SR, n_fft=256, n_mels=64, fmin=50, fmax=2000)},
}
FIRST_TRAINABLE = 2     # variante B: bloques 3-4 + fc1
LR_PRETRAINED, LR_HEAD, WEIGHT_DECAY = 1e-4, 1e-3, 1e-4
BATCH_SIZE, EVAL_BATCH = 64, 256
GAIN_DB, NOISE_STD = 3.0, 0.005
TIME_MASK_MAX, FREQ_MASK_MAX = 40, 6
PAD_DB = -100.0
WINDOW_OUT_COLS = ["file_id", "subject_id", "domain", "window_idx", "start_sec", "label"]
FULL_OUT_COLS = ["file_id", "subject_id", "window_idx", "start_sec"]


# ── Datos y features, todo en el dispositivo ─────────────────────────────────
class AudioBank:
    """El audio int16 entero en el dispositivo; arma batches de ventanas de
    4 s a partir de (inicio absoluto, largo), con ceros después del largo."""

    def __init__(self, audio: np.ndarray, device):
        self.audio = torch.from_numpy(audio).to(device)
        self.arange = torch.arange(WIN, device=device)

    def batch(self, starts: np.ndarray, lengths: np.ndarray) -> torch.Tensor:
        s = torch.tensor(starts, device=self.audio.device)[:, None]
        n = torch.tensor(lengths, device=self.audio.device)[:, None]
        idx = torch.clamp(s + self.arange, max=len(self.audio) - 1)
        return self.audio[idx].float() / 32768.0 * (self.arange < n)


class LogMel(nn.Module):
    """(B, n) onda a 4 kHz -> (B, 1, frames, 64) en dB, la entrada de la CNN6."""

    def __init__(self, frontend: str):
        super().__init__()
        cfg = FRONTENDS[frontend]
        self.n_fft, self.hop, self.scale = cfg["n_fft"], cfg["hop"], cfg["power_scale"]
        self.register_buffer("window", torch.hann_window(self.n_fft, periodic=True))
        self.register_buffer("mel", torch.tensor(cfg["mel"](), dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(y, self.n_fft, self.hop, window=self.window, center=True, pad_mode="reflect",
                          return_complex=True).abs() ** 2 * self.scale         # (B, bins, T)
        mel = self.mel @ spec[:, :self.mel.shape[1]]
        return (10.0 * torch.log10(torch.clamp(mel, min=1e-10))).transpose(1, 2)[:, None]


def augment_wave(y: torch.Tensor) -> torch.Tensor:
    gain = 10.0 ** ((torch.rand(len(y), 1, device=y.device) * 2 - 1) * GAIN_DB / 20.0)
    return torch.clamp(y * gain + torch.randn_like(y) * NOISE_STD, -1.0, 1.0)


def augment_spec(x: torch.Tensor, content_bands: int) -> torch.Tensor:
    """Una máscara de tiempo y una de frecuencia por ejemplo (variante B)."""
    B, _, T, F = x.shape
    dev = x.device
    w = torch.randint(0, TIME_MASK_MAX + 1, (B, 1), device=dev)
    t0 = (torch.rand(B, 1, device=dev) * (T - w + 1)).long()
    t = torch.arange(T, device=dev)[None]
    f_w = torch.randint(0, FREQ_MASK_MAX + 1, (B, 1), device=dev)
    f0 = (torch.rand(B, 1, device=dev) * (content_bands - f_w + 1)).long()
    f = torch.arange(F, device=dev)[None]
    mask = ((t >= t0) & (t < t0 + w))[:, :, None] | ((f >= f0) & (f < f0 + f_w))[:, None, :]
    return x.masked_fill(mask[:, None], PAD_DB)


class Cnn6Classifier(nn.Module):
    def __init__(self, weights: Path, frontend: str):
        super().__init__()
        self.frontend = LogMel(frontend)
        self.net = FineTuneCnn6(load_pretrained_cnn6(weights), FIRST_TRAINABLE)
        self.net.head = nn.Linear(512, N_CLASSES)

    def train(self, mode: bool = True):
        super().train(mode)
        self.net.train(mode)   # deja la BatchNorm congelada
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = torch.full((len(x),), x.shape[2], device=x.device)
        return self.net(x, lengths)


def predict(model, bank: AudioBank, starts: np.ndarray, lengths: np.ndarray, amp: bool) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(starts), EVAL_BATCH):
            x = model.frontend(bank.batch(starts[s:s + EVAL_BATCH], lengths[s:s + EVAL_BATCH]))
            with torch.autocast(device_type=x.device.type, enabled=amp):
                logits = model(x)
            out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out)


# ── Un fold, con guardado/retome por época ───────────────────────────────────
def run_fold(i: int, args, windows: pd.DataFrame, full: pd.DataFrame, bank: AudioBank, out_dir: Path,
             device, amp: bool) -> None:
    result_json = out_dir / f"result_fold{i}.json"
    if result_json.exists():
        print(f"[fold {i}] ya terminado en una sesión anterior — se omite")
        return
    torch.manual_seed(SEED + i)
    np.random.seed(SEED + i)
    fold = pd.read_csv(args.package_dir / "folds" / f"fold{i}.csv").set_index("subject_id")["split"]
    w = windows.assign(split=windows["subject_id"].map(fold).to_numpy())
    fw = full.assign(split=full["subject_id"].map(fold).to_numpy())
    tr, va = w[w.split == "train"], w[w.split == "val"]
    y_tr, y_va = tr["label"].to_numpy(), va["label"].to_numpy()

    model = Cnn6Classifier(args.package_dir / "Cnn6_mAP=0.343.pth", args.frontend).to(device)
    optimizer = torch.optim.Adam([
        {"params": model.net.pretrained_params(), "lr": LR_PRETRAINED},
        {"params": model.net.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=amp)
    weights = class_weights(y_tr, "inv")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    content_bands = FRONTENDS[args.frontend]["content_bands"]

    def ckpt_path(c: str) -> Path:
        return out_dir / (f"fold{i}_best.pt" if c == args.select else f"fold{i}_best_{c}.pt")

    last_pt = out_dir / f"fold{i}_last.pt"
    start_epoch, best, since_best, history = 1, {c: -1.0 for c in CRITERIA}, 0, []
    if last_pt.exists():
        state = torch.load(last_pt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best, since_best, history = state["epoch"] + 1, state["best"], state["since_best"], state["history"]
        np.random.set_state(state["np_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"])
        print(f"[fold {i}] retomando desde la época {start_epoch}")

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[fold {i}] ventanas train {len(tr)} | val {len(va)} | test {(w.split == 'test').sum()} | "
          f"SPRSound completas val+test {fw.split.isin(['val', 'test']).sum()} | "
          f"entrenables {n_trainable / 1e6:.2f} M | pesos {np.round(weights.numpy(), 2).tolist()}", flush=True)

    tr_starts, tr_len = tr["abs_start"].to_numpy(), np.full(len(tr), WIN)
    va_starts, va_len = va["abs_start"].to_numpy(), np.full(len(va), WIN)
    stopped = since_best >= args.patience
    for epoch in range(start_epoch, args.epochs + 1):
        if stopped:
            break
        t0 = time.perf_counter()
        model.train()
        perm = np.random.permutation(len(tr))
        total = 0.0
        for s in range(0, len(perm), BATCH_SIZE):
            b = perm[s:s + BATCH_SIZE]
            with torch.no_grad():
                x = augment_spec(model.frontend(augment_wave(bank.batch(tr_starts[b], tr_len[b]))), content_bands)
            y = torch.from_numpy(y_tr[b]).to(device)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(model(x).float(), y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(b)
        val_probs = predict(model, bank, va_starts, va_len, amp)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.tensor(val_probs).clamp_min(1e-8)),
                                                torch.tensor(y_va), weight=weights))
        m = window_metrics(y_va, val_probs)
        history.append({"epoch": epoch, "train_loss": total / len(perm), "val_loss": val_loss, **m,
                        "sec": time.perf_counter() - t0})
        improved = [c for c in CRITERIA if m[c] > best[c]]
        marks = "".join(" *" if c == args.select else " +" for c in improved)   # *: principal, +: el otro
        print(f"[fold {i}] [{epoch:02d}] loss {total / len(perm):.4f}/{val_loss:.4f} | AUC murm {m['auc_murmur']:.3f} "
              f"crk {m['auc_crackle']:.3f} whz {m['auc_wheeze']:.3f} | dominio {m['domain_acc']:.3f} "
              f"| bal6 {m['bal_acc_6']:.3f} | sel A {m['select']:.3f} B {m['select_domain']:.3f} "
              f"| {history[-1]['sec']:.0f} s{marks}", flush=True)
        for c in improved:
            best[c] = m[c]
            torch.save(model.state_dict(), ckpt_path(c))
        since_best = 0 if improved else since_best + 1
        stopped = since_best >= args.patience
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "since_best": since_best, "history": history,
                    "np_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state() if device.type == "cuda" else None}, last_pt)
        if stopped:
            print(f"[fold {i}] early stopping (ningún criterio mejoró en {args.patience} épocas)")
    pd.DataFrame(history).to_csv(out_dir / f"history_fold{i}.csv", index=False)

    # Probabilidades del mejor checkpoint de cada criterio: ventanas del
    # índice (val y test) y todas las de SPRSound (val y test).
    res = {"fold": i, "epochs_run": len(history)}
    for c in (args.select, *[c for c in CRITERIA if c != args.select]):
        model.load_state_dict(torch.load(ckpt_path(c), map_location=device))
        suffix = "" if c == args.select else f"_{c}"
        rows, test_m = [], None
        for role in ("val", "test"):
            part = w[w.split == role]
            probs = predict(model, bank, part["abs_start"].to_numpy(), np.full(len(part), WIN), amp)
            out = part[WINDOW_OUT_COLS].assign(fold=i, role=role)
            for k, v in derived_probs(probs).items():
                out[k] = v
            rows.append(out)
            if role == "test":
                test_m = window_metrics(part["label"].to_numpy(), probs)
        pd.concat(rows).to_csv(out_dir / f"window_probs_fold{i}{suffix}.csv", index=False)
        rows = []
        for role in ("val", "test"):
            part = fw[fw.split == role]
            probs = derived_probs(predict(model, bank, part["abs_start"].to_numpy(), part["length"].to_numpy(), amp))
            out = part[FULL_OUT_COLS].assign(fold=i, role=role)
            for k in FULL_COLS:
                out[k] = probs[k]
            rows.append(out)
        pd.concat(rows).to_csv(out_dir / f"full_spr_fold{i}{suffix}.csv", index=False)
        best_row = max(history, key=lambda h: h[c])
        prefix = "" if c == args.select else f"{c}_"
        res.update({f"{prefix}best_epoch": best_row["epoch"],
                    **{f"{prefix}val_{k}": v for k, v in best_row.items() if k not in ("epoch", "sec")},
                    **{f"{prefix}test_{k}": v for k, v in test_m.items()}})
        print(f"[fold {i}] {c}{' (principal)' if c == args.select else ''}: mejor época {best_row['epoch']} | "
              f"val {best_row[c]:.4f} | test: AUC murm {test_m['auc_murmur']:.3f} crk {test_m['auc_crackle']:.3f} "
              f"whz {test_m['auc_wheeze']:.3f} dominio {test_m['domain_acc']:.3f}", flush=True)
    result_json.write_text(json.dumps(res, indent=2, default=float))
    last_pt.unlink(missing_ok=True)   # el fold terminó: el estado intermedio ya no hace falta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, required=True, help="Carpeta con el contenido del zip")
    parser.add_argument("--out-dir", type=Path, required=True, help="Dónde guardar (en Colab: una carpeta de Drive)")
    parser.add_argument("--frontend", choices=list(FRONTENDS), required=True)
    parser.add_argument("--run-name", help="Por defecto, cnn6_<frontend>")
    parser.add_argument("--select", choices=CRITERIA, default="select_domain")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(K)))
    parser.add_argument("--limit", type=int, default=None,
                        help="Solo para pruebas: N grabaciones al azar por dominio")
    args = parser.parse_args()

    run = args.run_name or f"cnn6_{args.frontend}"
    out_dir = args.out_dir / run
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    if device.type == "cpu":
        torch.set_flush_denormal(True)   # las bandas vacías de panns32k (-100 dB) dan denormales

    pkg = args.package_dir
    recs = pd.read_csv(pkg / "recordings.csv")
    if args.limit:
        recs = recs.groupby("domain", group_keys=False).sample(args.limit, random_state=0)
    offset = recs.set_index("file_id")["offset"]
    windows = pd.read_csv(pkg / "windows.csv").query("file_id in @offset.index").reset_index(drop=True)
    windows["abs_start"] = windows["file_id"].map(offset) + windows["start_sample"]
    full = pd.read_csv(pkg / "full_windows_spr.csv").query("file_id in @offset.index").reset_index(drop=True)
    full["abs_start"] = full["file_id"].map(offset) + full["start_sample"]
    bank = AudioBank(np.load(pkg / "audio_4k_int16.npy"), device)

    n_params = sum(p.numel() for p in Cnn6Classifier(pkg / "Cnn6_mAP=0.343.pth", args.frontend).parameters())
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "folds"}
    config.update({"model": "cnn6", "data": "circor_spr", "first_trainable": FIRST_TRAINABLE, "n_params": n_params})
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"Run {run}: CNN6 (bloques 3-4 + fc1, {n_params / 1e6:.2f} M parámetros), frontend {args.frontend}, "
          f"selección {args.select}, folds {args.folds} | {len(windows)} ventanas del índice, "
          f"{len(full)} de SPRSound completas | {device}"
          f"{' (' + torch.cuda.get_device_name(0) + ')' if amp else ''}", flush=True)

    for i in args.folds:
        run_fold(i, args, windows, full, bank, out_dir, device, amp)
    per_fold = pd.DataFrame([json.loads(p.read_text()) for p in sorted(out_dir.glob("result_fold*.json"))])
    per_fold.to_csv(out_dir / "per_fold.csv", index=False)
    print(f"\nFolds terminados: {per_fold['fold'].tolist()}")
    cols = [c for c in per_fold.columns if c.startswith("test_")]
    print(f"\n{per_fold[cols].agg(['mean', 'std']).T.round(4).to_string()}")


if __name__ == "__main__":
    main()
