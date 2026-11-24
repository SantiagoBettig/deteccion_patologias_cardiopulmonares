"""
unified_model/device_intervention.py
=======================================
Prueba causal de la hipótesis del estetoscopio (avances/28 §10, avances/29):
si la decisión de dominio de los modelos depende de la firma espectral del
Yunting II (el estetoscopio de SPRSound, que corta debajo de ~100 Hz),
entonces cambiar SOLO esa firma, sin tocar los modelos, tiene que mover la
decisión.

Transformaciones (sobre el audio de las caches, ya preprocesado; después se
vuelve a normalizar a RMS 0.1, como en audio_ops.preprocess):
- "hp100": pasa-altos Butterworth de orden 4 a 100 Hz (fase cero). Prueba
  solo la falta de graves.
- "eq_spr": ecualización al espectro del Yunting. Para cada grupo (base y
  equipo o modo) la ganancia es la diferencia entre el espectro mediano de
  SPRSound y el del grupo (dB, suavizada en 5 bins y limitada a ±40 dB),
  aplicada con un FIR de fase lineal (firwin2, 1025 coeficientes).
- "eq_akg" (sentido inverso, solo SPRSound): ecualiza las grabaciones de
  test de SPRSound al espectro de ICBHI AKG C417L, el equipo pulmonar
  externo más común. Cada modelo de fold predice solo los pacientes de test
  de su fold (no vio esas grabaciones).

Predicciones esperadas si la hipótesis es cierta:
- pulmonares externas (ICBHI, KAUH, HLS-LS) con hp100 o eq_spr → pasan a
  pulmonar;
- cardíacas externas (PASCAL B, HLS-HS) con eq_spr → pasan a pulmonar
  (el atajo manda aunque el sonido sea cardíaco);
- SPRSound con eq_akg → pasa a cardíaco.

Modelos: cnn6_ours64, cs_wide y el gate k-fold del pipeline (solo el
dominio importa acá; igual se guardan todas las salidas). Las ventanas y la
carga de modelos son las de external_inference.py.

Salida: reports/external_validation/intervention/
    window_probs/<transformación>__<base>__<modelo>.csv
    domain_summary.csv     fracción a cardíaco por grupo: original contra transformada
    task_auc.csv           AUC normal-anormal condicional en las pulmonares (¿se destruyó el contenido?)
    spectra.png            espectro de un grupo antes y después, contra SPRSound

Uso:
    python unified_model/device_intervention.py            # inferencia (con cache) y resumen
    python unified_model/device_intervention.py --summary  # solo el resumen
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.signal import butter, fftconvolve, firwin2, sosfiltfilt, welch
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (EXTERNAL_REPORTS_DIR, SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV,
                          SPRSOUND_KFOLD_DIR, SPRSOUND_RECORDINGS_CSV, external_paths)
from integration.evaluate_integration import load_recordings
from unified_model.dataset import derived_probs
from unified_model.evaluate_circor_spr import FULL_COLS
from unified_model.external_inference import gate_predict, load_dataset, load_model, window_table
from unified_model.run_cnn6_gpu import AudioBank, predict as predict_wave
from unified_model.run_kfold import K

SR = 4000
OUT = EXTERNAL_REPORTS_DIR / "intervention"
PROBS = OUT / "window_probs"
MODELS = ["cnn6_ours64", "cs_wide", "gate_kfold_cs"]
NPERSEG = 512
NUMTAPS = 1025
SHORT = {"AKG C417L Microphone": "AKG C417L", "3M Littmann Classic II SE": "Littmann Classic II SE",
         "3M Littmann 3200 Electronic": "Littmann 3200", "WelchAllyn Meditron Master Elite": "Meditron"}


# ── Grupos y espectros ────────────────────────────────────────────────────────
def groups(data: str, recs: pd.DataFrame) -> pd.Series:
    """Grupo (equipo / modo / conjunto) de cada grabación, alineado con recs."""
    if data == "ICBHI":
        eq = load_recordings().set_index("file_id")["equipment"].map(SHORT)
        return recs.file_id.map(eq).map(lambda e: f"ICBHI · {e}")
    if data == "HLS":
        s = pd.read_csv(external_paths("hls")[0]).set_index("file_id")["set"]
        return recs.file_id.map(s).map(lambda v: f"HLS · {v}")
    if data == "KAUH":
        m = pd.read_csv(external_paths("kauh")[0]).set_index("file_id")["modo"]
        return recs.file_id.map(m).map(lambda v: f"KAUH · {v}")
    if data == "PASCAL":
        return pd.Series("PASCAL B", index=recs.index)
    return pd.Series("SPRSound", index=recs.index)


def segment(audio: np.ndarray, r) -> np.ndarray:
    return np.asarray(audio[int(r.offset):int(r.offset) + int(r.n_samples)], dtype=np.float64) / 32768.0


def median_spectrum(audio: np.ndarray, recs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Espectro relativo mediano (dB) de las grabaciones dadas, 0-2000 Hz."""
    specs = []
    for r in recs.itertuples():
        y = segment(audio, r)
        if len(y) < NPERSEG:
            continue
        _, p = welch(y, SR, nperseg=NPERSEG)
        specs.append(10 * np.log10(p / p.sum() + 1e-12))
    f = np.linspace(0, SR / 2, NPERSEG // 2 + 1)
    return f, np.median(np.array(specs), axis=0)


def eq_filter(f: np.ndarray, source_db: np.ndarray, target_db: np.ndarray) -> np.ndarray:
    gain = target_db - source_db
    gain = np.convolve(gain, np.ones(5) / 5, mode="same")
    gain[f < 20] = gain[np.searchsorted(f, 20)]
    gain = np.clip(gain, -40, 40)
    return firwin2(NUMTAPS, f / (SR / 2), 10 ** (gain / 20))


def apply(y: np.ndarray, how: str, fir: np.ndarray | None) -> np.ndarray:
    if how == "hp100":
        y = sosfiltfilt(butter(4, 100, "highpass", fs=SR, output="sos"), y)
    else:
        y = fftconvolve(y, fir, mode="same")
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 1e-8:
        y = y * (0.1 / rms)
    return np.clip(y, -1.0, 1.0)


def transformed_audio(audio: np.ndarray, recs: pd.DataFrame, grp: pd.Series, how: str,
                      target: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, dict]:
    """Audio int16 compacto transformado (mismos offsets) y los filtros por grupo."""
    out = np.array(audio, dtype=np.int16, copy=True)
    firs = {}
    if how != "hp100":
        f, tgt = target
        for g in grp.unique():
            _, src = median_spectrum(audio, recs[grp == g])
            firs[g] = (eq_filter(f, src, tgt), src)
    for r, g in zip(recs.itertuples(), grp):
        y = apply(segment(audio, r), how, firs.get(g, (None,))[0])
        out[int(r.offset):int(r.offset) + int(r.n_samples)] = np.round(y * 32767).astype(np.int16)
    return out, firs


# ── Inferencia ────────────────────────────────────────────────────────────────
def infer(how: str, data: str, recs: pd.DataFrame, audio: np.ndarray, folds_of: pd.Series | None) -> None:
    """folds_of: para SPRSound, el fold en el que cada grabación es test (cada
    modelo predice solo las suyas); None = todas las grabaciones con los 5."""
    bank = AudioBank(audio, "cpu")
    for name in MODELS:
        path = PROBS / f"{how}__{data}__{name}.csv"
        if path.exists():
            continue
        gate = name == "gate_kfold_cs"
        win = window_table(recs, gate)
        parts = []
        t0 = time.perf_counter()
        for i in range(K):
            w = win if folds_of is None else win[win.file_id.map(folds_of) == i]
            starts, lengths = w["start"].to_numpy(), w["length"].to_numpy()
            model = load_model(name, i)
            part = w[["file_id", "window_idx", "start_sec"]].assign(fold=i)
            if gate:
                part["prob_cardiac"] = gate_predict(model, bank, starts, lengths)
            else:
                d = derived_probs(predict_wave(model, bank, starts, lengths, amp=False))
                for c in FULL_COLS:
                    part[c] = d[c]
            parts.append(part)
        pd.concat(parts).to_csv(path, index=False)
        print(f"  {how} · {data} · {name}: {sum(len(p) for p in parts)} ventanas, "
              f"{time.perf_counter() - t0:.0f} s", flush=True)


def sprsound_test() -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Grabaciones de SPRSound con el fold donde son test, en audio compacto."""
    spr = pd.read_csv(SPRSOUND_RECORDINGS_CSV)[["file_id", "subject_id"]]
    fold_of = {}
    for i in range(K):
        f = pd.read_csv(SPRSOUND_KFOLD_DIR / f"fold{i}.csv")
        for s in f.loc[f.split == "test", "subject_id"]:
            fold_of[s] = i
    spr["fold"] = spr.subject_id.map(fold_of)
    spr = spr.dropna(subset=["fold"])
    audio = np.load(SPRSOUND_AUDIO_CACHE_NPY, mmap_mode="r")
    off = pd.read_csv(SPRSOUND_AUDIO_OFFSETS_CSV).set_index("file_id")
    spr = spr.merge(off, left_on="file_id", right_index=True)
    chunks, new, pos = [], [], 0
    for r in spr.itertuples():
        chunks.append(np.asarray(audio[int(r.offset):int(r.offset) + int(r.n_samples)]))
        new.append(pos)
        pos += int(r.n_samples)
    spr = spr.assign(offset=new).reset_index(drop=True)
    return spr, np.concatenate(chunks), spr.set_index("file_id")["fold"].astype(int)


# ── Resumen ───────────────────────────────────────────────────────────────────
def recording_domain(path: Path) -> pd.DataFrame:
    """P(cardíaco) media por grabación y fold, y la del ensamble (fold 5)."""
    w = pd.read_csv(path)
    col = "prob_cardiac" if "prob_cardiac" in w else "p_cardiac"
    r = w.groupby(["file_id", "fold"])[col].mean().rename("p_cardiac").reset_index()
    ens = r.groupby("file_id").p_cardiac.mean().reset_index().assign(fold=K)
    return pd.concat([r, ens], ignore_index=True)


def summary(group_of: dict) -> None:
    rows, auc_rows = [], []
    for path in sorted(PROBS.glob("*.csv")):
        how, data, model = path.stem.split("__")
        new = recording_domain(path)
        if data == "SPRSound":
            orig = None
        else:
            orig_w = EXTERNAL_REPORTS_DIR / "window_probs" / f"{model}_{data}.csv"
            orig = recording_domain(orig_w)
        for tag, d in (("transformada", new), ("original", orig)):
            if d is None:
                continue
            d = d.assign(group=d.file_id.map(group_of[data]))
            ens = d[d.fold == K]
            per = d[d.fold < K].groupby(["group", "fold"]).p_cardiac.apply(lambda s: (s > 0.5).mean())
            for g, gd in ens.groupby("group"):
                rows.append({"transformacion": how, "base": data, "group": g, "modelo": model, "audio": tag,
                             "n": len(gd), "a_cardiaco_ens": (gd.p_cardiac > 0.5).mean(),
                             "a_cardiaco_folds": f"{per.loc[g].mean():.3f} ± {per.loc[g].std():.3f}"})
        # AUC normal-anormal condicional (contenido pulmonar conservado) en las pulmonares externas
        if model != "gate_kfold_cs" and data in ("ICBHI", "KAUH"):
            for tag, p in (("transformada", path),
                           ("original", EXTERNAL_REPORTS_DIR / "window_probs" / f"{model}_{data}.csv")):
                w = pd.read_csv(p)
                w = w.assign(abn=(1 - w.p_pulm_normal / (1 - w.p_cardiac).clip(lower=1e-6)).clip(0, 1))
                s = w.groupby("file_id").abn.mean()
                lab = (load_recordings().set_index("file_id") if data == "ICBHI"
                       else pd.read_csv(external_paths("kauh")[0]).set_index("file_id"))
                lab = lab.loc[s.index]
                y = ((lab.has_crackles + lab.has_wheezes) > 0).astype(int)
                auc_rows.append({"transformacion": how, "base": data, "modelo": model, "audio": tag,
                                 "auc_normal_anormal_cond_ens": roc_auc_score(y, s)})
    out = pd.DataFrame(rows)
    out = out.pivot_table(index=["transformacion", "base", "group", "modelo", "n"], columns="audio",
                          values="a_cardiaco_ens", aggfunc="first").reset_index()
    out.to_csv(OUT / "domain_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "domain_summary_long.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    print(f"\n== Fracción mandada a cardíaco (ensamble): original contra transformada ==\n{out.round(3).to_string(index=False)}")
    if auc_rows:
        a = pd.DataFrame(auc_rows).pivot_table(index=["transformacion", "base", "modelo"], columns="audio",
                                               values="auc_normal_anormal_cond_ens").reset_index()
        a.to_csv(OUT / "task_auc.csv", index=False)
        print(f"\n== AUC normal-anormal condicional (ensamble) ==\n{a.round(3).to_string(index=False)}")


def plot_domain() -> None:
    """domain.png: fracción mandada a cardíaco por grupo, original contra transformadas."""
    d = pd.read_csv(OUT / "domain_summary.csv")
    d = d[d.transformacion != "eq_akg"]
    order = ["PASCAL B", "HLS · HS", "HLS · LS", "ICBHI · AKG C417L", "ICBHI · Littmann Classic II SE",
             "ICBHI · Littmann 3200", "ICBHI · Meditron", "KAUH · campana", "KAUH · diafragma", "KAUH · extendido"]
    titles = {"cnn6_ours64": "CNN6", "cs_wide": "cs_wide", "gate_kfold_cs": "Gate del pipeline"}
    style = {"original": ("#555555", "o", "Original"), "hp100": ("#2a78d6", "s", "Pasa-altos 100 Hz"),
             "eq_spr": ("#eb6834", "^", "Ecualizada al Yunting II")}
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.6), sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, m in zip(axes, MODELS):
        s = d[d.modelo == m].drop_duplicates(["transformacion", "group"])
        orig = s[s.transformacion == "hp100"].set_index("group")["original"].reindex(order)
        ax.axhspan(y[1] - 0.5, y[0] + 0.5, color="#f1e3d6", zorder=0)   # grupos cardíacos
        ax.scatter(orig, y, **dict(zip(("color", "marker"), style["original"][:2])), s=40, zorder=3,
                   label=style["original"][2])
        for how in ("hp100", "eq_spr"):
            v = s[s.transformacion == how].set_index("group")["transformada"].reindex(order)
            c, mk, lab = style[how]
            ax.scatter(v, y, color=c, marker=mk, s=40, zorder=3, label=lab)
        ax.set_xlim(-0.05, 1.05)
        ax.set_title(titles[m], loc="left", fontweight="bold")
        ax.set_xlabel("Fracción clasificada como cardíaca")
        ax.grid(axis="x", alpha=0.4)
    axes[0].set_yticks(y, ["PASCAL B (cardíaca)", "HLS · HS (cardíaca)", "HLS · LS"] + order[3:])
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "domain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--plot", action="store_true", help="solo regenera domain.png desde el CSV")
    args = parser.parse_args()
    if args.plot:
        plot_domain()
        return
    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_flush_denormal(True)
    PROBS.mkdir(parents=True, exist_ok=True)

    spr, spr_audio, spr_fold = sprsound_test()
    group_of = {"SPRSound": pd.Series("SPRSound", index=spr.file_id)}
    target_spr = median_spectrum(spr_audio, spr)
    datasets = {d: load_dataset(d) for d in ("ICBHI", "KAUH", "HLS", "PASCAL")}
    hls_sets = pd.read_csv(external_paths("hls")[0]).set_index("file_id")["set"]
    for d, (recs, audio) in datasets.items():
        if d == "HLS":   # solo HS y LS (las mezclas no entran)
            keep = recs.file_id.map(hls_sets).isin(["HS", "LS"]).to_numpy()
            recs = recs[keep].reset_index(drop=True)
            datasets[d] = (recs, audio)
        group_of[d] = pd.Series(groups(d, recs).to_numpy(), index=recs.file_id)

    if not args.summary:
        icbhi_recs, icbhi_audio = datasets["ICBHI"]
        akg = icbhi_recs[group_of["ICBHI"].to_numpy() == "ICBHI · AKG C417L"]
        target_akg = median_spectrum(icbhi_audio, akg)
        fig, ax = plt.subplots(figsize=(7, 3.8))
        f = target_spr[0]
        ax.plot(f[1:], target_spr[1][1:], color="#eb6834", lw=2, label="SPRSound (Yunting II), objetivo")
        for d, (recs, audio) in datasets.items():
            for how in ("hp100", "eq_spr"):
                new_audio, firs = transformed_audio(audio, recs, group_of[d].reset_index(drop=True), how, target_spr)
                if how == "eq_spr" and d == "KAUH":
                    _, before = median_spectrum(audio, recs[group_of[d].to_numpy() == "KAUH · diafragma"])
                    _, after = median_spectrum(new_audio, recs[group_of[d].to_numpy() == "KAUH · diafragma"])
                    ax.plot(f[1:], before[1:], color="#2a78d6", lw=1.5, ls="--", label="KAUH diafragma, original")
                    ax.plot(f[1:], after[1:], color="#2a78d6", lw=2, label="KAUH diafragma, ecualizada")
                infer(how, d, recs, new_audio, None)
        new_audio, _ = transformed_audio(spr_audio, spr, pd.Series("SPRSound", index=spr.index), "eq_akg",
                                         target_akg)
        infer("eq_akg", "SPRSound", spr, new_audio, spr_fold)
        ax.set_xscale("log")
        ax.set_xlim(20, 2000)
        ax.set_ylim(-75, -5)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Potencia relativa (dB)")
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8, frameon=False)
        ax.set_title("Ecualización al espectro del Yunting II (ejemplo)", loc="left", fontweight="bold")
        fig.savefig(OUT / "spectra.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    summary(group_of)
    plot_domain()
    print(f"\n✓ {OUT}")


if __name__ == "__main__":
    main()
