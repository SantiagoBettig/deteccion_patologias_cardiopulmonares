"""
eda/sprsound_preprocessing_check.py
======================================
Paso 2 del plan de SPRSound (avances/22 §8): ¿tiene sentido aplicarle el
mismo preprocesamiento y la misma regla de ventanas que a ICBHI?

Usa las tablas de eda/explore_sprsound.py (reports/sprsound/) y deja todo en
reports/sprsound/preprocessing/. Las grabaciones "Poor Quality" quedan
afuera.

Qué mira:
    1. Espectro nativo (8 kHz) por tipo de evento: cuánta energía, y cuánta
       diferencia entre clases, cae por encima de 2048 Hz (lo que borra el
       pasa-bajos del pipeline).
    2. Espectro preprocesado (4 kHz) de SPRSound contra los ciclos de ICBHI,
       por clase: corrimiento de dominio por equipo.
    3. Nivel y recorte: rms_normalize lleva cada grabación a RMS 0.1 y
       recorta a ±1; si los crackles tienen picos > 10× el RMS, se recortan.
    4. Huecos sin anotar (los eventos cubren 34-76% de cada grabación):
       duración, nivel y curtosis en banda 150-1800 Hz (proxy de
       transitorios tipo crackle) de los huecos, contra eventos normales y
       con crackle. Si los huecos de grabaciones DAS se parecen a los
       eventos con crackle más que los huecos de grabaciones Normal, hay
       crackles sin marcar en los huecos.
    5. Ventanas de 4 s (paso 2 s) con la regla de make_unified_windows.py,
       tratando cada evento como un ciclo: clases, descartadas, borderline,
       umbral 0.5 vs 1 s, y cuántas tocan estridor.
    6. Espectrogramas de ejemplo con los eventos marcados.

Uso:
    python eda/sprsound_preprocessing_check.py [--max-recs N]
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import AUDIO_DIR_V2, PULMONARY_CYCLES_INDEX_CSV
from preprocessing.audio_ops import load_audio, lowpass_filter, resample_audio, rms_normalize
from preprocessing.make_unified_windows import label_pulmonary
from integration.window_inference import SR, fixed_window_starts

ROOT = Path(__file__).resolve().parents[2]
SPR_ROOT = ROOT / "data" / "raw" / "SPRSound"
TABLES = ROOT / "reports" / "sprsound"
OUT_DIR = TABLES / "preprocessing"
WAV_DIRS = {"train2022": "BioCAS2022/train2022_wav", "test2022_inter": "BioCAS2022/test2022_wav",
            "test2022_intra": "BioCAS2022/test2022_wav", "test2023": "BioCAS2023/test2023_wav",
            "test2024": "BioCAS2024/test2024_wav", "test2025": "BioCAS2025/test2025_wav"}
CRACKLE_TYPES = {"Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"}
WHEEZE_TYPES = {"Wheeze", "Rhonchi", "Wheeze+Crackle"}
GROUPS = ["normal", "crackle", "wheeze", "both"]
NPERSEG_NATIVE, NPERSEG_4K = 512, 256       # 15.6 Hz de resolución en los dos
CHUNK_SEC = 0.5                              # trozos para la curtosis (depende del largo)
MIN_GAP_SEC = 0.5
BAND = butter(4, [150, 1800], btype="band", fs=SR, output="sos")


def event_group(t: str) -> str | None:
    c, w = t in CRACKLE_TYPES, t in WHEEZE_TYPES
    return "both" if c and w else "crackle" if c else "wheeze" if w else "normal" if t == "Normal" else None


def psd(y: np.ndarray, sr: int, nperseg: int) -> np.ndarray | None:
    if len(y) < nperseg:
        return None
    return welch(y, fs=sr, nperseg=nperseg)[1]


def chunk_features(y: np.ndarray, rec_rms: float) -> list[tuple[float, float]]:
    """(curtosis en banda 150-1800 Hz, RMS relativo a la grabación) por trozo
    de CHUNK_SEC del audio preprocesado a 4 kHz."""
    n = int(CHUNK_SEC * SR)
    out = []
    for s in range(0, len(y) - n + 1, n):
        seg = y[s:s + n]
        out.append((float(kurtosis(sosfiltfilt(BAND, seg))), float(np.sqrt(np.mean(seg ** 2)) / rec_rms)))
    return out


def gaps(events: pd.DataFrame, dur: float) -> list[tuple[float, float]]:
    """Intervalos sin ningún evento, de al menos MIN_GAP_SEC."""
    iv = sorted(zip(events["start_sec"], events["end_sec"]))
    out, cur = [], 0.0
    for a, b in iv:
        if a - cur >= MIN_GAP_SEC:
            out.append((cur, a))
        cur = max(cur, b)
    if dur - cur >= MIN_GAP_SEC:
        out.append((cur, dur))
    return out


def analyze_sprsound(args: tuple) -> dict:
    stem, split, record, events = args
    y8, sr8 = load_audio(SPR_ROOT / WAV_DIRS[split] / f"{stem}.wav")
    # preprocess() a mano para tener las dos versiones del mismo audio
    y4, _ = resample_audio(lowpass_filter(y8, sr8), sr8, SR)
    raw_rms = float(np.sqrt(np.mean(y4 ** 2)))
    y4n = rms_normalize(y4)
    clipped = float(np.mean(np.abs(y4n) >= 1.0))
    res = {"stem": stem, "raw_rms": raw_rms, "raw_peak": float(np.abs(y8).max()), "clipped_frac": clipped,
           "native": {g: [] for g in GROUPS}, "pre": {g: [] for g in GROUPS}, "chunks": [], "events": []}
    for _, e in events.iterrows():
        g = event_group(e["type"])
        a8, b8 = int(e["start_sec"] * sr8), int(e["end_sec"] * sr8)
        a4, b4 = int(e["start_sec"] * SR), int(e["end_sec"] * SR)
        seg = y4n[a4:b4]
        if g is not None:
            for key, p in (("native", psd(y8[a8:b8], sr8, NPERSEG_NATIVE)), ("pre", psd(seg, SR, NPERSEG_4K))):
                if p is not None:
                    res[key][g].append(p)
        peak_to_rms = float(np.abs(seg).max() / (np.sqrt(np.mean(seg ** 2)) + 1e-12)) if len(seg) else np.nan
        res["events"].append({"stem": stem, "type": e["type"], "group": g, "peak_to_rms": peak_to_rms,
                              "clipped_frac": float(np.mean(np.abs(seg) >= 1.0)) if len(seg) else np.nan})
        if g in ("normal", "crackle"):
            res["chunks"] += [{"kind": f"evento {g}", "record": record, "kurt": k, "rel_rms": r}
                              for k, r in chunk_features(seg, 0.1)]
    for a, b in gaps(events, len(y8) / sr8):
        res["chunks"] += [{"kind": "hueco", "record": record, "kurt": k, "rel_rms": r}
                          for k, r in chunk_features(y4n[int(a * SR):int(b * SR)], 0.1)]
        res["events"].append({"stem": stem, "type": "(hueco)", "group": None, "gap_sec": b - a})
    return res


def analyze_icbhi(args: tuple) -> dict:
    file_id, cycles = args
    y, sr = load_audio(AUDIO_DIR_V2 / file_id)
    y4n = rms_normalize(resample_audio(lowpass_filter(y, sr), sr, SR)[0])
    res = {"pre": {g: [] for g in GROUPS}, "clipped_frac": float(np.mean(np.abs(y4n) >= 1.0)),
           "chunks": [], "events": []}
    for _, c in cycles.iterrows():
        g = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}[c["crackle"] + 2 * c["wheeze"]]
        seg = y4n[int(c["start_sec"] * SR):int((c["start_sec"] + c["duration_sec"]) * SR)]
        p = psd(seg, SR, NPERSEG_4K)
        if p is not None:
            res["pre"][g].append(p)
        if len(seg):
            res["events"].append({"group": g, "peak_to_rms": float(np.abs(seg).max() / (np.sqrt(np.mean(seg ** 2)) + 1e-12)),
                                  "clipped_frac": float(np.mean(np.abs(seg) >= 1.0))})
        if g in ("normal", "crackle"):
            res["chunks"] += [{"kind": f"ICBHI {g}", "record": "", "kurt": k, "rel_rms": r}
                              for k, r in chunk_features(seg, 0.1)]
    return res


def mean_db(psds: list[np.ndarray]) -> np.ndarray:
    return 10 * np.log10(np.mean(psds, axis=0) + 1e-20)


def band_fraction(psds: list[np.ndarray], freqs: np.ndarray, lo: float, hi: float) -> float:
    p = np.mean(psds, axis=0)
    return float(p[(freqs >= lo) & (freqs < hi)].sum() / p.sum())


def windows_table(recs: pd.DataFrame, evs: pd.DataFrame) -> pd.DataFrame:
    """Lo mismo que pulmonary_windows() de make_unified_windows.py, con los
    eventos de SPRSound como ciclos (crackle/wheeze según el tipo)."""
    win, hop = 4 * SR, 2 * SR
    by = {s: g for s, g in evs.groupby("stem")}
    rows = []
    for _, r in recs.iterrows():
        # largo tras el resample 8 kHz -> 4 kHz (como preprocessed_length())
        n = int(np.ceil(round(r["duration_sec"] * 8000) * SR / 8000))
        c = by.get(r["stem"])
        if c is None:
            continue
        cs, ce, dur = c["start_sec"].to_numpy(), c["end_sec"].to_numpy(), c["duration_sec"].to_numpy()
        flags = {"crackle": c["type"].isin(CRACKLE_TYPES).to_numpy(), "wheeze": c["type"].isin(WHEEZE_TYPES).to_numpy(),
                 "stridor": (c["type"] == "Stridor").to_numpy()}
        for w, s in enumerate(fixed_window_starts(n, win, hop)):
            a, b = s / SR, (s + min(win, n - s)) / SR
            ov = np.clip(np.minimum(b, ce) - np.maximum(a, cs), 0, None)
            row = {"stem": r["stem"], "patient": r["patient"], "split": r["split"], "has_patient_id": r["has_patient_id"],
                   "window_idx": w, "annotated_sec": float(ov.sum()), "win_sec": b - a}
            for col, m in flags.items():
                row[f"{col}_ov_sec"] = float(ov[m].max()) if m.any() else 0.0
                row[f"{col}_ov_frac"] = float((ov[m] / dur[m]).max()) if m.any() else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def plot_examples(recs: pd.DataFrame, evs: pd.DataFrame, path: Path, seed: int = 0) -> None:
    colors = {"normal": "tab:green", "crackle": "tab:red", "wheeze": "tab:blue", "both": "tab:purple", None: "tab:gray"}
    picks = pd.concat([recs[recs["record_annotation"] == lab].sample(2, random_state=seed)
                       for lab in ("Normal", "DAS", "CAS", "CAS & DAS")])
    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 2.4 * len(picks)))
    for ax, (_, r) in zip(axes, picks.iterrows()):
        y, sr = load_audio(SPR_ROOT / WAV_DIRS[r["split"]] / f"{r['stem']}.wav")
        y4 = rms_normalize(resample_audio(lowpass_filter(y, sr), sr, SR)[0])
        ax.specgram(y4, NFFT=256, Fs=SR, noverlap=192, cmap="magma")
        for _, e in evs[evs["stem"] == r["stem"]].iterrows():
            ax.axvspan(e["start_sec"], e["end_sec"], ymin=0.92, ymax=1.0, color=colors[event_group(e["type"])])
            ax.text(e["start_sec"], 1850, e["type"][:7], fontsize=6, color="white")
        ax.set_title(f"{r['stem']} — {r['record_annotation']} ({r['split']})", fontsize=8)
        ax.set_ylabel("Hz")
    axes[-1].set_xlabel("s  (barra de color: evento; sin barra: hueco sin anotar)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-recs", type=int, default=None, help="para probar rápido")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    recs = pd.read_csv(TABLES / "recordings.csv", dtype={"patient": str})
    evs = pd.read_csv(TABLES / "events.csv", dtype={"patient": str})
    recs = recs[recs["record_annotation"] != "Poor Quality"]
    if args.max_recs:
        recs = recs.sample(args.max_recs, random_state=0)
    evs = evs[evs["stem"].isin(recs["stem"])]
    by = {s: g for s, g in evs.groupby("stem")}
    jobs = [(r.stem, r.split, r.record_annotation if isinstance(r.record_annotation, str) else "(sin etiqueta)",
             by[r.stem]) for r in recs.itertuples() if r.stem in by]

    cycles = pd.read_csv(PULMONARY_CYCLES_INDEX_CSV)
    icb_jobs = list(cycles.groupby("file_id"))
    if args.max_recs:
        icb_jobs = icb_jobs[:args.max_recs // 4]

    with ProcessPoolExecutor(args.workers) as ex:
        spr = list(ex.map(analyze_sprsound, jobs, chunksize=16))
        icb = list(ex.map(analyze_icbhi, icb_jobs, chunksize=8))
    print(f"SPRSound: {len(spr)} grabaciones | ICBHI: {len(icb)} grabaciones")

    f8 = np.fft.rfftfreq(NPERSEG_NATIVE, 1 / 8000)
    f4 = np.fft.rfftfreq(NPERSEG_4K, 1 / SR)
    nat = {g: [p for r in spr for p in r["native"][g]] for g in GROUPS}
    pre = {g: [p for r in spr for p in r["pre"][g]] for g in GROUPS}
    ipre = {g: [p for r in icb for p in r["pre"][g]] for g in GROUPS}

    # 1. Energía por banda en el audio nativo
    bands = pd.DataFrame({g: {"n_eventos": len(nat[g]),
                              "<100 Hz": band_fraction(nat[g], f8, 0, 100),
                              "100-2048 Hz": band_fraction(nat[g], f8, 100, 2048),
                              ">2048 Hz (lo borra el pasa-bajos)": band_fraction(nat[g], f8, 2048, 4001)}
                          for g in GROUPS}).T
    print("\n── Fracción de energía por banda, audio nativo 8 kHz\n" + bands.round(4).to_string())
    bands.to_csv(OUT_DIR / "band_energy_native.csv")

    # Diferencia entre clases por frecuencia: ¿dónde está la información?
    ref = mean_db(nat["normal"])
    diff = pd.DataFrame({g: mean_db(nat[g]) - ref for g in ("crackle", "wheeze", "both")}, index=f8)
    span = lambda lo, hi: diff[(diff.index >= lo) & (diff.index < hi)].abs().mean()
    print("\n── |dB clase − dB normal| medio por banda (nativo)\n" +
          pd.DataFrame({"100-2048 Hz": span(100, 2048), "2048-4000 Hz": span(2048, 4001)}).round(2).to_string())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for g in GROUPS:
        axes[0].plot(f8, mean_db(nat[g]), label=f"{g} (n={len(nat[g])})")
    axes[0].axvline(2048, color="k", ls="--", lw=0.8)
    axes[0].set(title="SPRSound nativo (8 kHz), PSD media por tipo de evento", xlabel="Hz", ylabel="dB")
    axes[0].legend(fontsize=8)
    for g in GROUPS:
        axes[1].plot(f4, mean_db(pre[g]), label=f"SPRSound {g}")
        if ipre[g]:
            axes[1].plot(f4, mean_db(ipre[g]), ls="--", color=axes[1].lines[-1].get_color(), label=f"ICBHI {g}")
    axes[1].set(title="Preprocesado (4 kHz, RMS 0.1): SPRSound (—) vs ICBHI (--)", xlabel="Hz", ylabel="dB")
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "spectra.png", dpi=110)
    plt.close(fig)

    # 2. Comparación de dominio, en dB relativos a la clase normal de cada base
    dom = pd.DataFrame({
        f"{base} {g}-normal": [float(np.mean((mean_db(d[g]) - mean_db(d["normal"]))[(f4 >= lo) & (f4 < hi)]))
                               for lo, hi in ((50, 150), (150, 500), (500, 1000), (1000, 2000))]
        for base, d in (("SPR", pre), ("ICBHI", ipre)) for g in ("crackle", "wheeze", "both")},
        index=["50-150", "150-500", "500-1000", "1000-2000"])
    norm_shift = pd.Series([float(np.mean((mean_db(pre["normal"]) - mean_db(ipre["normal"]))[(f4 >= lo) & (f4 < hi)]))
                            for lo, hi in ((50, 150), (150, 500), (500, 1000), (1000, 2000))], index=dom.index)
    dom["normal SPR - normal ICBHI"] = norm_shift
    print("\n── dB medios por banda (preprocesado): contraste de cada clase con la normal de su base, "
          "y normal SPR vs normal ICBHI\n" + dom.round(2).T.to_string())
    dom.T.to_csv(OUT_DIR / "domain_band_contrast.csv")

    # 3. Nivel y recorte
    ev = pd.DataFrame([e for r in spr for e in r["events"] if e["type"] != "(hueco)"])
    iev = pd.DataFrame([e for r in icb for e in r["events"]])
    lvl = pd.DataFrame({"SPRSound": pd.Series([r["clipped_frac"] for r in spr]).describe(percentiles=[.5, .9, .99]),
                        "ICBHI": pd.Series([r["clipped_frac"] for r in icb]).describe(percentiles=[.5, .9, .99])})
    print("\n── Fracción de muestras recortadas a ±1 tras rms_normalize, por grabación\n" + lvl.round(5).to_string())
    print("\n── Grabaciones con algún recorte: SPRSound "
          f"{np.mean([r['clipped_frac'] > 0 for r in spr]):.1%}, ICBHI {np.mean([r['clipped_frac'] > 0 for r in icb]):.1%}")
    ptr = pd.concat([ev.assign(base="SPRSound"), iev.assign(base="ICBHI")]).groupby(["base", "group"]).agg(
        n=("peak_to_rms", "size"), peak_to_rms_med=("peak_to_rms", "median"),
        peak_to_rms_p90=("peak_to_rms", lambda x: x.quantile(0.9)),
        eventos_con_recorte=("clipped_frac", lambda x: (x > 0).mean()))
    print("\n── Pico/RMS dentro de cada evento (preprocesado) y eventos con alguna muestra recortada\n"
          + ptr.round(3).to_string())
    ptr.to_csv(OUT_DIR / "peak_to_rms.csv")
    rr = pd.Series([r["raw_rms"] for r in spr])
    print(f"\nRMS antes de normalizar (SPRSound): mediana {rr.median():.4f}, p1 {rr.quantile(.01):.4f}, "
          f"p99 {rr.quantile(.99):.4f}; pico nativo máximo {max(r['raw_peak'] for r in spr):.3f}")

    # 4. Huecos
    gp = pd.DataFrame([e for r in spr for e in r["events"] if e["type"] == "(hueco)"])
    print(f"\n── Huecos ≥ {MIN_GAP_SEC} s sin anotar: {len(gp)} en {gp['stem'].nunique()} grabaciones\n"
          + gp["gap_sec"].describe().round(2).to_string())
    ch = pd.concat([pd.DataFrame([c for r in spr for c in r["chunks"]]),
                    pd.DataFrame([c for r in icb for c in r["chunks"]])])
    ch["grupo"] = np.where(ch["kind"] == "hueco", "hueco en grabación " + ch["record"], ch["kind"])
    ch = ch.replace([np.inf, -np.inf], np.nan).dropna(subset=["kurt"])
    gtab = ch.groupby("grupo").agg(n_trozos=("kurt", "size"), curtosis_med=("kurt", "median"),
                                   curtosis_p75=("kurt", lambda x: x.quantile(0.75)),
                                   frac_curtosis_gt3=("kurt", lambda x: (x > 3).mean()),
                                   rms_rel_med=("rel_rms", "median"))
    print(f"\n── Trozos de {CHUNK_SEC} s: curtosis en banda 150-1800 Hz (transitorios) y RMS relativo "
          "(1 = nivel medio de la grabación)\n" + gtab.round(3).to_string())
    gtab.to_csv(OUT_DIR / "gap_vs_event_chunks.csv")

    # 5. Ventanas
    all_recs = pd.read_csv(TABLES / "recordings.csv", dtype={"patient": str})
    all_recs = all_recs[all_recs["record_annotation"] != "Poor Quality"]
    all_evs = pd.read_csv(TABLES / "events.csv", dtype={"patient": str})
    w = windows_table(all_recs, all_evs)
    print(f"\n── Ventanas de 4 s: {len(w)}; sin ningún evento (se descartan): {(w['annotated_sec'] == 0).sum()}")
    w = w[w["annotated_sec"] > 0].copy()
    w["annotated_frac"] = w["annotated_sec"] / w["win_sec"]
    print("Fracción de la ventana cubierta por eventos:\n" + w["annotated_frac"].describe().round(2).to_string())
    names = {2: "pulm_normal", 3: "pulm_crackle", 4: "pulm_wheeze", 5: "pulm_both"}
    lab = {t: label_pulmonary(w, t) for t in (0.5, 1.0)}
    w["label"], w["borderline"] = lab[1.0]
    w["stridor"] = (w["stridor_ov_sec"] >= 1.0) | (w["stridor_ov_frac"] >= 0.5)
    w["label_name"] = w["label"].map(names)
    comp = pd.DataFrame({f"{t:g}s": lab[t][0].map(names).value_counts() for t in lab})
    comp.loc["borderline"] = [lab[t][1].sum() for t in lab]
    comp.loc["cambian entre 0.5 y 1 s"] = (lab[0.5][0] != lab[1.0][0]).sum()
    print("\n── Clases con la regla actual (umbral 0.5 vs 1 s)\n" + comp.to_string())
    print(f"Ventanas que tocan estridor (≥ regla): {w['stridor'].sum()}")
    summ = w.groupby("label_name").agg(ventanas=("label", "size"), pacientes=("patient", "nunique"),
                                       ventanas_sin_id=("has_patient_id", lambda x: int((~x).sum())))
    icbhi = pd.Series({"pulm_normal": (4086, 116), "pulm_crackle": (2713, 71),
                       "pulm_wheeze": (1204, 60), "pulm_both": (972, 36)})
    summ["ICBHI ventanas"] = icbhi.map(lambda x: x[0])
    summ["ICBHI pacientes"] = icbhi.map(lambda x: x[1])
    print("\n── Por clase (umbral 1 s) contra ICBHI (avances/22 §1)\n" + summ.to_string())
    print("\n── Ventanas por split y clase\n" + pd.crosstab(w["split"], w["label_name"], margins=True).to_string())
    summ.to_csv(OUT_DIR / "windows_by_class.csv")
    w.to_csv(OUT_DIR / "windows_4s.csv", index=False)

    # 6. Ejemplos
    plot_examples(all_recs, all_evs, OUT_DIR / "examples_spectrograms.png")
    print(f"\n✓ {OUT_DIR}")


if __name__ == "__main__":
    main()
