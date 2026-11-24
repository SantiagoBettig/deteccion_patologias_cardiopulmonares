"""
eda/device_characterization.py
=================================
¿En qué se diferencian los equipos de grabación, y eso explica la decisión
de dominio de los modelos fuera de su base? (avances/28 §3, §12).

Grupos (equipo, dominio), sobre el audio que ve el modelo (caches
preprocesadas: pasa-bajos 2048 Hz, 4 kHz, RMS 0.1, sin pasa-altos):
- CirCor: Littmann 3200 + DigiScope Collector (Oliveira et al., 2022), cardíaco.
- SPRSound: Yunting model II (Zhang et al., 2022), pulmonar.
- ICBHI por equipo: AKG C417L, Littmann Classic II SE, Littmann 3200,
  Meditron (pulmonar). El Littmann 3200 es el MISMO equipo que CirCor con
  otro dominio.
- HLS-CMDS: Littmann CORE, cardíaco (HS) y pulmonar (LS), el mismo equipo
  con los dos dominios.

Para separar equipo de contenido se comparan también solo las grabaciones
pulmonares normales (SPRSound "Normal", ICBHI sin crackle ni wheeze, HLS-LS
"Normal").

Rasgos por grabación (Welch, nperseg 512 → bins de 7.8 Hz):
- espectro promedio normalizado (LTAS, potencia relativa en dB);
- fracción de energía por banda (dB): 20-100, 100-200, 200-400, 400-800,
  800-1600 y 1600-2000 Hz;
- centroide espectral y frecuencia de rolloff del 95%;
- piso de ruido: percentil 10 de la energía por tramos de 50 ms respecto de
  la mediana (dB; más negativo = más rango dinámico, menos ruido de fondo);
- fracción de muestras recortadas (|y| ≥ 0.999 tras la normalización).

"Regla espectral": una regresión logística sobre esos rasgos, entrenada
SOLO para separar CirCor de SPRSound (el problema de dominio que vio el
modelo al entrenar). Se valida con 5 folds por paciente y se aplica a las
bases externas. Si la regla reproduce la decisión de dominio de los modelos
afuera, la decisión se puede explicar por el espectro del equipo.

Salida: reports/external_validation/devices/
    features.csv            rasgos por grabación
    group_summary.csv       mediana [p25, p75] por grupo
    rule_vs_models.csv      fracción a cardíaco: regla espectral contra modelos (ensamble)
    rule_correlation.csv    Spearman entre P(cardíaco) de la regla y de cada modelo, por grupo
    ltas.png, bands.png, rule_vs_models.png

Uso:
    python eda/device_characterization.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import (EXTERNAL_REPORTS_DIR, external_paths, HLS_AUDIO_CACHE_NPY, HLS_AUDIO_OFFSETS_CSV, HLS_RECORDINGS_CSV,
                          SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV, SPRSOUND_RECORDINGS_CSV,
                          UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV)
from integration.evaluate_integration import load_recordings

SR = 4000
OUT = EXTERNAL_REPORTS_DIR / "devices"
BANDS = [(20, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 2000)]
BAND_COLS = [f"band_{lo}_{hi}" for lo, hi in BANDS]
FEATURES = BAND_COLS + ["centroid_hz", "rolloff95_hz", "noise_floor_db"]
ENS = 5
SHORT = {"AKG C417L Microphone": "AKG C417L", "3M Littmann Classic II SE": "Littmann Classic II SE",
         "3M Littmann 3200 Electronic": "Littmann 3200", "WelchAllyn Meditron Master Elite": "Meditron"}
GROUP_ORDER = ["CirCor · Littmann 3200 · cardíaco", "SPRSound · Yunting II · pulmonar",
               "ICBHI · Littmann 3200 · pulmonar", "ICBHI · Littmann Classic II SE · pulmonar",
               "ICBHI · AKG C417L · pulmonar", "ICBHI · Meditron · pulmonar",
               "HLS · Littmann CORE · cardíaco", "HLS · Littmann CORE · pulmonar"]
NEW_GROUPS = ["PASCAL B · DigiScope · cardíaco", "KAUH · Littmann 3200 (campana) · pulmonar",
              "KAUH · Littmann 3200 (diafragma) · pulmonar", "KAUH · Littmann 3200 (extendido) · pulmonar"]
ALL_GROUPS = GROUP_ORDER + NEW_GROUPS
# Comparación con el mismo equipo de CirCor (ltas_littmann3200.png), con la paleta en orden fijo
L3200_GROUPS = [GROUP_ORDER[0], NEW_GROUPS[0], GROUP_ORDER[2]] + NEW_GROUPS[1:] + [GROUP_ORDER[1]]
L3200_COLORS = dict(zip(L3200_GROUPS, ["#2a78d6", "#6250d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#eb6834"]))
DOMAIN_COLOR = {"cardíaco": "#2a78d6", "pulmonar": "#eb6834"}
COLORS = {  # paleta categórica de referencia, en orden fijo
    GROUP_ORDER[0]: "#2a78d6", GROUP_ORDER[1]: "#eb6834", GROUP_ORDER[2]: "#1baf7a", GROUP_ORDER[3]: "#eda100",
    GROUP_ORDER[4]: "#e87ba4", GROUP_ORDER[5]: "#008300", GROUP_ORDER[6]: "#6250d6", GROUP_ORDER[7]: "#e34948"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False, "figure.dpi": 150,
    "savefig.bbox": "tight", "axes.titlesize": 10.5, "axes.titleweight": "bold", "text.color": INK,
})


# ── Rasgos ────────────────────────────────────────────────────────────────────
def features(y: np.ndarray) -> tuple[dict, np.ndarray]:
    f, p = welch(y, SR, nperseg=512)
    keep = (f >= 20) & (f <= 2000)
    f, p = f[keep], p[keep]
    total = p.sum() + 1e-20
    out = {c: 10 * np.log10(p[(f >= lo) & (f < hi)].sum() / total + 1e-12) for c, (lo, hi) in zip(BAND_COLS, BANDS)}
    out["centroid_hz"] = float((f * p).sum() / total)
    out["rolloff95_hz"] = float(f[np.searchsorted(np.cumsum(p) / total, 0.95)])
    frame = int(0.05 * SR)
    e = (y[: len(y) // frame * frame].reshape(-1, frame) ** 2).mean(1) + 1e-12
    out["noise_floor_db"] = float(10 * np.log10(np.percentile(e, 10) / np.median(e))) if len(e) >= 5 else np.nan
    out["clip_frac"] = float((np.abs(y) >= 0.999).mean())
    return out, 10 * np.log10(p / total + 1e-12)


def collect() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    recs = load_recordings()
    spr = pd.read_csv(SPRSOUND_RECORDINGS_CSV)
    hls = pd.read_csv(HLS_RECORDINGS_CSV)
    hls = hls[hls.set.isin(["HS", "LS"])]
    rows = []
    for r in recs.itertuples():
        if r.source_db == "CIR":
            g = GROUP_ORDER[0]
        else:
            g = f"ICBHI · {SHORT[r.equipment]} · pulmonar"
        rows.append({"file_id": r.file_id, "subject_id": r.subject_id, "group": g, "base": r.source_db,
                     "normal": None if r.source_db == "CIR" else bool(r.has_crackles + r.has_wheezes == 0)})
    for r in spr.itertuples():
        rows.append({"file_id": r.file_id, "subject_id": r.subject_id, "group": GROUP_ORDER[1], "base": "SPR",
                     "normal": r.record_annotation == "Normal"})
    for r in hls.itertuples():
        rows.append({"file_id": r.file_id, "subject_id": r.file_id, "base": "HLS",
                     "group": GROUP_ORDER[6] if r.set == "HS" else GROUP_ORDER[7],
                     "normal": None if r.set == "HS" else r.hls_type == "Normal"})
    kauh = pd.read_csv(external_paths("kauh")[0])
    modo = {"campana": NEW_GROUPS[1], "diafragma": NEW_GROUPS[2], "extendido": NEW_GROUPS[3]}
    for r in kauh.itertuples():
        rows.append({"file_id": r.file_id, "subject_id": r.subject_id, "base": "KAU", "group": modo[r.modo],
                     "normal": bool(r.has_crackles + r.has_wheezes == 0) and r.lung_icbhi})
    pascal = pd.read_csv(external_paths("pascal")[0])
    for r in pascal.itertuples():
        rows.append({"file_id": r.file_id, "subject_id": r.subject_id, "base": "PAS", "group": NEW_GROUPS[0],
                     "normal": None})
    meta = pd.DataFrame(rows)
    caches = {"KAU": external_paths("kauh")[1:], "PAS": external_paths("pascal")[1:],"CIR": (UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV),
              "ICB": (UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV),
              "SPR": (SPRSOUND_AUDIO_CACHE_NPY, SPRSOUND_AUDIO_OFFSETS_CSV),
              "HLS": (HLS_AUDIO_CACHE_NPY, HLS_AUDIO_OFFSETS_CSV)}
    feats, spectra = [], []
    freqs = None
    for base, g in meta.groupby("base", sort=False):
        npy, off = caches[base]
        audio = np.load(npy, mmap_mode="r")
        offsets = pd.read_csv(off).set_index("file_id")
        for fid in g.file_id:
            o, n = offsets.loc[fid, ["offset", "n_samples"]]
            y = np.asarray(audio[int(o):int(o) + int(n)], dtype=np.float32) / 32768.0
            ft, spec = features(y)
            feats.append({"file_id": fid, **ft})
            spectra.append(spec)
        print(f"  {base}: {len(g)} grabaciones", flush=True)
    f, _ = welch(np.zeros(512), SR, nperseg=512)
    freqs = f[(f >= 20) & (f <= 2000)]
    meta = meta.merge(pd.DataFrame(feats), on="file_id")
    return meta, np.array(spectra), freqs


# ── Regla espectral ───────────────────────────────────────────────────────────
def spectral_rule(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    train = df[df.base.isin(["CIR", "SPR"])].dropna(subset=FEATURES)
    X, y, groups = train[FEATURES].to_numpy(), (train.base == "CIR").astype(int).to_numpy(), train.subject_id
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    cv_acc = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        model.fit(X[tr], y[tr])
        cv_acc.append((model.predict(X[te]) == y[te]).mean())
    model.fit(X, y)
    coefs = dict(zip(FEATURES, model[-1].coef_[0]))
    ok = df[FEATURES].notna().all(axis=1)
    df = df.copy()
    df.loc[ok, "rule_p_cardiac"] = model.predict_proba(df.loc[ok, FEATURES].to_numpy())[:, 1]
    return df, {"cv_accuracy_mean": float(np.mean(cv_acc)), "cv_accuracy_std": float(np.std(cv_acc)), **coefs}


def model_domain() -> pd.DataFrame:
    """P(cardíaco) media por grabación del ensamble de cada sistema, en ICBHI y HLS."""
    parts = []
    for data in ("ICBHI", "HLS", "KAUH", "PASCAL"):
        p = pd.read_csv(EXTERNAL_REPORTS_DIR / f"predictions_{data}.csv").query("fold == @ENS")
        parts.append(p.pivot_table(index="file_id", columns="system", values="p_cardiac").reset_index())
    return pd.concat(parts, ignore_index=True)


# ── Figuras ───────────────────────────────────────────────────────────────────
def fig_ltas(df: pd.DataFrame, spectra: np.ndarray, freqs: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [("Todas las grabaciones", np.ones(len(df), bool)),
              ("Solo pulmonares normales (mismo contenido)", (df.normal == True).to_numpy())]
    for ax, (title, mask) in zip(axes, panels):
        for g in GROUP_ORDER:
            m = mask & (df.group == g).to_numpy()
            if m.sum() < 5:
                continue
            ls = "--" if "cardíaco" in g else "-"
            ax.plot(freqs, np.median(spectra[m], axis=0), color=COLORS[g], lw=1.8, ls=ls,
                    label=f"{g} (n = {m.sum()})")
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 500, 1000, 2000], ["50", "100", "200", "500", "1000", "2000"])
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_title(title, loc="left")
        ax.grid(axis="both")
    axes[0].set_ylabel("Potencia relativa (dB, mediana por grupo)")
    axes[0].set_ylim(-75, -5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.17))
    fig.suptitle("Espectro promedio por equipo (audio preprocesado; punteado = cardíaco)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "ltas.png")
    plt.close(fig)


def fig_ltas_l3200(df: pd.DataFrame, spectra: np.ndarray, freqs: np.ndarray) -> None:
    """El mismo equipo (Littmann 3200 / DigiScope) en los dos dominios, contra SPRSound."""
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for g in L3200_GROUPS:
        m = (df.group == g).to_numpy()
        ls = "--" if "cardíaco" in g else "-"
        ax.plot(freqs, np.median(spectra[m], axis=0), color=L3200_COLORS[g], lw=1.8, ls=ls, label=f"{g} (n = {m.sum()})")
    ax.set_xscale("log")
    ax.set_xticks([50, 100, 200, 500, 1000, 2000], ["50", "100", "200", "500", "1000", "2000"])
    ax.set_xlabel("Frecuencia (Hz)")
    ax.set_ylabel("Potencia relativa (dB, mediana por grupo)")
    ax.set_ylim(-75, -5)
    ax.grid(axis="both")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.set_title("Littmann 3200 / DigiScope en las cuatro bases, contra SPRSound (punteado = cardíaco)", loc="left")
    fig.savefig(OUT / "ltas_littmann3200.png")
    plt.close(fig)


def fig_bands(df: pd.DataFrame) -> None:
    cols = BAND_COLS[1:5] + ["noise_floor_db"]
    titles = ["100-200 Hz", "200-400 Hz", "400-800 Hz", "800-1600 Hz", "Piso de ruido"]
    fig, axes = plt.subplots(1, len(cols), figsize=(13, 5.2), sharey=False)
    for ax, c, t in zip(axes, cols, titles):
        data = [df.loc[df.group == g, c].dropna().to_numpy() for g in ALL_GROUPS]
        bp = ax.boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True,
                        medianprops={"color": INK, "linewidth": 1.4})
        for patch, g in zip(bp["boxes"], ALL_GROUPS):
            patch.set_facecolor(DOMAIN_COLOR[g.rsplit(" · ", 1)[1]])
            patch.set_alpha(0.75)
            patch.set_edgecolor(INK2)
        ax.set_yticks(range(1, len(ALL_GROUPS) + 1), ALL_GROUPS if c == cols[0] else [""] * len(ALL_GROUPS),
                      fontsize=8)
        ax.invert_yaxis()
        ax.set_title(t, loc="left")
        ax.set_xlabel("dB")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Fracción de energía por banda y piso de ruido, por grupo (azul: cardíaco; naranja: pulmonar)",
                 x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "bands.png")
    plt.close(fig)


def fig_rule(table: pd.DataFrame) -> None:
    groups = [g for g in GROUP_ORDER[2:] + NEW_GROUPS if g in table.index]
    series = [("Regla espectral", "regla", "#52514e", "D"), ("CNN6", "cnn6_ours64", "#2a78d6", "o"),
              ("cs_wide", "cs_wide", "#eb6834", "s"), ("Gate del pipeline", "pipeline", "#1baf7a", "^")]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    y = np.arange(len(groups))
    for k, (label, col, color, marker) in enumerate(series):
        ax.scatter(table.loc[groups, col], y + (k - 1.5) * 0.17, color=color, marker=marker, s=45, label=label,
                   zorder=3)
    ax.set_yticks(y, groups, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03)
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel("Fracción de grabaciones mandadas a cardíaco")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_title("La regla espectral (CirCor contra SPRSound) y los modelos, afuera", loc="left")
    ax.legend(loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.4, -0.24))
    fig.savefig(OUT / "rule_vs_models.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    df, spectra, freqs = collect()
    df, rule = spectral_rule(df)
    df.to_csv(OUT / "features.csv", index=False)

    q = lambda s: f"{s.median():.1f} [{s.quantile(.25):.1f}, {s.quantile(.75):.1f}]"
    summary = df.groupby("group").agg(n=("file_id", "size"), **{c: (c, q) for c in FEATURES + ["clip_frac"]})
    summary = summary.reindex(ALL_GROUPS)
    summary.to_csv(OUT / "group_summary.csv")
    print(f"\n== Rasgos por grupo: mediana [p25, p75] ==\n{summary.to_string()}")
    normal = df[df.normal == True].groupby("group")[FEATURES].median().reindex(ALL_GROUPS).dropna(how="all")
    normal.round(1).to_csv(OUT / "group_summary_pulmonar_normal.csv")
    print(f"\n== Solo pulmonares normales (mediana) ==\n{normal.round(1).to_string()}")

    print(f"\n== Regla espectral CirCor contra SPRSound ==\n"
          f"accuracy 5 folds por paciente: {rule['cv_accuracy_mean']:.4f} ± {rule['cv_accuracy_std']:.4f}\n"
          f"coeficientes (rasgos estandarizados; + = cardíaco): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in rule.items() if k in FEATURES))
    pd.Series(rule).to_csv(OUT / "rule.csv", header=["valor"])

    md = model_domain()
    ext = df[df.base.isin(["ICB", "HLS", "KAU", "PAS"])].merge(md, on="file_id")
    table = ext.groupby("group").apply(lambda g: pd.Series({
        "n": len(g), "regla": (g.rule_p_cardiac > 0.5).mean(),
        **{s: (g[s] > 0.5).mean() for s in ("cnn6_ours64", "cs_wide", "pipeline")}}), include_groups=False)
    table = table.reindex([g for g in ALL_GROUPS if g in table.index])
    table.to_csv(OUT / "rule_vs_models.csv")
    print(f"\n== Fracción mandada a cardíaco (ensamble) ==\n{table.round(3).to_string()}")
    corr = []
    for g, d in list(ext.groupby("group")) + [("ICBHI (todas)", ext[ext.base == "ICB"]),
                                               ("HLS (HS + LS)", ext[ext.base == "HLS"])]:
        corr.append({"group": g, **{s: spearmanr(d.rule_p_cardiac, d[s]).statistic
                                    for s in ("cnn6_ours64", "cs_wide", "pipeline")}})
    corr = pd.DataFrame(corr)
    corr.to_csv(OUT / "rule_correlation.csv", index=False)
    print(f"\n== Spearman: P(cardíaco) de la regla contra la de cada modelo ==\n{corr.round(3).to_string(index=False)}")

    fig_ltas(df, spectra, freqs)
    fig_ltas_l3200(df, spectra, freqs)
    fig_bands(df)
    fig_rule(table)
    print(f"\n✓ {OUT}")


if __name__ == "__main__":
    main()
