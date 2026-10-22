"""
integration/external_hls.py
==============================
Validación externa de punta a punta con HLS-CMDS (maniquí clínico, un solo
estetoscopio: 3M Littmann CORE) — avances/20 §4. Ningún modelo vio nunca
estos audios, y la configuración (duración de ventana, regla de agregación,
umbrales) se congela ANTES con los datos propios: acá no se elige nada.

Conjuntos:
- HS (50 grabaciones cardíacas): murmullo = los 4 tipos de "Murmur"; el
  resto (normal, S3, S4, fibrilación auricular, taquicardia, bloqueo AV) es
  "sin murmullo", igual que abnormal_heart_unspecified al entrenar.
- LS (50 pulmonares): normal, crackles (finos y gruesos) y wheezing tienen
  equivalente directo en ICBHI -> métricas principales (33 grabaciones).
  Rhonchi (8) y pleural rub (9) no son clases de ICBHI: se reporta qué
  predice el modelo (y una variante con rhonchi = wheeze, "wheeze grave").
- Mix (145 tríos): una mezcla corazón + pulmón (M) y sus dos componentes por
  separado (H, L). Mide cuánto cambia cada rama cuando el otro sonido está
  presente — en una auscultación real de pulmón también se oye el corazón.

Cada rama se evalúa con cada uno de sus 5 modelos de fold (con la regla y
los umbrales de ese fold) y se reporta media ± desvío entre modelos.

Uso:
    python integration/external_hls.py [--murmur-sec 2.5] [--pulmonary-sec 4]
        [--strategy pooled_fixed] [--murmur-probs murmur.csv] [--pulmonary-probs pulmonary.csv]
    (los CSV de probabilidades in-domain tienen que corresponder a esas
    duraciones: de ahí salen la regla y los umbrales)
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import INTEGRATION_PROBS_DIR, INTEGRATION_REPORTS_DIR
from unified_dataset.config import HLS_ROOT
from preprocessing.audio_ops import preprocess
from integration.window_inference import (
    K, SR, PULMONARY_HIGHPASS, MAX_FRAMES, fixed_window_starts, gate_slice_windows, load_gate,
    load_murmur_folds, load_pulmonary_folds, mfcc_batch, pulmonary_batch,
)
from integration.evaluate_integration import (
    MURMUR, PULMONARY, STRATEGIES, GATE_THRESHOLD, aggregate, binary_metrics, calibrate_folds,
    icbhi_from_4class, load_recordings, window_thresholds,
)
from pulmonary_adventitious_model.metrics import CLASS_NAMES, to_4class

OUT_DIR = INTEGRATION_REPORTS_DIR / "external_hls"
LUNG_LABELS = {  # tipo HLS -> (crackle, wheeze); None = sin equivalente en ICBHI
    "Normal": (0, 0), "Fine Crackles": (1, 0), "Coarse Crackles": (1, 0), "Wheezing": (0, 1),
    "Rhonchi": None, "Pleural Rub": None,
}


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df.apply(lambda c: c.str.strip() if pd.api.types.is_string_dtype(c) else c)   # hay IDs con espacios (" M_W_LMA")


def load_hls() -> pd.DataFrame:
    """Una fila por audio: path, conjunto, dominio y etiquetas."""
    rows = []
    hs = _read_csv(HLS_ROOT / "HS" / "HS.csv")
    for _, r in hs.iterrows():
        rows.append({"audio_id": f"HS/{r['Heart Sound ID']}", "set": "HS", "sound_type": "cardiac",
                     "hls_type": r["Heart Sound Type"], "location": r["Location"]})
    ls = _read_csv(HLS_ROOT / "LS" / "LS.csv")
    for _, r in ls.iterrows():
        rows.append({"audio_id": f"LS/{r['Lung Sound ID']}", "set": "LS", "sound_type": "pulmonary",
                     "hls_type": r["Lung Sound Type"], "location": r["Location"]})
    mix = _read_csv(HLS_ROOT / "Mix.csv")
    for _, r in mix.iterrows():
        common = {"location": r["Location"], "heart_type": r["Heart Sound Type"], "lung_type": r["Lung Sound Type"],
                  "mix_id": r["Mixed Sound ID"]}
        rows += [{"audio_id": f"Mix/{r['Heart Sound ID']}", "set": "Mix-H", "sound_type": "cardiac",
                  "hls_type": r["Heart Sound Type"], **common},
                 {"audio_id": f"Mix/{r['Lung Sound ID']}", "set": "Mix-L", "sound_type": "pulmonary",
                  "hls_type": r["Lung Sound Type"], **common},
                 {"audio_id": f"Mix/{r['Mixed Sound ID']}", "set": "Mix-M", "sound_type": "mix",
                  "hls_type": f"{r['Heart Sound Type']} + {r['Lung Sound Type']}", **common}]
    df = pd.DataFrame(rows)
    df["heart_type"] = df["heart_type"].fillna(df["hls_type"].where(df["set"] == "HS"))
    df["lung_type"] = df["lung_type"].fillna(df["hls_type"].where(df["set"] == "LS"))
    df["murmur"] = df["heart_type"].str.contains("Murmur", na=False).astype(int)
    lab = df["lung_type"].map(lambda t: LUNG_LABELS.get(t) if isinstance(t, str) else None)
    df["lung_icbhi"] = lab.notna()
    df["has_crackles"] = lab.map(lambda v: v[0] if v else np.nan)
    df["has_wheezes"] = lab.map(lambda v: v[1] if v else np.nan)
    return df


# ── Inferencia (con caché) ────────────────────────────────────────────────────
def run_inference(hls: pd.DataFrame, murmur_sec: float, pulmonary_sec: float, smoke: bool = False) -> dict[str, pd.DataFrame]:
    """Probabilidades por ventana de las tres redes sobre TODOS los audios
    (cada rama también sobre el dominio que no le corresponde: así el
    sistema completo tiene salida aunque el gate rutee mal)."""
    tag = f"m{murmur_sec:g}_p{pulmonary_sec:g}" + ("_smoke" if smoke else "")
    paths = {b: INTEGRATION_PROBS_DIR / f"hls_{b}_{tag if b != 'gate' else 'v5' + ('_smoke' if smoke else '')}.csv"
             for b in ("gate", "murmur", "pulmonary")}
    if all(p.exists() for p in paths.values()):
        return {b: pd.read_csv(p) for b, p in paths.items()}

    torch.set_flush_denormal(True)
    gate, murmur, pulmonary = load_gate(), load_murmur_folds(), load_pulmonary_folds()
    mw, pw = int(murmur_sec * SR), int(pulmonary_sec * SR)
    p_max_frames = MAX_FRAMES if pulmonary_sec * 100 <= MAX_FRAMES else int(round(pulmonary_sec * 100)) + 1
    rows = {"gate": [], "murmur": [], "pulmonary": []}
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, audio_id in enumerate(hls["audio_id"], 1):
            path = HLS_ROOT / f"{audio_id}.wav"
            y, _ = preprocess(path, target_sr=SR)
            chunks = gate_slice_windows(y)
            for w, p in enumerate(torch.sigmoid(gate(mfcc_batch(chunks))).numpy()):
                rows["gate"].append({"audio_id": audio_id, "window_idx": w, "prob_cardiac": float(p)})
            starts = fixed_window_starts(len(y), mw, mw // 2)
            x = mfcc_batch([y[s:s + mw] for s in starts])
            for k in range(K):
                for w, p in enumerate(torch.sigmoid(murmur[k](x)).numpy()):
                    rows["murmur"].append({"audio_id": audio_id, "fold": k, "window_idx": w, "prob": float(p)})
            y, _ = preprocess(path, target_sr=SR, highpass_cutoff=PULMONARY_HIGHPASS)
            starts = fixed_window_starts(len(y), pw, pw // 2)
            x, lens = pulmonary_batch([y[s:s + pw] for s in starts], p_max_frames)
            for k in range(K):
                for w, p in enumerate(torch.sigmoid(pulmonary[k](x, lens)).numpy()):
                    rows["pulmonary"].append({"audio_id": audio_id, "fold": k, "window_idx": w,
                                              "p_crackle": float(p[0]), "p_wheeze": float(p[1])})
            if i % 50 == 0:
                print(f"  {i}/{len(hls)} audios ({time.perf_counter() - t0:.0f} s)", flush=True)
    out = {b: pd.DataFrame(r) for b, r in rows.items()}
    for b, df in out.items():
        df.to_csv(paths[b], index=False)
    return out


# ── Predicción por grabación con la configuración congelada ──────────────────
def frozen_config(task, probs_csv: str, strategy: str) -> tuple[dict, dict]:
    recs = load_recordings()
    cols = ["prob"] if task is MURMUR else ["p_crackle", "p_wheeze"]
    rec = aggregate(pd.read_csv(INTEGRATION_PROBS_DIR / probs_csv), cols, window_thresholds())
    rec = rec.merge(recs[["file_id"] + task.labels], on="file_id")
    return calibrate_folds(rec, task, strategy)


def predict_recordings(windows: pd.DataFrame, task, calib, chosen) -> pd.DataFrame:
    """Una fila por (audio, fold): score y predicción con la regla y los
    umbrales congelados de ese fold."""
    cols = ["prob"] if task is MURMUR else ["p_crackle", "p_wheeze"]
    agg = aggregate(windows, cols, window_thresholds(), keys=["audio_id", "fold"])
    parts = []
    for k in range(K):
        a = agg[agg.fold == k]
        rule, thr = chosen[k], calib[k, chosen[k]][0]
        if task is MURMUR:
            parts.append(pd.DataFrame({"audio_id": a.audio_id, "fold": k, "rule": rule,
                                       "score": a[f"{rule}__prob"], "pred": (a[f"{rule}__prob"] > thr[0]).astype(int)}))
        else:
            sc, sw = a[f"{rule}__p_crackle"], a[f"{rule}__p_wheeze"]
            pc, pw = (sc > thr[0]).astype(int), (sw > thr[1]).astype(int)
            parts.append(pd.DataFrame({"audio_id": a.audio_id, "fold": k, "rule": rule,
                                       "s_crackle": sc, "s_wheeze": sw, "pred_crackle": pc, "pred_wheeze": pw,
                                       "pred_4class": to_4class(pc.to_numpy(), pw.to_numpy())}))
    return pd.concat(parts, ignore_index=True)


def mean_std(rows: list[dict]) -> pd.Series:
    df = pd.DataFrame(rows)
    return df.mean().round(3).astype(str) + " ± " + df.std().round(3).astype(str)


def pulmonary_metrics(d: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score
    true4 = to_4class(d["has_crackles"].astype(int).to_numpy(), d["has_wheezes"].astype(int).to_numpy())
    out = icbhi_from_4class(true4, d["pred_4class"].to_numpy())
    out["accuracy_4class"] = float((d["pred_4class"].to_numpy() == true4).mean())
    for lab, s in [("crackle", "s_crackle"), ("wheeze", "s_wheeze")]:
        y = d[f"has_{lab}s"].astype(int)
        out[f"{lab}_auc"] = roc_auc_score(y, d[s]) if y.nunique() > 1 else np.nan
        out[f"{lab}_recall"] = float(d.loc[y == 1, f"pred_{lab}"].mean()) if (y == 1).any() else np.nan
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--murmur-sec", type=float, default=2.5)
    parser.add_argument("--pulmonary-sec", type=float, default=4.0)
    parser.add_argument("--strategy", default="pooled_fixed", choices=STRATEGIES)
    parser.add_argument("--murmur-probs", default="murmur.csv")
    parser.add_argument("--pulmonary-probs", default="pulmonary.csv")
    parser.add_argument("--smoke", action="store_true",
                        help="prueba del código con 3 audios por conjunto (métricas sin sentido)")
    args = parser.parse_args()
    global OUT_DIR
    if args.smoke:
        OUT_DIR = OUT_DIR.with_name("external_hls_smoke")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    hls = load_hls()
    if args.smoke:
        hls = hls.groupby(["set", "murmur"], group_keys=False).head(3)
    probs = run_inference(hls, args.murmur_sec, args.pulmonary_sec, args.smoke)
    m_calib, m_chosen = frozen_config(MURMUR, args.murmur_probs, args.strategy)
    p_calib, p_chosen = frozen_config(PULMONARY, args.pulmonary_probs, args.strategy)
    config = pd.DataFrame([{"fold": k, "murmur_rule": m_chosen[k], "murmur_thr": m_calib[k, m_chosen[k]][0][0],
                            "pulm_rule": p_chosen[k], "crackle_thr": p_calib[k, p_chosen[k]][0][0],
                            "wheeze_thr": p_calib[k, p_chosen[k]][0][1]} for k in range(K)])
    config.to_csv(OUT_DIR / "frozen_config.csv", index=False)

    gate = probs["gate"].groupby("audio_id")["prob_cardiac"].agg(["mean", "size"]).reset_index()
    gate.columns = ["audio_id", "gate_mean", "gate_windows"]
    gate["routed"] = np.where(gate["gate_mean"] > GATE_THRESHOLD, "cardiac", "pulmonary")
    mur = predict_recordings(probs["murmur"], MURMUR, m_calib, m_chosen)
    pul = predict_recordings(probs["pulmonary"], PULMONARY, p_calib, p_chosen)
    full = (hls.merge(gate, on="audio_id")
               .merge(mur, on="audio_id")
               .merge(pul.drop(columns="rule").rename(columns={}), on=["audio_id", "fold"], suffixes=("", "_p")))
    full.to_csv(OUT_DIR / "recordings_by_fold.csv", index=False)

    res = {}
    # 1. Gate (un solo modelo)
    g = hls.merge(gate, on="audio_id")
    gate_rows = []
    for s in ("HS", "LS", "Mix-H", "Mix-L"):
        d = g[g.set == s]
        gate_rows.append({"conjunto": s, "grabaciones": len(d), "accuracy": (d.routed == d.sound_type).mean(),
                          "prob_cardiaco_media": d.gate_mean.mean()})
    d = g[g.set == "Mix-M"]
    gate_rows.append({"conjunto": "Mix-M (mezcla)", "grabaciones": len(d), "accuracy": np.nan,
                      "prob_cardiaco_media": d.gate_mean.mean(), "ruteadas_a_cardiaco": (d.routed == "cardiac").mean()})
    res["gate"] = pd.DataFrame(gate_rows)

    # 2. Ramas con ruteo perfecto, por modelo de fold
    branch = {}
    for name, sel in [("HS", full.set == "HS"), ("Mix-H (corazón solo)", full.set == "Mix-H"),
                      ("Mix-M (corazón + pulmón)", full.set == "Mix-M")]:
        branch["murmullo", name] = mean_std([binary_metrics(d.murmur, d.pred, d.score)
                                             for _, d in full[sel].groupby("fold")])
    for name, sel in [("LS", (full.set == "LS") & full.lung_icbhi),
                      ("Mix-L (pulmón solo)", (full.set == "Mix-L") & full.lung_icbhi),
                      ("Mix-M (corazón + pulmón)", (full.set == "Mix-M") & full.lung_icbhi)]:
        branch["pulmonar", name] = mean_std([pulmonary_metrics(d) for _, d in full[sel].groupby("fold")])
    # variante: rhonchi como wheeze (sibilancia grave)
    ls_r = full[(full.set == "LS") & (full.lung_icbhi | (full.hls_type == "Rhonchi"))].copy()
    ls_r.loc[ls_r.hls_type == "Rhonchi", ["has_crackles", "has_wheezes"]] = [0, 1]
    branch["pulmonar", "LS, rhonchi = wheeze"] = mean_std([pulmonary_metrics(d) for _, d in ls_r.groupby("fold")])
    res["branch"] = pd.DataFrame(branch).T
    res["branch"].index.names = ["rama", "conjunto"]

    # 3. Qué predice la rama pulmonar para cada tipo de sonido de LS (todos los modelos)
    ls = full[full.set == "LS"]
    res["lung_types"] = (pd.crosstab(ls.hls_type, ls.pred_4class.map(dict(enumerate(CLASS_NAMES))), normalize="index")
                         .reindex(columns=CLASS_NAMES, fill_value=0).round(2))
    res["heart_types"] = full[full.set == "HS"].groupby("hls_type")["pred"].mean().round(2).rename("prop_pred_murmullo").to_frame()

    # 4. Sistema completo sobre HS + LS (sonidos con equivalente): gate -> rama
    sysrows = []
    for k, d in full[full.set.isin(["HS", "LS"]) & ((full.set == "HS") | full.lung_icbhi)].groupby("fold"):
        true = np.where(d.sound_type == "cardiac", d.murmur,
                        2 + to_4class(d.has_crackles.fillna(0).astype(int), d.has_wheezes.fillna(0).astype(int)))
        perfect = np.where(d.sound_type == "cardiac", d.pred, 2 + d.pred_4class)
        system = np.where(d.routed == "cardiac", d.pred, 2 + d.pred_4class)
        c = (d.sound_type == "cardiac").to_numpy()
        sysrows.append({"accuracy_6clases_sistema": (system == true).mean(),
                        "accuracy_6clases_ruteo_perfecto": (perfect == true).mean(),
                        "murmullo_f2_sistema": binary_metrics(d.murmur[c], (system[c] == 1).astype(int))["f2"],
                        "murmullo_f2_ruteo_perfecto": binary_metrics(d.murmur[c], (perfect[c] == 1).astype(int))["f2"],
                        "pulmonar_score_sistema": icbhi_from_4class(true[~c] - 2, system[~c] - 2)["icbhi_score"],
                        "pulmonar_score_ruteo_perfecto": icbhi_from_4class(true[~c] - 2, perfect[~c] - 2)["icbhi_score"]})
    res["system"] = mean_std(sysrows).to_frame("media ± desvío (5 modelos)")

    for name, df in res.items():
        df.to_csv(OUT_DIR / f"{name}.csv")
    print(f"Configuración congelada ({args.strategy}, murmullo {args.murmur_sec:g} s, pulmonar {args.pulmonary_sec:g} s):")
    print(config.to_string(index=False))
    for name, title in [("gate", "Gate v5 por grabación"), ("branch", "Ramas con ruteo perfecto (media ± desvío, 5 modelos)"),
                        ("lung_types", "Rama pulmonar: predicción por tipo de sonido de LS"),
                        ("heart_types", "Rama de murmullo: proporción predicha con murmullo por tipo (HS)"),
                        ("system", "Sistema completo sobre HS + LS")]:
        print(f"\n== {title} ==")
        print(res[name].to_string())
    print(f"\n✓ Reportes en: {OUT_DIR}")


if __name__ == "__main__":
    main()
