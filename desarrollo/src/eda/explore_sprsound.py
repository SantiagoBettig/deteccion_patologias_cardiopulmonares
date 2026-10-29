"""
eda/explore_sprsound.py
==========================
Exploración de SPRSound (avances/22 §8, paso 1 del plan): base pulmonar
pediátrica con eventos anotados (inicio/fin en ms), candidata a ampliar los
datos de crackle/wheeze.

No modifica nada: lee data/raw/SPRSound/ y escribe tablas en
reports/sprsound/.

Qué mira:
    1. Inventario: grabaciones por versión/split, frecuencia de muestreo,
       duración, foco, edad, sexo.
    2. Pacientes: cuántos hay, en cuántos splits aparecen (intra/inter test),
       cruce con los Patient Summary (diagnóstico).
    3. Etiquetas: por grabación (record_annotation) y por evento (tipo,
       duración, cobertura de la grabación), y consistencia entre ambas.
    4. Por paciente: cuántos tienen al menos un evento crackle / wheeze, que
       es lo que limita a ICBHI (avances/22 §6).
    5. Las anotaciones de Classification/ y Detection/ contra las de
       BioCAS20xx/ (mismos audios, JSON distintos en parte).

Uso:
    python eda/explore_sprsound.py
"""

import json
import re
from pathlib import Path

import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
SPR_ROOT = ROOT / "data" / "raw" / "SPRSound"
OUT_DIR = ROOT / "reports" / "sprsound"

# Carpeta de JSON de cada split -> carpeta de WAV
SPLITS = {
    "train2022": ("BioCAS2022/train2022_json", "BioCAS2022/train2022_wav"),
    "test2022_inter": ("BioCAS2022/test2022_json/inter_test_json", "BioCAS2022/test2022_wav"),
    "test2022_intra": ("BioCAS2022/test2022_json/intra_test_json", "BioCAS2022/test2022_wav"),
    "test2023": ("BioCAS2023/test2023_json", "BioCAS2023/test2023_wav"),
    "test2024": ("BioCAS2024/test2024_json", "BioCAS2024/test2024_wav"),
    "test2025": ("BioCAS2025/test2025_json", "BioCAS2025/test2025_wav"),
}
ALT_JSON_DIRS = ["Classification/train_classification_json", "Classification/valid_classification_json",
                 "Detection/train_detection_json", "Detection/valid_detection_json",
                 "Detection/test2024_detection_json"]
# 390 archivos de los test 2023-2025 vienen sin número de paciente
# ("_3.9_0_p1_14465"): se les da un pseudo-ID por split + edad + sexo, que
# puede juntar pacientes distintos pero nunca separa a uno mismo.
NAME_RE = re.compile(r"^(?P<patient>\d*)_(?P<age>[\d.]+)_(?P<gender>[01])_(?P<loc>p\d)_(?P<rec>\d+)$")
CRACKLE_TYPES = {"Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"}
WHEEZE_TYPES = {"Wheeze", "Rhonchi", "Wheeze+Crackle"}   # sin Stridor: se mira aparte


def load_json(path: Path) -> tuple[str | None, list[tuple[int, int, str]]]:
    d = json.loads(path.read_text(encoding="utf-8"))
    events = [(int(e["start"]), int(e["end"]), e["type"]) for e in d.get("event_annotation", [])]
    return d.get("record_annotation"), events


def load_recordings() -> tuple[pd.DataFrame, pd.DataFrame]:
    recs, evs = [], []
    for split, (jdir, wdir) in SPLITS.items():
        for jp in sorted((SPR_ROOT / jdir).glob("*.json")):
            m = NAME_RE.match(jp.stem)
            record, events = load_json(jp)
            info = sf.info(str(SPR_ROOT / wdir / f"{jp.stem}.wav"))
            patient = m["patient"] or f"noid_{split}_{m['age']}_{m['gender']}"
            recs.append({"stem": jp.stem, "split": split, "patient": patient, "has_patient_id": bool(m["patient"]),
                         "age": float(m["age"]),
                         "gender": "M" if m["gender"] == "0" else "F", "location": m["loc"],
                         "record_annotation": record, "sample_rate_hz": info.samplerate,
                         "channels": info.channels, "duration_sec": info.frames / info.samplerate,
                         "n_events": len(events)})
            evs += [{"stem": jp.stem, "split": split, "patient": patient, "start_sec": s / 1000,
                     "end_sec": e / 1000, "type": t} for s, e, t in events]
    recs, evs = pd.DataFrame(recs), pd.DataFrame(evs)
    evs["duration_sec"] = evs["end_sec"] - evs["start_sec"]
    return recs, evs


def load_patient_summaries() -> pd.DataFrame:
    rows = []
    for f in sorted((SPR_ROOT / "Patient Summary").glob("*.csv")):
        d = pd.read_csv(f, dtype={"patient_num": str}).drop(columns=["Unnamed: 0"], errors="ignore")
        rows.append(d.assign(summary=f.stem))
    d = pd.concat(rows, ignore_index=True)
    d["disease"] = d["disease"].str.strip().str.replace("Control Group", "Control group")
    return d


def compare_alt_annotations(recs: pd.DataFrame) -> pd.DataFrame:
    """Para cada JSON de Classification/Detection: ¿los eventos coinciden con
    los del JSON de BioCAS del mismo audio? ¿y la etiqueta de grabación?"""
    base = {}
    for split, (jdir, _) in SPLITS.items():
        for jp in (SPR_ROOT / jdir).glob("*.json"):
            base[jp.stem] = load_json(jp)
    rows = []
    for adir in ALT_JSON_DIRS:
        for jp in sorted((SPR_ROOT / adir).rglob("*.json")):   # valid_classification tiene subcarpetas
            rec, ev = load_json(jp)
            b = base.get(jp.stem)
            rows.append({"alt_dir": adir, "stem": jp.stem, "in_biocas": b is not None,
                         "same_record": b is not None and (rec is None or rec == b[0]),
                         "alt_has_record": rec is not None,
                         "same_events": b is not None and ev == b[1],
                         "same_event_types": b is not None and sorted(t for *_, t in ev) == sorted(t for *_, t in b[1]),
                         "n_events_alt": len(ev), "n_events_base": len(b[1]) if b else None})
    return pd.DataFrame(rows)


def per_patient(recs: pd.DataFrame, evs: pd.DataFrame) -> pd.DataFrame:
    e = evs.assign(crackle=evs["type"].isin(CRACKLE_TYPES), wheeze=evs["type"].isin(WHEEZE_TYPES),
                   both=evs["type"] == "Wheeze+Crackle", stridor=evs["type"] == "Stridor")
    flags = e.groupby("patient")[["crackle", "wheeze", "both", "stridor"]].any()
    good = recs[recs["record_annotation"] != "Poor Quality"]
    p = recs.groupby("patient").agg(n_recordings=("stem", "count"), age=("age", "first"),
                                    gender=("gender", "first"),
                                    splits=("split", lambda s: "|".join(sorted(set(s)))))
    p["n_good_recordings"] = good.groupby("patient").size().reindex(p.index, fill_value=0)
    return p.join(flags).fillna({c: False for c in flags.columns})


def section(title: str, obj) -> None:
    print(f"\n── {title} " + "─" * max(0, 70 - len(title)))
    print(obj.to_string() if hasattr(obj, "to_string") else obj)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    recs, evs = load_recordings()
    summ = load_patient_summaries()
    recs.to_csv(OUT_DIR / "recordings.csv", index=False)
    evs.to_csv(OUT_DIR / "events.csv", index=False)

    # 1. Inventario
    section("Grabaciones por split y etiqueta de grabación",
            pd.crosstab(recs["split"], recs["record_annotation"].fillna("(sin etiqueta)"), margins=True))
    section("Frecuencia de muestreo / canales", recs.groupby(["sample_rate_hz", "channels"]).size())
    section("Duración de las grabaciones (s)", recs.groupby("split")["duration_sec"].describe().round(2))
    section("Focos", pd.crosstab(recs["split"], recs["location"]))
    section("Edad (años) por paciente", recs.groupby("patient")["age"].first().describe().round(2))

    # 2. Pacientes
    pat_split = recs.groupby("patient")["split"].agg(lambda s: tuple(sorted(set(s))))
    section("Pacientes por split", recs.groupby("split")["patient"].nunique())
    section(f"Pacientes únicos: {recs['patient'].nunique()} — combinaciones de splits", pat_split.value_counts())
    noid = recs[~recs["has_patient_id"]]
    ided = recs[recs["has_patient_id"]].groupby(["age", "gender"])["patient"].nunique()
    section(f"Sin número de paciente: {len(noid)} grabaciones, {noid['patient'].nunique()} pseudo-pacientes "
            "(split+edad+sexo); cuántos comparten edad+sexo con algún paciente identificado",
            noid.groupby(["split", "patient", "age", "gender"]).size().rename("n_rec").reset_index()
            .assign(ided_same_age_sex=lambda d: [ided.get((a, g), 0) for a, g in zip(d["age"], d["gender"])])
            .groupby("split").agg(pseudo=("patient", "size"), n_rec=("n_rec", "sum"),
                                  con_coincidencia=("ided_same_age_sex", lambda x: int((x > 0).sum()))))
    pats = set(recs["patient"])
    cover = summ.groupby("summary")["patient_num"].apply(lambda s: f"{len(set(s))} en el CSV, "
                                                              f"{len(set(s) & pats)} con audio")
    section("Patient Summary: cobertura", cover)
    sdis = summ.groupby("patient_num")["disease"].agg(lambda s: "|".join(sorted(set(s))))
    section("Pacientes con diagnóstico distinto entre summaries", sdis[sdis.str.contains(r"\|")])
    section("Pacientes con audio sin diagnóstico", len(pats - set(summ["patient_num"])))

    # 3. Etiquetas
    section("Eventos por tipo", evs.groupby("type").agg(
        n_events=("type", "size"), n_recordings=("stem", "nunique"), n_patients=("patient", "nunique"),
        total_sec=("duration_sec", "sum"), median_sec=("duration_sec", "median"),
        p10_sec=("duration_sec", lambda x: x.quantile(0.1)), p90_sec=("duration_sec", lambda x: x.quantile(0.9)),
    ).round(2).sort_values("n_events", ascending=False))
    section("Eventos por split y tipo", pd.crosstab(evs["split"], evs["type"], margins=True))

    ann = evs.groupby("stem")["duration_sec"].sum()
    r = recs.set_index("stem")
    r["annotated_frac"] = (ann / r["duration_sec"]).reindex(r.index).fillna(0)
    first = evs.groupby("stem")["start_sec"].min()
    section("Fracción de la grabación cubierta por eventos (sin Poor Quality)",
            r[r["record_annotation"] != "Poor Quality"].groupby("split")["annotated_frac"].describe().round(2))
    section("Inicio del primer evento (s)", first.describe().round(2))
    neg = evs[(evs["duration_sec"] <= 0) | (evs["end_sec"] > r.loc[evs["stem"], "duration_sec"].to_numpy() + 0.05)]
    section("Eventos con duración ≤ 0 o que terminan después del audio", len(neg))

    # consistencia grabación ↔ eventos
    t = evs.groupby("stem")["type"].agg(set)
    has_c = t.apply(lambda s: bool(s & CRACKLE_TYPES))
    has_w = t.apply(lambda s: bool(s & (WHEEZE_TYPES | {"Stridor"})))
    implied = (has_c.map({True: "DAS", False: ""}) + has_w.map({True: "CAS", False: ""})).map(
        {"": "Normal", "DAS": "DAS", "CAS": "CAS", "DASCAS": "CAS & DAS"})
    r["implied"] = implied.reindex(r.index).fillna("(sin eventos)")
    lab = r[r["record_annotation"].notna()]
    section("Etiqueta de grabación (filas) vs la que implican los eventos (columnas)",
            pd.crosstab(lab["record_annotation"], lab["implied"]))

    # 4. Por paciente
    p = per_patient(recs, evs)
    p = p.join(sdis.rename("disease"))
    p.to_csv(OUT_DIR / "patients.csv")
    section("Pacientes con al menos un evento de cada tipo (crackle incluye Wheeze+Crackle; "
            "wheeze incluye Rhonchi y Wheeze+Crackle)", p[["crackle", "wheeze", "both", "stridor"]].sum())
    section("Pacientes por diagnóstico y presencia de eventos",
            p.groupby(p["disease"].fillna("(sin diagnóstico)"))[["crackle", "wheeze", "both"]]
             .agg(["sum", "count"]).iloc[:, [0, 1, 2, 4]].set_axis(["crackle", "n", "wheeze", "both"], axis=1)
             .sort_values("n", ascending=False))
    section("Grabaciones por paciente", p["n_recordings"].describe().round(2))

    # 5. Anotaciones alternativas
    alt = compare_alt_annotations(recs)
    alt.to_csv(OUT_DIR / "alt_annotations_vs_biocas.csv", index=False)
    section("Classification/Detection vs BioCAS20xx",
            alt.groupby("alt_dir")[["in_biocas", "alt_has_record", "same_record", "same_events",
                                     "same_event_types"]].agg(["sum", "count"]).xs("sum", axis=1, level=1)
            .join(alt.groupby("alt_dir").size().rename("n")))

    print(f"\n✓ Tablas en {OUT_DIR}")


if __name__ == "__main__":
    main()
