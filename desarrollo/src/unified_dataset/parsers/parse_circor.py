"""
parsers/parse_circor.py
=======================
Parser para CirCor DigiScope Phonocardiogram Dataset v1.0.3.

Estructura esperada en CIRCOR_ROOT:
    CirCor/
    ├── training_data/
    │   ├── ABCDE_XY.wav
    │   ├── ABCDE_XY.hea
    │   ├── ABCDE_XY.tsv
    │   └── ABCDE.txt
    └── training_data.csv     ← metadatos consolidados (sep=";")

Lógica de etiquetado
--------------------
Cada fila del CSV corresponde a un SUJETO (no a un WAV).
Un sujeto puede tener múltiples WAVs (una por ubicación).
Se genera una fila en el CSV maestro por cada WAV.

`pathology_label` se calcula POR UBICACIÓN, no por sujeto: un sujeto con
`Murmur=Present` puede tener grabaciones en ubicaciones donde el murmullo
no fue audible (columna `Murmur locations` != `Recording locations:`), y
esas grabaciones no deben etiquetarse `heart_murmur`.

Mapeo de pathology_label (por ubicación `loc_code`):
  Murmur=Present + loc_code en Murmur locations → heart_murmur
  Outcome=Abnormal (resto de los casos)          → abnormal_heart_unspecified
  Outcome=Normal, o Murmur=Unknown (resto)       → normal_heart   (conservador)

`clinical_outcome` (normal/abnormal): el Outcome crudo de CirCor, constante
por sujeto — a diferencia de pathology_label, NO se colapsa con la
audibilidad del murmullo. Necesario porque hay 29 sujetos con
Murmur=Present y Outcome=Normal (murmullo inocente): pathology_label los
etiqueta heart_murmur en las ubicaciones donde se oye, perdiendo el
outcome real. Ver avances/10_cardiaco_split_dos_tareas.md — hace falta
para entrenar el modelo de outcome (etapa 2) por separado del de murmullo.
"""

import re
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CIRCOR_ROOT,
    LOCATION_MAP,
    MASTER_COLUMNS,
)

# Separador del CSV de CirCor
CIRCOR_SEP = ";"

# Regex para extraer subject_id y location del nombre de WAV
# Formato: ABCDE_XY[_n].wav
WAV_PATTERN = re.compile(r"^(\d+)_([A-Za-z]+)(?:_\d+)?\.wav$")


def _derive_pathology(murmur: str, outcome: str, loc_code: str, murmur_locations: set[str]) -> str:
    murmur  = str(murmur).strip()
    outcome = str(outcome).strip()

    if murmur == "Present" and loc_code in murmur_locations:
        return "heart_murmur"
    if outcome == "Abnormal":
        return "abnormal_heart_unspecified"
    # Outcome=Normal, o Murmur=Unknown + Outcome=Normal → normal conservadoramente
    return "normal_heart"


def _normalize_age_category(age_str: str) -> str:
    mapping = {
        "Neonate":    "neonate",
        "Infant":     "infant",
        "Child":      "child",
        "Adolescent": "adolescent",
        "Young Adult": "adult",
    }
    return mapping.get(str(age_str).strip(), pd.NA)


def parse_circor(counter_start: int = 0) -> tuple[pd.DataFrame, int]:
    """
    Parsea training_data.csv de CirCor y expande a una fila por WAV.

    Parámetros
    ----------
    counter_start : int
        Valor inicial del contador para file_id.

    Retorna
    -------
    df : pd.DataFrame con columnas MASTER_COLUMNS
    counter_end : int
    """
    csv_path = CIRCOR_ROOT / "training_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró training_data.csv en {CIRCOR_ROOT}")

    df_meta = pd.read_csv(csv_path, sep=CIRCOR_SEP)
    df_meta.columns = df_meta.columns.str.strip()
    print(f"[CirCor] training_data.csv — {len(df_meta)} sujetos. "
          f"Columnas: {list(df_meta.columns)}")

    # Height/Weight: al menos un valor corrupto en el CSV crudo (ej. sujeto
    # 84784, Weight="28.160.000.000.000.000") hace que pandas infiera toda
    # la columna como texto en vez de numérica. Se fuerza a numérico — los
    # valores no parseables quedan NaN (ya se manejan como faltantes).
    n_bad_height = df_meta["Height"].apply(lambda v: pd.to_numeric(v, errors="coerce")).isna().sum() - df_meta["Height"].isna().sum()
    n_bad_weight = df_meta["Weight"].apply(lambda v: pd.to_numeric(v, errors="coerce")).isna().sum() - df_meta["Weight"].isna().sum()
    if n_bad_height or n_bad_weight:
        print(f"[CirCor] ⚠ Height/Weight con formato inválido en el CSV crudo: "
              f"{n_bad_height} altura(s), {n_bad_weight} peso(s) — se descartan como NaN.")
    df_meta["Height"] = pd.to_numeric(df_meta["Height"], errors="coerce")
    df_meta["Weight"] = pd.to_numeric(df_meta["Weight"], errors="coerce")

    audio_dir = CIRCOR_ROOT / "training_data"
    rows = []
    counter = counter_start
    missing_wavs = 0

    for _, subject in df_meta.iterrows():
        subject_id   = str(subject["Patient ID"]).strip()
        locations_raw = str(subject["Recording locations:"]).strip()
        murmur       = subject["Murmur"]
        outcome      = subject["Outcome"]
        age_cat      = _normalize_age_category(subject.get("Age", pd.NA))
        sex          = str(subject.get("Sex", "")).strip()
        sex          = sex if sex in ("Male", "Female") else pd.NA
        sex_code     = {"Male": "M", "Female": "F"}.get(str(sex), pd.NA)
        height       = subject.get("Height", pd.NA)
        weight       = subject.get("Weight", pd.NA)

        murmur_locations_raw = subject.get("Murmur locations", pd.NA)
        murmur_locations = set(
            l.strip() for l in str(murmur_locations_raw).split("+") if l.strip()
        ) if pd.notna(murmur_locations_raw) else set()

        # Las ubicaciones vienen como "AV+PV+TV+MV" → lista
        # Nota: algunos sujetos repiten una ubicación en este campo
        # (p.ej. "AV+PV+AV+TV+MV") cuando hay grabaciones adicionales en el
        # mismo punto. Como el glob de más abajo ya trae TODOS los WAVs de
        # esa ubicación (incluidos los sufijos _1/_2/_3), una ubicación
        # repetida provoca que se vuelva a procesar el mismo conjunto de
        # archivos. Se deduplica preservando el orden de aparición.
        locations = list(dict.fromkeys(
            l.strip() for l in locations_raw.split("+") if l.strip()
        ))

        for loc_code in locations:
            pathology = _derive_pathology(murmur, outcome, loc_code, murmur_locations)

            # Buscar todos los WAVs de este sujeto en esta ubicación
            # Pueden ser: ABCDE_AV.wav, ABCDE_AV_1.wav, ABCDE_AV_2.wav
            pattern = f"{subject_id}_{loc_code}*.wav"
            wav_files = sorted(audio_dir.glob(pattern))

            if not wav_files:
                missing_wavs += 1

            for wav_path in wav_files:
                file_id = f"CIR_{counter:05d}.wav"
                counter += 1
                loc_unified = LOCATION_MAP.get(loc_code, loc_code.lower())

                rows.append({
                    "file_id":               file_id,
                    "source_db":             "CIR",
                    "original_filename":     wav_path.name,
                    "subject_id":            f"CIR_{subject_id}",
                    "subject_type":          "real",
                    "sex":                   sex_code,
                    "age_category":          age_cat,
                    "age_years":             pd.NA,
                    "height_cm":             height if pd.notna(height) else pd.NA,
                    "weight_kg":             weight if pd.notna(weight) else pd.NA,
                    "bmi":                   pd.NA,
                    "sound_type":            "cardiac",
                    "pathology_label":       pathology,
                    "clinical_outcome":      str(outcome).strip().lower(),
                    "auscultation_location": loc_unified,
                    "equipment":             "3M Littmann CORE Digital Stethoscope",
                    "sample_rate_hz":        4000,
                    "duration_sec":          pd.NA,   # variable; se puede leer del WAV
                    "has_crackles":          pd.NA,
                    "has_wheezes":           pd.NA,
                })

    if missing_wavs:
        print(f"[CirCor] ⚠ {missing_wavs} combinaciones sujeto/ubicación "
              f"sin WAV encontrado.")

    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    print(f"[CirCor] Total procesados: {len(df)} registros.\n")
    return df, counter


if __name__ == "__main__":
    df, _ = parse_circor()
    print(df[["file_id", "pathology_label", "age_category",
              "sex", "auscultation_location"]].head(20).to_string())
    print("\nDistribución de patologías:")
    print(df["pathology_label"].value_counts())
    print("\nDistribución de edades:")
    print(df["age_category"].value_counts())
