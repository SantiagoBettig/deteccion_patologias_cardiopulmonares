"""
parsers/parse_icbhi.py
======================
Parser para ICBHI 2017 Respiratory Sound Database.
 
Estructura esperada en ICBHI_ROOT:
    ICBHI/
    ├── audio_and_txt_files/
    │   ├── PAT_IDX_LOC_MODE_EQUIP.wav
    │   └── PAT_IDX_LOC_MODE_EQUIP.txt    ← anotación de ciclos
    ├── demographic_info.csv               ← sep=";", sin header, 6 columnas
    ├── patient_diagnosis.csv             ← sep=";"
    └── filename_differences.txt          ← fe de erratas del equipo (ver abajo)
 
Nombre de WAV: 5 elementos separados por "_"
    1. Patient number  (101–226)
    2. Recording index (ej: 1u)
    3. Chest location  (Tc, Al, Ar, Pl, Pr, Ll, Lr)
    4. Acquisition mode (sc | mc)
    5. Equipment       (AKGC417L | LittC2SE | Litt3200 | Meditron)
 
Archivo TXT de anotación (por WAV):
    col1: inicio del ciclo (s)
    col2: fin del ciclo (s)
    col3: crackle (1/0)
    col4: wheeze  (1/0)
 
Lógica: una fila en el CSV maestro = un WAV (grabación completa).
has_crackles / has_wheezes = True si AL MENOS UN ciclo lo contiene.

Fe de erratas del equipo (ver avances/12):
    ICBHI publicó 91 grabaciones con "Meditron" en el nombre cuando en
    realidad se grabaron con AKGC417L o LittC2SE. filename_differences.txt
    lista el nombre correcto de cada una (sin extensión). Los archivos en
    disco NO se renombran: original_filename sigue siendo el nombre real del
    archivo (trazabilidad + copia del WAV); solo se corrige el equipo.
    sample_rate_hz se lee del header del WAV, no se infiere del equipo.
"""
 
import pandas as pd
import soundfile as sf
from pathlib import Path
 
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ICBHI_ROOT,
    ICBHI_DIAGNOSIS_MAP,
    LOCATION_MAP,
    MASTER_COLUMNS,
)
 
EQUIPMENT_MAP = {
    "AKGC417L": "AKG C417L Microphone",
    "LittC2SE": "3M Littmann Classic II SE",
    "Litt3200": "3M Littmann 3200 Electronic",
    "Meditron":  "WelchAllyn Meditron Master Elite",
}
 


def _read_equipment_errata(errata_path: Path) -> dict[str, str]:
    """
    Lee filename_differences.txt (un nombre correcto por línea, entre
    comillas simples, sin extensión) y devuelve {stem en disco: stem correcto}.
    En disco, cada uno de estos archivos figura con equipo "Meditron".
    """
    if not errata_path.exists():
        print(f"[ICBHI] ⚠ No se encontró {errata_path.name}: "
              "no se corrige el equipo de las 91 grabaciones mal nombradas.")
        return {}
    errata = {}
    for line in errata_path.read_text(encoding="utf-8").splitlines():
        correct = line.strip().strip("'")
        if correct:
            errata[correct.rsplit("_", 1)[0] + "_Meditron"] = correct
    return errata
 
 
def _parse_annotation_txt(txt_path: Path) -> tuple[bool, bool]:
    """
    Lee el archivo .txt de anotación de ciclos.
    Retorna (has_crackles, has_wheezes) como bool.
    """
    try:
        df = pd.read_csv(txt_path, sep="\t", header=None,
                         names=["start", "end", "crackle", "wheeze"])
        return bool(df["crackle"].any()), bool(df["wheeze"].any())
    except Exception as e:
        print(f"[ICBHI] ⚠ Error leyendo {txt_path.name}: {e}")
        return False, False
 
 
def _read_demographic(demo_path: Path) -> pd.DataFrame:
    """
    Lee demographic_info.csv.
    El archivo no tiene header y usa ";" como separador.
    Columnas: patient_id, age, sex, adult_bmi, child_weight, child_height
    """
    df = pd.read_csv(
        demo_path,
        sep=";",
        header=None,
        names=["patient_id", "age", "sex", "adult_bmi", "child_weight", "child_height"],
    )
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df = df.set_index("patient_id")
    return df
 
 
def _age_to_category(age) -> str:
    try:
        age_f = float(age)
        if age_f < 1:
            return "infant"
        elif age_f < 12:
            return "child"
        elif age_f < 18:
            return "adolescent"
        else:
            return "adult"
    except (TypeError, ValueError):
        return pd.NA
 
 
def parse_icbhi(counter_start: int = 0) -> tuple[pd.DataFrame, int]:
    """
    Parsea todos los WAVs de ICBHI 2017.
 
    Parámetros
    ----------
    counter_start : int
        Valor inicial del contador para file_id.
 
    Retorna
    -------
    df : pd.DataFrame con columnas MASTER_COLUMNS
    counter_end : int
    """
    audio_dir = ICBHI_ROOT / "audio_and_txt_files"
    demo_path = ICBHI_ROOT / "demographic_info.csv"
    diag_path = ICBHI_ROOT / "patient_diagnosis.csv"
    errata    = _read_equipment_errata(ICBHI_ROOT / "filename_differences.txt")
 
    for p in (audio_dir, demo_path, diag_path):
        if not p.exists():
            raise FileNotFoundError(f"No se encontró: {p}")
 
    # ── Diagnósticos por paciente ─────────────────────────────────────────────
    df_diag = pd.read_csv(diag_path, sep=";", header=None,
                          names=["patient_id", "diagnosis"])
    df_diag["patient_id"] = df_diag["patient_id"].astype(str).str.strip()
    diag_lookup = dict(zip(df_diag["patient_id"], df_diag["diagnosis"]))
    print(f"[ICBHI] Diagnósticos cargados: {len(diag_lookup)} pacientes.")
 
    # ── Demografía por paciente ───────────────────────────────────────────────
    df_demo = _read_demographic(demo_path)
    print(f"[ICBHI] Demografía cargada: {len(df_demo)} pacientes.")
 
    def get_demo(patient_id: str, col: str):
        try:
            val = df_demo.loc[patient_id, col]
            return val if pd.notna(val) else pd.NA
        except KeyError:
            return pd.NA
 
    # ── Iterar sobre WAVs ─────────────────────────────────────────────────────
    wav_files = sorted(audio_dir.glob("*.wav"))
    print(f"[ICBHI] WAVs encontrados: {len(wav_files)}")
 
    rows = []
    counter = counter_start
    parse_errors = 0
    errata_applied = 0
 
    for wav_path in wav_files:
        # Equipo corregido según la fe de erratas; el archivo no se renombra
        stem = errata.get(wav_path.stem, wav_path.stem)
        errata_applied += stem != wav_path.stem
        parts = stem.split("_")
 
        if len(parts) < 5:
            print(f"[ICBHI] ⚠ Nombre inesperado, se omite: {wav_path.name}")
            parse_errors += 1
            continue
 
        patient_id = parts[0]
        loc_code   = parts[2]
        equip_code = parts[4]
 
        # Anotación de ciclos
        txt_path = wav_path.with_suffix(".txt")
        has_crackles, has_wheezes = (
            _parse_annotation_txt(txt_path) if txt_path.exists()
            else (False, False)
        )
 
        # Diagnóstico → pathology_label unificado
        raw_diag  = diag_lookup.get(patient_id, "Unknown")
        pathology = ICBHI_DIAGNOSIS_MAP.get(raw_diag, "unknown")
 
        # Demografía
        age_years  = get_demo(patient_id, "age")
        sex_raw    = str(get_demo(patient_id, "sex")).strip()
        sex_code   = sex_raw if sex_raw in ("M", "F") else pd.NA
        age_cat    = _age_to_category(age_years)
        bmi        = get_demo(patient_id, "adult_bmi")
        weight_kg  = get_demo(patient_id, "child_weight")
        height_cm  = get_demo(patient_id, "child_height")
 
        loc_unified = LOCATION_MAP.get(loc_code, loc_code.lower())
        equip_name  = EQUIPMENT_MAP.get(equip_code, equip_code)
        sample_rate = sf.info(str(wav_path)).samplerate
 
        file_id  = f"ICB_{counter:05d}.wav"
        counter += 1
 
        rows.append({
            "file_id":               file_id,
            "source_db":             "ICB",
            "original_filename":     wav_path.name,
            "subject_id":            f"ICB_{patient_id}",
            "subject_type":          "real",
            "sex":                   sex_code,
            "age_category":          age_cat,
            "age_years":             age_years,
            "height_cm":             height_cm,
            "weight_kg":             weight_kg,
            "bmi":                   bmi,
            "sound_type":            "pulmonary",
            "pathology_label":       pathology,
            "auscultation_location": loc_unified,
            "equipment":             equip_name,
            "sample_rate_hz":        sample_rate,
            "duration_sec":          pd.NA,
            "has_crackles":          has_crackles,
            "has_wheezes":           has_wheezes,
        })
 
    print(f"[ICBHI] Equipo corregido por fe de erratas: {errata_applied}/{len(errata)} grabaciones.")
    if errata_applied != len(errata):
        print("[ICBHI] ⚠ Algunas entradas de la fe de erratas no coinciden con ningún WAV.")
    if parse_errors:
        print(f"[ICBHI] ⚠ {parse_errors} archivos omitidos por nombre inesperado.")
 
    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    print(f"[ICBHI] Total procesados: {len(df)} registros.\n")
    return df, counter
 
 
if __name__ == "__main__":
    df, _ = parse_icbhi()
    print(df[["file_id", "pathology_label", "age_category", "age_years",
              "sex", "bmi", "has_crackles", "has_wheezes"]].head(20).to_string())
    print("\nDistribución de patologías:")
    print(df["pathology_label"].value_counts())
    print("\nNaN por columna demográfica:")
    cols = ["age_years", "sex", "age_category", "bmi", "weight_kg", "height_cm"]
    print(df[cols].isnull().sum())