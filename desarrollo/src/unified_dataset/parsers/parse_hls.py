"""
parsers/parse_hls.py
====================
Parser para HLS-CMDS v2.
 
Estructura esperada en HLS_ROOT:
    Dataset/
    ├── HS/
    │   ├── HS.csv          columnas: Gender, Heart Sound Type, Location, Heart Sound ID
    │   └── *.wav           nombre = Heart Sound ID + ".wav"  (ej: F_N_RC.wav)
    ├── LS/
    │   ├── LS.csv          columnas: Gender, Lung Sound Type, Location, Lung Sound ID
    │   └── *.wav           nombre = Lung Sound ID + ".wav"
    └── Mix/                ignorado en esta etapa
"""
 
import pandas as pd
from pathlib import Path
 
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    HLS_ROOT,
    LOCATION_MAP,
    MASTER_COLUMNS,
)
 
# ── Mapeo desde texto completo → pathology_label unificado ───────────────────
HLS_HEART_PATHOLOGY_MAP = {
    "Normal":               "normal_heart",
    "Late Diastolic Murmur":"heart_murmur",
    "Mid Systolic Murmur":  "heart_murmur",
    "Late Systolic Murmur": "heart_murmur",
    "Early Systolic Murmur":"heart_murmur",
    "Atrial Fibrillation":  "atrial_fibrillation",
    "S4":                   "extra_heart_sound",
    "S3":                   "extra_heart_sound",
    "Tachycardia":          "tachycardia",
    "AV Block":             "av_block",
}
 
HLS_LUNG_PATHOLOGY_MAP = {
    "Normal":         "normal_lung",
    "Wheezing":       "wheezing",
    "Fine Crackles":  "crackles",
    "Coarse Crackles":"crackles",
    "Rhonchi":        "rhonchi",
    "Pleural Rub":    "pleural_rub",
}
 
# ── Mapeo de ubicación: HLS usa "Apex" en HS pero abreviaturas en LS ─────────
HLS_LOCATION_MAP = {
    "Apex": "apex",
    "RC":   "right_costal_margin",
    "LC":   "left_costal_margin",
    "RUSB": "right_upper_sternal_border",
    "LUSB": "left_upper_sternal_border",
    "LLSB": "left_lower_sternal_border",
    "RUA":  "right_upper_anterior",
    "RMA":  "right_mid_anterior",
    "RLA":  "right_lower_anterior",
    "LUA":  "left_upper_anterior",
    "LMA":  "left_mid_anterior",
    "LLA":  "left_lower_anterior",
}
 
 
def _read_hls_csv(csv_path: Path) -> pd.DataFrame:
    for sep in (",", ";"):
        df = pd.read_csv(csv_path, sep=sep)
        if len(df.columns) > 1:
            df.columns = df.columns.str.strip()
            return df
    raise ValueError(f"No se pudo parsear {csv_path}")
 
 
def parse_hls(counter_start: int = 0) -> tuple[pd.DataFrame, int]:
    """
    Parsea HS.csv y LS.csv de HLS-CMDS v2. Ignora Mix.
 
    Parámetros
    ----------
    counter_start : int
        Valor inicial del contador para generar file_id únicos.
 
    Retorna
    -------
    df : pd.DataFrame con columnas MASTER_COLUMNS
    counter_end : int
    """
    rows = []
    counter = counter_start
 
    # ── Sonidos cardíacos ─────────────────────────────────────────────────────
    hs_folder  = HLS_ROOT / "HS"
    hs_csv     = hs_folder / "HS.csv"
 
    if hs_csv.exists():
        df_hs = _read_hls_csv(hs_csv)
        print(f"[HLS] HS.csv — {len(df_hs)} registros.")
 
        for _, row in df_hs.iterrows():
            sound_id   = str(row["Heart Sound ID"]).strip()
            sound_type = str(row["Heart Sound Type"]).strip()
            location   = str(row["Location"]).strip()
            gender     = str(row["Gender"]).strip().upper()
 
            filename    = f"{sound_id}.wav"
            wav_path    = hs_folder / filename
            if not wav_path.exists():
                print(f"[HLS] ⚠ WAV no encontrado: {wav_path}")
 
            pathology   = HLS_HEART_PATHOLOGY_MAP.get(sound_type, "unknown")
            loc_unified = HLS_LOCATION_MAP.get(location, location.lower())
            file_id     = f"HLS_{counter:05d}.wav"
            counter    += 1
 
            rows.append({
                "file_id":               file_id,
                "source_db":             "HLS",
                "original_filename":     filename,
                "subject_id":            pd.NA,  # maniquí: no hay sujeto real que agrupar
                "subject_type":          "manikin",
                "sex":                   gender if gender in ("M", "F") else pd.NA,
                "age_category":          pd.NA,
                "age_years":             pd.NA,
                "height_cm":             pd.NA,
                "weight_kg":             pd.NA,
                "bmi":                   pd.NA,
                "sound_type":            "cardiac",
                "pathology_label":       pathology,
                "auscultation_location": loc_unified,
                "equipment":             "3M Littmann CORE Digital Stethoscope",
                "sample_rate_hz":        22050,
                "duration_sec":          15.0,
                "has_crackles":          pd.NA,
                "has_wheezes":           pd.NA,
            })
    else:
        print(f"[HLS] ⚠ No se encontró HS.csv en {hs_folder}")
 
    # ── Sonidos pulmonares ────────────────────────────────────────────────────
    ls_folder = HLS_ROOT / "LS"
    ls_csv    = ls_folder / "LS.csv"
 
    if ls_csv.exists():
        df_ls = _read_hls_csv(ls_csv)
        print(f"[HLS] LS.csv — {len(df_ls)} registros.")
 
        for _, row in df_ls.iterrows():
            sound_id   = str(row["Lung Sound ID"]).strip()
            sound_type = str(row["Lung Sound Type"]).strip()
            location   = str(row["Location"]).strip()
            gender     = str(row["Gender"]).strip().upper()
 
            filename    = f"{sound_id}.wav"
            wav_path    = ls_folder / filename
            if not wav_path.exists():
                print(f"[HLS] ⚠ WAV no encontrado: {wav_path}")
 
            pathology   = HLS_LUNG_PATHOLOGY_MAP.get(sound_type, "unknown")
            loc_unified = HLS_LOCATION_MAP.get(location, location.lower())
            file_id     = f"HLS_{counter:05d}.wav"
            counter    += 1
 
            rows.append({
                "file_id":               file_id,
                "source_db":             "HLS",
                "original_filename":     filename,
                "subject_id":            pd.NA,  # maniquí: no hay sujeto real que agrupar
                "subject_type":          "manikin",
                "sex":                   gender if gender in ("M", "F") else pd.NA,
                "age_category":          pd.NA,
                "age_years":             pd.NA,
                "height_cm":             pd.NA,
                "weight_kg":             pd.NA,
                "bmi":                   pd.NA,
                "sound_type":            "pulmonary",
                "pathology_label":       pathology,
                "auscultation_location": loc_unified,
                "equipment":             "3M Littmann CORE Digital Stethoscope",
                "sample_rate_hz":        22050,
                "duration_sec":          15.0,
                "has_crackles":          pd.NA,
                "has_wheezes":           pd.NA,
            })
    else:
        print(f"[HLS] ⚠ No se encontró LS.csv en {ls_folder}")
 
    df = pd.DataFrame(rows, columns=MASTER_COLUMNS)
    print(f"[HLS] Total procesados: {len(df)} registros.\n")
    return df, counter
 
 
if __name__ == "__main__":
    df, _ = parse_hls()
    print(df[["file_id", "sound_type", "pathology_label",
              "auscultation_location", "sex"]].to_string())
    print("\nDistribución de patologías:")
    print(df["pathology_label"].value_counts())