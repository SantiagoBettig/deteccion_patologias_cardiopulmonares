"""
config.py
=========
Rutas base y constantes compartidas por todos los parsers.
Ajustar ROOT_DATA a la ubicación real de los datasets en tu máquina.
"""

from pathlib import Path

# ── Rutas base ────────────────────────────────────────────────────────────────
# ROOT_DATA es la carpeta "desarrollo/" del repo, calculada de forma relativa
# a este archivo (config.py está en desarrollo/src/unified_dataset/).
ROOT_DATA    = Path(__file__).resolve().parents[2]
DATA_ROOT    = ROOT_DATA / "data" / "raw"
HLS_ROOT     = DATA_ROOT / "HLS-CMDS" / "Dataset"
CIRCOR_ROOT  = DATA_ROOT / "CirCor"
ICBHI_ROOT   = DATA_ROOT / "ICBHI"

# v2: dataset unificado sin HLS-CMDS, con subject_id agregado.
# Se guarda en una carpeta separada de "unified/" (v1, con HLS) para no
# pisar el dataset anterior mientras se valida el nuevo.
UNIFIED_DIR  = ROOT_DATA / "data" / "unified_v2"
AUDIO_DIR    = UNIFIED_DIR / "audio"
MASTER_CSV   = UNIFIED_DIR / "unified_dataset_v2.csv"

# ── Prefijos para file_id ─────────────────────────────────────────────────────
PREFIX = {
    "HLS": "HLS",
    "CIR": "CIR",
    "ICB": "ICB",
}

# ── Mapeo de etiquetas HLS ────────────────────────────────────────────────────
HLS_SOUND_TYPE_MAP = {
    # cardíacos
    "NH":  "cardiac",
    "LDM": "cardiac",
    "MSM": "cardiac",
    "LSM": "cardiac",
    "AF":  "cardiac",
    "S4":  "cardiac",
    "ESM": "cardiac",
    "S3":  "cardiac",
    "T":   "cardiac",
    "AVB": "cardiac",
    # pulmonares
    "NL":  "pulmonary",
    "W":   "pulmonary",
    "FC":  "pulmonary",
    "CC":  "pulmonary",
    "R":   "pulmonary",
    "PR":  "pulmonary",
}

HLS_PATHOLOGY_MAP = {
    "NH":  "normal_heart",
    "LDM": "heart_murmur",
    "MSM": "heart_murmur",
    "LSM": "heart_murmur",
    "ESM": "heart_murmur",
    "AF":  "atrial_fibrillation",
    "S4":  "extra_heart_sound",
    "S3":  "extra_heart_sound",
    "T":   "tachycardia",
    "AVB": "av_block",
    "NL":  "normal_lung",
    "W":   "wheezing",
    "FC":  "crackles",
    "CC":  "crackles",
    "R":   "rhonchi",
    "PR":  "pleural_rub",
}

# ── Mapeo de diagnósticos ICBHI ───────────────────────────────────────────────
ICBHI_DIAGNOSIS_MAP = {
    "Healthy":        "normal_lung",
    "COPD":           "copd",
    "Pneumonia":      "pneumonia",
    "Bronchiectasis": "bronchiectasis",
    "URTI":           "respiratory_infection",
    "LRTI":           "respiratory_infection",
    "Bronchiolitis":  "bronchiolitis_asthma",
    "Asthma":         "bronchiolitis_asthma",
}

# ── Mapeo de ubicaciones anatómicas al vocabulario unificado ──────────────────
# Cardíacas
LOCATION_MAP = {
    # HLS — cardíacas
    "A":    "apex",
    "RUSB": "right_upper_sternal_border",
    "LUSB": "left_upper_sternal_border",
    "LLSB": "left_lower_sternal_border",
    "RC":   "right_costal_margin",
    "LC":   "left_costal_margin",
    # HLS — pulmonares
    "RUA":  "right_upper_anterior",
    "RMA":  "right_mid_anterior",
    "RLA":  "right_lower_anterior",
    "LUA":  "left_upper_anterior",
    "LMA":  "left_mid_anterior",
    "LLA":  "left_lower_anterior",
    # CirCor
    "AV":   "right_upper_sternal_border",   # aortic valve point
    "PV":   "left_upper_sternal_border",    # pulmonic valve point
    "TV":   "left_lower_sternal_border",    # tricuspid valve point
    "MV":   "apex",                         # mitral valve point
    "Phc":  "other",
    # ICBHI
    "Tc":   "trachea",
    "Al":   "anterior_left",
    "Ar":   "anterior_right",
    "Pl":   "posterior_left",
    "Pr":   "posterior_right",
    "Ll":   "lateral_left",
    "Lr":   "lateral_right",
}

# ── Columnas del CSV maestro (orden canónico) ─────────────────────────────────
MASTER_COLUMNS = [
    # identificación y auditoría
    "file_id",
    "source_db",
    "original_filename",
    # características del sujeto
    "subject_id",         # ID de sujeto en la fuente original (para split sin leakage)
    "subject_type",       # real | manikin
    "sex",
    "age_category",       # neonate | infant | child | adolescent | adult
    "age_years",
    "height_cm",
    "weight_kg",
    "bmi",
    # características de la grabación
    "sound_type",         # cardiac | pulmonary
    "pathology_label",    # clase unificada
    "clinical_outcome",   # normal | abnormal — solo CirCor, ver parsers/parse_circor.py
    "auscultation_location",
    "equipment",
    "sample_rate_hz",
    "duration_sec",
    # features auxiliares (solo ICBHI)
    "has_crackles",       # True | False | NaN
    "has_wheezes",        # True | False | NaN
]
