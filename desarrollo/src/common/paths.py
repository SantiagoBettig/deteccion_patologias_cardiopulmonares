"""
common/paths.py
================
Rutas compartidas entre splits/, preprocessing/ y gate_model/.
Todas relativas a este archivo, para que el repo funcione en cualquier máquina.
"""

import os
from pathlib import Path

SRC_ROOT        = Path(__file__).resolve().parents[1]        # desarrollo/src
DESARROLLO_ROOT = SRC_ROOT.parent                             # desarrollo/
DATA_DIR        = DESARROLLO_ROOT / "data"
REPORTS_DIR     = DESARROLLO_ROOT / "reports"

# ── Dataset unificado v2 (sin HLS-CMDS, con subject_id) ────────────────────────
UNIFIED_V2_DIR = DATA_DIR / "unified_v2"
MASTER_CSV_V2  = UNIFIED_V2_DIR / "unified_dataset_v2.csv"
AUDIO_DIR_V2   = UNIFIED_V2_DIR / "audio"
SPLITS_CSV     = UNIFIED_V2_DIR / "splits.csv"

# ── Modelo de patología cardíaca (etapa 2) ─────────────────────────────────────
# Split independiente por dominio (ver splits/make_cardiac_splits.py) y
# ventanas por ciclo cardíaco (ver preprocessing/make_cardiac_cycle_windows.py).
CARDIAC_SPLITS_CSV       = UNIFIED_V2_DIR / "cardiac_splits.csv"
CARDIAC_SPLITS_REPORTS_DIR = REPORTS_DIR / "splits_cardiac"
CARDIAC_DIR              = DATA_DIR / "cardiac"
CARDIAC_WINDOWS_DIR      = CARDIAC_DIR / "windows"
CARDIAC_WINDOWS_INDEX_CSV = CARDIAC_DIR / "windows_index.csv"
CARDIAC_REPORTS_DIR      = REPORTS_DIR / "cardiac"
CARDIAC_CHECKPOINTS_DIR  = SRC_ROOT / "cardiac_model" / "checkpoints"

# Ventanas con más solapamiento SOLO en train (hop=1 ciclo en vez de 2) —
# val/test quedan idénticas a CARDIAC_WINDOWS_DIR (mismo hop=2), para que
# la comparación contra v1-v5 del modelo de murmullo sea limpia (ver
# preprocessing/make_cardiac_cycle_windows_dense_train.py y
# avances/10_cardiaco_split_dos_tareas.md, sección 12). Carpeta separada
# para no tocar las ventanas ya usadas por cardiac_model/ y
# cardiac_outcome_model/.
CARDIAC_DENSE_DIR              = DATA_DIR / "cardiac_murmur_dense"
CARDIAC_DENSE_WINDOWS_DIR      = CARDIAC_DENSE_DIR / "windows"
CARDIAC_DENSE_WINDOWS_INDEX_CSV = CARDIAC_DENSE_DIR / "windows_index.csv"

# cardiac_model/ (3 clases) quedó como evidencia del experimento descartado
# — ver avances/10_cardiaco_split_dos_tareas.md. Reemplazado por dos tareas
# binarias independientes, cada una con su propio model.py/dataset.py para
# poder ajustar arquitectura/hiperparámetros de una sin afectar a la otra.
CARDIAC_MURMUR_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints"
CARDIAC_MURMUR_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur"

# v1: baseline (MFCC + pos_weight x1.4 + checkpoint por F2 + dropout + umbral
# calibrado, recall 0.75 en test) — congelado como evidencia antes de probar
# log-Mel (ver avances/10_cardiaco_split_dos_tareas.md, sección 6).
CARDIAC_MURMUR_V1_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v1_mfcc"
CARDIAC_MURMUR_V1_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v1_mfcc"

# v2: log-Mel en vez de MFCC, mismo resto de la receta que v1 (ver
# cardiac_murmur_model/run_v2_logmel.py)
CARDIAC_MURMUR_V2_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v2_logmel"
CARDIAC_MURMUR_V2_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v2_logmel"

# v3: MFCC (igual que v1, perdió v2 en la comparación) + ReduceLROnPlateau
# sobre val_f2 (ver cardiac_murmur_model/run_v3_lr_scheduler.py)
CARDIAC_MURMUR_V3_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v3_lr_scheduler"
CARDIAC_MURMUR_V3_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v3_lr_scheduler"

# v4: MFCC (igual que v1, sigue siendo la mejor versión) + metadata del
# paciente (sexo/edad/altura/peso) fusionada con el embedding de audio (ver
# cardiac_murmur_model/run_v4_metadata.py)
CARDIAC_MURMUR_V4_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v4_metadata"
CARDIAC_MURMUR_V4_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v4_metadata"

# v5: MFCC (igual que v1) + FocalLoss en vez de BCE+pos_weight (ver
# cardiac_murmur_model/run_v5_focal_loss.py)
CARDIAC_MURMUR_V5_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v5_focal_loss"
CARDIAC_MURMUR_V5_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v5_focal_loss"

# v6: MFCC (igual que v1) + ubicación de auscultación como feature (ver
# cardiac_murmur_model/run_v6_location.py)
CARDIAC_MURMUR_V6_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v6_location"
CARDIAC_MURMUR_V6_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v6_location"

# v7: mejor receta de v1/v5/v6 + ventanas densas en train (ver
# cardiac_murmur_model/run_v7_dense_train.py)
CARDIAC_MURMUR_V7_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_murmur_model" / "checkpoints_v7_dense_train"
CARDIAC_MURMUR_V7_REPORTS_DIR      = REPORTS_DIR / "cardiac_murmur" / "runs" / "v7_dense_train"

CARDIAC_OUTCOME_CHECKPOINTS_DIR = SRC_ROOT / "cardiac_outcome_model" / "checkpoints"
CARDIAC_OUTCOME_REPORTS_DIR      = REPORTS_DIR / "cardiac_outcome"

# ── Modelo pulmonar: sonidos adventicios por ciclo (etapa 2) ───────────────────
# Un WAV por ciclo respiratorio anotado en los .txt de ICBHI (ver
# preprocessing/make_pulmonary_cycles.py). Carpeta plana, sin subcarpetas por
# split: la evaluación es k-fold por paciente desde el inicio (ver
# splits/make_pulmonary_kfold.py), así que el split no es fijo por archivo.
PULMONARY_DIR               = DATA_DIR / "pulmonary"
PULMONARY_CYCLES_DIR        = PULMONARY_DIR / "cycles"
PULMONARY_CYCLES_INDEX_CSV  = PULMONARY_DIR / "cycles_index.csv"
PULMONARY_KFOLD_DIR         = UNIFIED_V2_DIR / "pulmonary_kfold"
PULMONARY_ADV_CHECKPOINTS_DIR = SRC_ROOT / "pulmonary_adventitious_model" / "checkpoints"
PULMONARY_ADV_REPORTS_DIR     = REPORTS_DIR / "pulmonary_adventitious"


# Pesos preentrenados de terceros (no versionados). PANNs CNN6 (Kong et al.
# 2020), Zenodo doi:10.5281/zenodo.3987831, CC-BY-4.0 — ver avances/15.
PRETRAINED_DIR  = DATA_DIR / "pretrained"
PANNS_CNN6_CKPT = PRETRAINED_DIR / "Cnn6_mAP=0.343.pth"


def pulmonary_cycles_paths(variant: str | None = None) -> tuple[Path, Path]:
    """(carpeta de WAVs, CSV índice) de los ciclos pulmonares. variant=None
    es el preprocesamiento base; una variante (ej. "hp100": pasa-altos a
    100 Hz) vive en data/pulmonary_<variant>/ para no pisar los ciclos que
    usan las corridas anteriores."""
    base = PULMONARY_DIR if variant is None else DATA_DIR / f"pulmonary_{variant}"
    return base / "cycles", base / "cycles_index.csv"

# ── Datos del modelo gate ──────────────────────────────────────────────────────
# GATE_DIR se puede sobreescribir con la variable de entorno GATE_DATA_DIR para
# apuntar a una copia local rápida de las ventanas (p. ej. en Colab, donde leer
# miles de archivos .wav chicos desde un Drive montado por FUSE es muchísimo
# más lento que desde el disco local de la VM). Sin la variable, el
# comportamiento es el mismo de siempre.
GATE_DIR              = Path(os.environ.get("GATE_DATA_DIR", str(DATA_DIR / "gate")))
GATE_WINDOWS_DIR      = GATE_DIR / "windows"
GATE_WINDOWS_INDEX_CSV = GATE_DIR / "windows_index.csv"

# Validación externa del gate con HLS-CMDS (ver preprocessing/make_hls_windows.py)
GATE_HLS_INDEX_CSV = GATE_DIR / "hls_external_index.csv"

# ── Checkpoints y reportes del modelo gate ─────────────────────────────────────
GATE_CHECKPOINTS_DIR = SRC_ROOT / "gate_model" / "checkpoints"
GATE_REPORTS_DIR      = REPORTS_DIR / "gate"
SPLITS_REPORTS_DIR    = REPORTS_DIR / "splits"
PREPROCESSING_REPORTS_DIR = REPORTS_DIR / "preprocessing"

# Diagnóstico de sesgo: clasificador auxiliar que predice source_db en vez
# de sound_type, sobre las mismas ventanas (ver gate_model/diagnostics_source_db.py)
GATE_DIAG_CHECKPOINTS_DIR = SRC_ROOT / "gate_model" / "checkpoints_source_diag"
GATE_DIAG_REPORTS_DIR      = REPORTS_DIR / "gate_diagnostics" / "source_db"

# Gate v5: MFCC en vez de espectrograma log-Mel (ver gate_model/run_v5_mfcc.py
# y avances/6_gate_v5_mfcc.md), carpetas separadas para no pisar el checkpoint
# vigente de v4.
GATE_V5_CHECKPOINTS_DIR = SRC_ROOT / "gate_model" / "checkpoints_v5_mfcc"
GATE_V5_REPORTS_DIR      = REPORTS_DIR / "gate" / "runs" / "v5_mfcc"
