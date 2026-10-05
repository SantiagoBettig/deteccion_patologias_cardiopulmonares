# Proyecto Final CEIA — Clasificación de Patologías Cardiopulmonares
## Resumen de sesión de trabajo

---

## 1. Descripción del proyecto

El objetivo es desarrollar un pipeline de machine learning para clasificar
patologías cardiopulmonares a partir de audio obtenido con estetoscopio digital.
El proyecto se enmarca en la Carrera de Especialización en Inteligencia Artificial
(CEIA) de FIUBA.

**Objetivo primario:** clasificar patologías a nivel de grabación completa.
**Objetivo secundario (futuro):** clasificar tipos de sonido (wheeze, crackle, etc.)

---

## 2. Bases de datos utilizadas

### 2.1 HLS-CMDS v2
- **Fuente:** https://github.com/torabiy/hls-cmds
- **Referencia:** Y. Torabi, S. Shirani, J. P. Reilly. *IEEE Data Descriptions*, doi: 10.1109/IEEEDATA.2025.3566012
- **Contenido:** 535 grabaciones (50 cardíacas + 50 pulmonares + 145 mixtas + 145 fuentes cardíacas + 145 fuentes pulmonares)
- **Sujeto:** maniquí clínico (no pacientes reales)
- **Sample rate:** 22,050 Hz, duración fija de 15 segundos
- **Archivos de metadatos:** `HS.csv`, `LS.csv`, `Mix.csv`
  - Columnas HS.csv: `Gender`, `Heart Sound Type`, `Location`, `Heart Sound ID`
  - Columnas LS.csv: `Gender`, `Lung Sound Type`, `Location`, `Lung Sound ID`
  - El nombre del WAV se construye como `{Sound ID}.wav`

### 2.2 CirCor DigiScope Phonocardiogram Dataset v1.0.3
- **Fuente:** https://physionet.org/content/circor-heart-sound/1.0.3/
- **Referencia:** Oliveira et al. *PhysioNet*, doi: 10.13026/tshs-mw03
- **Licencia:** Open Data Commons Attribution License v1.0
- **Contenido:** 5272 grabaciones de 1568 sujetos pediátricos (0–21 años)
- **Sample rate:** 4000 Hz uniforme
- **Archivos de metadatos:** `training_data.csv` (sep=`;`), `.txt` por sujeto, `.tsv` por grabación (segmentación S1/S2), `.hea` por grabación (formato WFDB)
- **Convención de nombres WAV:** `ABCDE_XY[_n].wav` donde `ABCDE` = ID de sujeto, `XY` = ubicación (AV/PV/TV/MV/Phc)

### 2.3 Respiratory Sound Database — ICBHI 2017
- **Fuente:** https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- **Referencia:** Rocha BM et al. (2019). *Physiological Measurement* 40 035001
- **Contenido:** 920 grabaciones de 126 pacientes, 6898 ciclos respiratorios anotados
- **Sample rate:** mixto (4k / 10k / 44.1k Hz según equipo)
- **Archivos de metadatos:**
  - `demographic_info.csv` (sep=`;`, sin header, 6 columnas: patient_id, age, sex, adult_bmi, child_weight, child_height)
  - `patient_diagnosis.csv` (sep=`;`, sin header, 2 columnas: patient_id, diagnosis)
  - `.txt` por grabación (anotación de ciclos respiratorios)
- **Convención de nombres WAV:** `PAT_IDX_LOC_MODE_EQUIP.wav` (5 elementos separados por `_`)

---

## 3. Decisiones de diseño

### 3.1 Estructura del proyecto

```
proyecto/
├── data/
│   ├── raw/
│   │   ├── HLS-CMDS/Dataset/
│   │   │   ├── HS/  (HS.csv + WAVs)
│   │   │   └── LS/  (LS.csv + WAVs)
│   │   ├── CirCor/
│   │   │   ├── training_data/  (WAVs + .hea + .tsv + .txt)
│   │   │   └── training_data.csv
│   │   └── ICBHI/
│   │       ├── audio_and_txt_files/  (WAVs + .txt de ciclos)
│   │       ├── demographic_info.csv
│   │       └── patient_diagnosis.csv
│   └── unified/
│       ├── audio/      ← WAVs renombrados con file_id opaco
│       └── master.csv  ← CSV maestro unificado
├── parsers/
│   ├── parse_hls.py
│   ├── parse_circor.py
│   ├── parse_icbhi.py
│   └── build_dataset.py
└── config.py
```

### 3.2 Formato de los WAVs unificados

- Los WAVs se **copian sin modificar** al directorio `unified/audio/`.
- No se resamplea, no se cambia bit depth, no se convierte a mono.
- La normalización queda para el pipeline de features (etapa posterior).
- El nombre asignado es un **identificador opaco** con prefijo de origen:
  - `HLS_00001.wav`, `CIR_00042.wav`, `ICB_00315.wav`

### 3.3 Columnas del CSV maestro

| Columna | Tipo | Descripción |
|---|---|---|
| `file_id` | str | Nombre asignado en el dataset unificado |
| `source_db` | str | Origen: `HLS` / `CIR` / `ICB` — **solo auditoría, no entra al modelo** |
| `original_filename` | str | Nombre original del archivo — **solo auditoría, no entra al modelo** |
| `subject_type` | str | `real` / `manikin` |
| `sex` | str | `M` / `F` |
| `age_category` | str | `neonate` / `infant` / `child` / `adolescent` / `adult` |
| `age_years` | float | Edad numérica (cuando disponible) |
| `height_cm` | float | Altura en cm |
| `weight_kg` | float | Peso en kg |
| `bmi` | float | BMI (solo adultos ICBHI) |
| `sound_type` | str | `cardiac` / `pulmonary` |
| `pathology_label` | str | Clase unificada (ver sección 3.4) |
| `auscultation_location` | str | Ubicación anatómica (vocabulario unificado) |
| `equipment` | str | Modelo de estetoscopio |
| `sample_rate_hz` | int | Frecuencia de muestreo del WAV |
| `duration_sec` | float | Duración en segundos |
| `has_crackles` | bool | Solo ICBHI: presencia de crackles en algún ciclo |
| `has_wheezes` | bool | Solo ICBHI: presencia de wheezes en algún ciclo |

### 3.4 Vocabulario unificado de patologías

**Cardíacas:**

| Clase unificada | Fuentes |
|---|---|
| `normal_heart` | HLS: Normal / CirCor: Murmur=Absent + Outcome=Normal |
| `heart_murmur` | HLS: LSM, MSM, ESM, LDM / CirCor: Murmur=Present |
| `atrial_fibrillation` | HLS: Atrial Fibrillation |
| `extra_heart_sound` | HLS: S3, S4 |
| `tachycardia` | HLS: Tachycardia |
| `av_block` | HLS: AV Block |
| `abnormal_heart_unspecified` | CirCor: Outcome=Abnormal + Murmur=Absent o Unknown |

**Pulmonares:**

| Clase unificada | Fuentes |
|---|---|
| `normal_lung` | HLS: Normal / ICBHI: Healthy |
| `wheezing` | HLS: Wheezing |
| `crackles` | HLS: Fine Crackles, Coarse Crackles |
| `rhonchi` | HLS: Rhonchi |
| `pleural_rub` | HLS: Pleural Rub |
| `copd` | ICBHI: COPD |
| `pneumonia` | ICBHI: Pneumonia |
| `bronchiectasis` | ICBHI: Bronchiectasis |
| `respiratory_infection` | ICBHI: URTI + LRTI (fusionadas por baja ocurrencia) |
| `bronchiolitis_asthma` | ICBHI: Bronchiolitis + Asthma (fusionadas por baja ocurrencia) |

### 3.5 Decisiones sobre HLS-CMDS

- Se **mantiene** en el dataset pese a ser grabado en maniquí, porque es la
  única fuente con clases cardíacas nombradas clínicamente (FA, S3, S4, etc.).
- El campo `subject_type = manikin` permite filtrarlos en etapas posteriores.
- Los **sonidos mixtos** (carpeta `Mix/`) se excluyen en esta etapa.
  Pueden incorporarse en el futuro como técnica de data augmentation.

### 3.6 Decisiones sobre desbalance de clases

- El desbalance **no se corrige** en la etapa de construcción del dataset.
- Se maneja en el pipeline de entrenamiento (oversampling, class weights, etc.).
- Las clases con muy baja ocurrencia en ICBHI se fusionan:
  - URTI + LRTI → `respiratory_infection`
  - Bronchiolitis + Asthma → `bronchiolitis_asthma`

### 3.7 Trazabilidad vs. features del modelo

- Las columnas `source_db` y `original_filename` **se incluyen en el CSV**
  para auditoría y depuración, pero se excluyen explícitamente del pipeline
  de entrenamiento. La exclusión se hace en el pipeline de ML, no en el CSV.

---

## 4. Scripts desarrollados

### `config.py`
Rutas base, prefijos de `file_id`, y todos los diccionarios de mapeo:
tipos de sonido, patologías, ubicaciones anatómicas y columnas del CSV maestro.

### `parsers/parse_hls.py`
- Lee `HS.csv` y `LS.csv`.
- Construye el nombre del WAV desde `Heart Sound ID` / `Lung Sound ID`.
- Mapea tipos de sonido desde texto completo a `pathology_label`.
- Produce 100 registros (50 cardíacos + 50 pulmonares).

### `parsers/parse_circor.py`
- Lee `training_data.csv` (sep=`;`).
- Expande de un registro por sujeto a un registro por WAV.
- Infiere `pathology_label` desde los campos `Murmur` y `Outcome`.
- Busca WAVs en `training_data/` usando glob con el patrón `ABCDE_XY*.wav`.

### `parsers/parse_icbhi.py`
- Lee `demographic_info.csv` con `header=None` y sep=`;`.
- Lee `patient_diagnosis.csv` y construye un lookup por `patient_id`.
- Por cada WAV, parsea el `.txt` de anotación para determinar
  `has_crackles` y `has_wheezes`.
- Infiere `age_category` desde `age_years`.
- Asigna `sample_rate_hz` según el equipo de grabación.

### `parsers/build_dataset.py`
Script integrador. Llama a los tres parsers en secuencia, concatena los
DataFrames, imprime un resumen estadístico, guarda `master.csv` y copia
los WAVs renombrados a `unified/audio/`.

Flags de control al inicio del archivo:
- `COPY_WAVS = True/False` — habilita/deshabilita la copia de WAVs.
- `DRY_RUN = True/False` — muestra estadísticas sin escribir nada.

---

## 5. Bugs encontrados y corregidos

| Script | Problema | Causa | Solución |
|---|---|---|---|
| `parse_hls.py` | `KeyError: 'File Name'` | La columna no existe; el nombre del WAV se construye desde `Heart Sound ID` | Eliminar `COL_FILENAME`; construir filename como `f"{sound_id}.wav"` |
| `parse_hls.py` | Columnas distintas en HS y LS | HS usa `Heart Sound Type` / `Heart Sound ID`, LS usa `Lung Sound Type` / `Lung Sound ID` | Separar en dos bloques de parsing independientes |
| `parse_icbhi.py` | Datos demográficos no aparecían en `master.csv` | `_read_demographic` probaba separadores en orden incorrecto; el separador espacio "funcionaba" devolviendo una sola columna mal formada | Leer directamente con `sep=";"` y `header=None` con nombres explícitos |

---

## 6. Próximos pasos

- [ ] Verificar el `master.csv` generado: distribución de clases, porcentaje de NaN, integridad de rutas.
- [ ] EDA (Exploratory Data Analysis) sobre el dataset unificado.
- [ ] Definir la estrategia de extracción de features (MFCCs, espectrogramas, etc.).
- [ ] Definir el split train/validation/test con estratificación por `pathology_label`.
- [ ] Diseñar el pipeline de preprocesamiento de audio (resample, normalización, etc.).
- [ ] Selección y entrenamiento de modelos de ML.
