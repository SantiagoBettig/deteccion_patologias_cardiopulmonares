# Comparación de bases de datos — Proyecto final CEIA

## Bases de datos utilizadas

| | HLS-CMDS | CirCor DigiScope v1.0.3 | ICBHI 2017 (Respiratory Sound DB) |
|---|---|---|---|
| **Tipo de sonido** | Cardíaco + Pulmonar (mixto) | Cardíaco (PCG) | Pulmonar |
| **Grabaciones** | 535 (v2) | 5272 | 920 |
| **Sujetos** | Maniquí clínico (no pacientes) | 1568 sujetos | 126 pacientes |
| **Rango etario** | N/A | 0–21 años (pediátrico) | Todas las edades |
| **Sample rate** | No especificado públicamente | 4000 Hz (uniforme) | 4k / 10k / 44.1k Hz (mixto) |
| **Formato audio** | WAV | WAV | WAV |

---

## HLS-CMDS

**Referencia:** Y. Torabi, S. Shirani and J. P. Reilly, *IEEE Data Descriptions*, doi: 10.1109/IEEEDATA.2025.3566012

**Repositorio:** https://github.com/torabiy/hls-cmds

### Contenido (v2)
- 50 grabaciones de sonidos cardíacos
- 50 grabaciones de sonidos pulmonares
- 145 grabaciones mixtas (cardíaco + pulmonar simultáneos)
- Para cada grabación mixta: fuente cardíaca (145) y fuente pulmonar (145) por separado
- **Total: 535 grabaciones**

### Clases de sonido

**Cardíacos:**
- Normal Heart
- Late Diastolic Murmur
- Mid Systolic Murmur
- Late Systolic Murmur
- Atrial Fibrillation
- Fourth Heart Sound (S4)
- Early Systolic Murmur
- Third Heart Sound (S3)
- Tachycardia
- Atrioventricular Block

**Pulmonares:**
- Normal Lung
- Wheezing
- Fine Crackles
- Coarse Crackles
- Rhonchi
- Pleural Rub

### Ubicaciones anatómicas (12 puntos)
Right Upper Sternal Border, Left Upper Sternal Border, Lower Left Sternal Border,
Right Costal Margin, Left Costal Margin, Apex,
Right Upper Anterior, Left Upper Anterior, Right Mid Anterior,
Left Mid Anterior, Right Lower Anterior, Left Lower Anterior

### Archivos de anotación
- La patología se infiere de la **estructura de carpetas / nombre de archivo** (no hay CSV de metadatos propio)
- Scripts auxiliares: `audio_plotter.py`, `audio_spectrogram.py`, `Donut_chart.py`
- Notebook de ejemplo: `HLS_CMDS.ipynb`

### ⚠ Consideraciones importantes
- Grabaciones realizadas en **maniquí clínico**, no en pacientes reales
- No hay datos demográficos individuales (edad, sexo, peso, talla)

---

## CirCor DigiScope Phonocardiogram Dataset v1.0.3

**Referencia:** Oliveira et al., *PhysioNet*, doi: 10.13026/tshs-mw03

**URL:** https://physionet.org/content/circor-heart-sound/1.0.3/

**Licencia:** Open Data Commons Attribution License v1.0

### Contenido
- 5272 grabaciones de 1568 sujetos
- Duración: 4.8 a 80.4 segundos (media: 22.9 s)
- Total: más de 33.5 horas de grabación
- Población pediátrica: 0–21 años (media: 6.1 años)

### Clases / anotaciones
No se etiquetan patologías con nombre de enfermedad. Las anotaciones describen:

- **Murmur:** `Present` / `Absent` / `Unknown`
- **Murmur locations:** combinación de `AV`, `PV`, `TV`, `MV`, `Phc`
- **Most audible location**
- **Systolic murmur:** timing, shape, pitch, grading (Levine I–III/VI), quality
- **Diastolic murmur:** timing, shape, pitch, grading (I–III/IV), quality
- **Outcome:** `Normal` / `Abnormal` (diagnóstico global del cardiólogo)

### Datos demográficos disponibles
| Campo | Valores posibles |
|---|---|
| Age | Neonate / Infant / Child / Adolescent / Young Adult |
| Sex | Female / Male |
| Height | cm (número) |
| Weight | kg (número) |
| Pregnancy status | True / False |

### Ubicaciones de auscultación
- `AV`: Aortic Valve (2nd intercostal space, right sternal border)
- `PV`: Pulmonic Valve (2nd intercostal space, left sternal border)
- `TV`: Tricuspid Valve (left lower sternal border)
- `MV`: Mitral Valve (5th intercostal space, midclavicular line)
- `Phc`: Otra ubicación

### Estructura de archivos
```
ABCDE_XY[_n].wav      → audio (por sujeto y ubicación)
ABCDE_XY[_n].hea      → header WFDB
ABCDE_XY[_n].tsv      → segmentación S1/S2 (timestamps en segundos)
ABCDE.txt             → metadatos del sujeto (shared entre todas sus grabaciones)
training_data.csv     → resumen de todos los sujetos en columnas
```

**Convención de nombres:** `ABCDE` = ID numérico del sujeto, `XY` = código de ubicación

**Formato TSV (segmentación):**
| Columna | Descripción |
|---|---|
| Col 1 | Inicio del segmento (segundos) |
| Col 2 | Fin del segmento (segundos) |
| Col 3 | Tipo: `1`=S1, `2`=sístole, `3`=S2, `4`=diástole, `0`=no anotado |

---

## Respiratory Sound Database — ICBHI 2017

**Referencia:** Rocha BM et al. (2019), *Physiological Measurement* 40 035001

**URL:** https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database

### Contenido
- 920 grabaciones de 126 pacientes
- 6898 ciclos respiratorios anotados
- Duración: 10–90 segundos por grabación
- Total: 5.5 horas de audio

### Clases de ciclos respiratorios
| Clase | Cantidad de ciclos |
|---|---|
| Normal | 3642 |
| Crackles | 1864 |
| Wheezes | 886 |
| Both (crackles + wheezes) | 506 |

### Diagnósticos de pacientes
- Healthy
- COPD (Chronic Obstructive Pulmonary Disease)
- Pneumonia
- Bronchiectasis
- Bronchiolitis
- URTI (Upper Respiratory Tract Infection)
- LRTI (Lower Respiratory Tract Infection)
- Asthma

### Datos demográficos disponibles
| Campo | Descripción |
|---|---|
| Patient number | ID (101–226) |
| Age | Años |
| Sex | M/F |
| Adult BMI | kg/m² |
| Child Weight | kg |
| Child Height | cm |

### Ubicaciones de auscultación (7 puntos torácicos)
`Tc` (Trachea), `Al` (Anterior left), `Ar` (Anterior right),
`Pl` (Posterior left), `Pr` (Posterior right), `Ll` (Lateral left), `Lr` (Lateral right)

### Equipos de grabación
- `AKGC417L`: AKG C417L Microphone
- `LittC2SE`: 3M Littmann Classic II SE Stethoscope
- `Litt3200`: 3M Littmann 3200 Electronic Stethoscope
- `Meditron`: WelchAllyn Meditron Master Elite Electronic Stethoscope

### Estructura de archivos
```
PAT_IDX_LOC_MODE_EQUIP.wav     → audio
PAT_IDX_LOC_MODE_EQUIP.txt     → anotación de ciclos respiratorios
demographic_info.csv            → datos demográficos de pacientes
patient_diagnosis.csv           → diagnóstico por paciente
filename_differences.txt        → lista de archivos con nombres corregidos (puede ignorarse)
```

**Convención de nombre:** 5 elementos separados por `_`
1. Patient number (ej: `101`)
2. Recording index (ej: `1u`)
3. Chest location (ej: `Tc`)
4. Acquisition mode: `sc` (single channel) / `mc` (multichannel)
5. Recording equipment (ej: `Meditron`)

**Formato TXT de anotación (por grabación):**
| Columna | Descripción |
|---|---|
| Col 1 | Inicio del ciclo respiratorio (segundos) |
| Col 2 | Fin del ciclo respiratorio (segundos) |
| Col 3 | Presencia de crackles: `1` / `0` |
| Col 4 | Presencia de wheezes: `1` / `0` |

---

## Resumen de heterogeneidades conocidas

| Aspecto | HLS-CMDS | CirCor | ICBHI 2017 |
|---|---|---|---|
| Sujeto real | ❌ Maniquí | ✅ | ✅ |
| Datos demográficos | ❌ | ✅ Completos | ✅ Parciales |
| Diagnóstico explícito | Por carpeta | Outcome Normal/Abnormal | Por paciente (8 clases) |
| Anotaciones temporales | ❌ | ✅ S1/S2 (TSV) | ✅ Ciclos respiratorios (TXT) |
| Sample rate uniforme | Desconocido | ✅ 4000 Hz | ❌ Mixto |
| Tipo de sonido | Cardíaco + Pulmonar | Cardíaco | Pulmonar |

---

## Decisiones de diseño para el dataset unificado

### Columnas obligatorias (presentes en todos los registros)
- `file_id`: nombre asignado en el dataset unificado (ej: `HLS_00001.wav`)
- `source_db`: base de datos de origen (`HLS`, `CIR`, `ICB`)
- `original_filename`: nombre original del archivo
- `sound_type`: `cardiac` / `pulmonary` / `mixed`
- `pathology_label`: etiqueta de patología unificada (a definir)
- `auscultation_location`: ubicación anatómica (vocabulario a unificar)

### Columnas opcionales (presentes según la base de datos)
- `age_category` / `age_years`
- `sex`
- `height_cm`
- `weight_kg`
- `bmi`
- `pregnancy`
- `equipment`
- `sample_rate`
- `duration_sec`