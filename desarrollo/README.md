# Detección de patologías cardiopulmonares — Desarrollo

Código del trabajo final de la Carrera de Especialización en Inteligencia
Artificial (CEIA, FIUBA). El objetivo y alcance completos están en
[`plan/planificacion_latex/charter.tex`](../plan/planificacion_latex/charter.tex);
el detalle de las decisiones tomadas durante el desarrollo está en
[`avances/`](../avances/). Este README cubre solo la parte de código: cómo
está organizada, qué hay hecho y cómo correrlo.

## Estructura

```
desarrollo/
├── .venv/                  # entorno virtual (no versionado, ver "Entorno" abajo)
├── requirements.txt
├── data/                   # NO versionado (ver "Datos" abajo)
│   ├── raw/                 # datasets originales, descargados a mano
│   ├── unified_v2/           # dataset unificado (CirCor + ICBHI, sin HLS-CMDS)
│   └── gate/                 # ventanas de audio del modelo gate
├── reports/                # NO versionado — gráficos/CSVs generados por los scripts
└── src/
    ├── common/
    │   └── paths.py          # rutas compartidas entre todos los paquetes
    ├── unified_dataset/
    │   ├── config.py         # rutas de datasets crudos + tablas de mapeo de etiquetas
    │   └── parsers/           # un parser por dataset (HLS, CirCor, ICBHI) + build_dataset.py
    ├── eda/
    │   └── dataset_eda.ipynb # EDA exploratorio del dataset unificado
    ├── splits/
    │   └── make_splits.py    # split train/val/test por subject_id
    ├── preprocessing/
    │   ├── audio_ops.py       # filtrado, resample, normalización RMS (compartido)
    │   ├── make_gate_windows.py  # ventaneo fijo (5s/50%) para el modelo gate
    │   ├── make_hls_windows.py   # ídem sobre HLS-CMDS, para validación externa
    │   └── plot_before_after.py  # gráficos de señal cruda vs. preprocesada
    ├── pulmonary_adventitious_model/
    │   ├── dataset.py, model.py       # ciclo -> log-Mel; CNN con 2 salidas (crackle, wheeze)
    │   ├── metrics.py                 # score oficial ICBHI + calibración de umbrales
    │   ├── train.py, run_kfold.py     # entrenamiento por fold + k-fold completo y reportes
    │   └── checkpoints/                # pesos entrenados (no versionado)
    └── gate_model/
        ├── dataset.py, model.py       # Dataset (log-Mel o MFCC) y arquitectura (PyTorch)
        ├── train.py, evaluate.py      # entrenamiento y evaluación sobre test
        ├── evaluate_external_hls.py   # evaluación sobre HLS-CMDS (ver más abajo)
        ├── run_pipeline.py            # corre train+evaluate+evaluate_external_hls seguidos
        ├── run_v5_mfcc.py             # gate v5 (MFCC, versión final), wrapper de run_pipeline.py
        ├── periodicity.py, dataset_periodicity.py,
        │   model_periodicity.py, run_v6_periodicity.py
        │                              # gate v6 (MFCC + periodicidad) — descartado, ver avances/8
        ├── plot_architecture.py       # diagrama de la arquitectura
        └── checkpoints/                # pesos entrenados (no versionado)
```

## Estado actual

Pensado para retomar el proyecto en una conversación nueva sin releer todo
`avances/` — un resumen de una línea por decisión, con el detalle completo
linkeado.

- **Dataset unificado (`unified_v2`)**: CirCor DigiScope (cardíaco) + ICBHI 2017
  (pulmonar), 4083 grabaciones. HLS-CMDS se descartó del dataset de
  entrenamiento (maniquí, muy pocos casos por clase) — ver
  `avances/3_preliminar_eda.md`.
- **Split train/val/test**: por `subject_id` (nunca se separa un mismo
  sujeto entre splits), estratificado por `pathology_label`. Las clases con
  menos de 3 sujetos van enteras a train (limitación documentada,
  `reports/splits/summary.csv` deja constancia de los números exactos).
- **Modelo gate (cardíaco vs. pulmonar) — versión vigente: v5 (MFCC).**
  Checkpoint en `gate_model/checkpoints_v5_mfcc/best.pt`. Historial completo,
  hipótesis descartadas y evidencia en
  `avances/4_gate_model_baseline_y_sesgo.md` y
  `avances/6_gate_v5_mfcc.md`; cada versión anterior queda congelada en
  `reports/gate/runs/`:
  - v1 (baseline): ~100% en test propio, pero solo 54.5% en HLS-CMDS
    (validación externa) — confirmado con un diagnóstico cuantitativo que
    el modelo aprendió a distinguir `source_db`, no el sonido en sí.
  - v2 (normalización z-score + augmentation de audio): F1 en HLS 0.81 —
    la mejora grande.
  - v3 (+ SpecAugment): empeoró (F1 en HLS 0.74) — descartado.
  - v4 (class weighting sobre v2, por el desbalance cardíaco:pulmonar
    ≈3.5:1): F1 en HLS 0.816, prácticamente igual a v2 — confirma que el
    desbalance no era la causa principal del problema de generalización.
  - **Causa más probable, aún sin resolver del todo**: un EDA comparativo
    (`preprocessing/compare_sources_eda.py`) encontró que la relación de
    brillo espectral cardíaco/pulmonar está *invertida* en HLS-CMDS
    respecto a CirCor/ICBHI — posiblemente una brecha de dominio real
    (maniquí vs. paciente) y no solo un artefacto de preprocesamiento.
  - **v5 (MFCC en vez de log-Mel, sobre v4) — versión final**: F1 en HLS
    0.847 (vs. 0.816 de v4), pero el recall de cardíaco casi no mejoró
    (74.0% vs. 76.0%) — la mejora es por mayor precisión, no por resolver la
    debilidad en cardíaco. Ver `avances/6_gate_v5_mfcc.md`.
  - v6 (rama adicional de periodicidad rítmica sobre v5): empeoró mucho
    (F1 en HLS 0.664, recall de cardíaco 49.7%) — descartado. Ver
    `avances/7_investigacion_debilidad_cardiaca.md` (motivación) y
    `avances/8_gate_v6_periodicity.md` (resultado negativo).
  - **Decisión**: se deja de iterar sobre el gate por ahora — v5 queda
    como versión final para esta etapa, con la debilidad en cardíaco sobre
    HLS-CMDS documentada como limitación conocida (brecha de dominio,
    ver `avances/7_...md`), y se avanza a la etapa 2.
- **Etapa 2 — cardíaco**: el intento de 3 clases se reemplazó por dos
  tareas binarias (murmullo, outcome); murmullo v1 es la versión final
  (F2≈0.71-0.74, validado por k-fold), outcome quedó como limitación
  conocida. Ver `avances/10` y `avances/11`.
- **Etapa 2 — pulmonar**: diagnosticar por paciente no es viable con ICBHI
  (5 de 6 clases con 6-16 pacientes, sesgo por edad), así que se plantea
  como **detección de sonidos adventicios por ciclo respiratorio**
  (crackle / wheeze, multi-etiqueta — tarea oficial del challenge ICBHI
  2017), con k-fold por paciente desde la primera corrida
  (`pulmonary_adventitious_model/`). Antes de esto se corrigió la fe de
  erratas del equipo de ICBHI (`avances/12`). Versión vigente: **v4**
  (pooling por máximo en el tiempo + pasa-altos 100 Hz + 40 épocas +
  checkpoint por AUC de val): score ICBHI **0.559±0.007**, AUC wheeze
  0.79 / crackle 0.68. Detecta peor en pacientes pediátricos (limitación
  conocida). Historial v1→v4 en `avances/13` y `avances/14`.
  Transfer learning, etapa 1: la CNN6 de PANNs **congelada** + un MLP chico
  iguala a v4 (0.559±0.024) sin entrenar la red — ver `avances/15`; pesos
  preentrenados en `data/pretrained/` (Zenodo doi:10.5281/zenodo.3987831).

## Entorno

Python 3.12.10 (fijado en `.python-version`, gestionado con `pyenv`).

```bash
python -m venv .venv
./.venv/Scripts/pip install -r requirements.txt
```

En VS Code, seleccionar `desarrollo/.venv/Scripts/python.exe` como intérprete
del proyecto y de los notebooks.

## Datos

**La carpeta `data/` no se versiona** (los datasets pesan varios GB en
total). Hay que descargar los tres datasets crudos a mano y ubicarlos así,
dentro de `desarrollo/data/raw/` (rutas exactas en
`src/unified_dataset/config.py`):

```
data/raw/
├── HLS-CMDS/
│   └── Dataset/
│       ├── HS/          # HS.csv + WAVs de sonidos cardíacos
│       ├── LS/          # LS.csv + WAVs de sonidos pulmonares
│       └── Mix/         # (no se usa en el pipeline actual)
├── CirCor/
│   ├── training_data/    # WAVs + .hea + .tsv por grabación
│   └── training_data.csv
└── ICBHI/
    ├── audio_and_txt_files/  # WAVs + .txt de anotación por grabación
    ├── demographic_info.csv
    └── patient_diagnosis.csv
```

Fuentes de descarga:

| Dataset | Link | Referencia |
|---|---|---|
| HLS-CMDS v2 | https://github.com/torabiy/hls-cmds | Torabi, Shirani, Reilly. *IEEE Data Descriptions*, doi: 10.1109/IEEEDATA.2025.3566012 |
| CirCor DigiScope v1.0.3 | https://physionet.org/content/circor-heart-sound/1.0.3/ | Oliveira et al. *PhysioNet*, doi: 10.13026/tshs-mw03 |
| ICBHI 2017 Respiratory Sound DB | https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database | Rocha BM et al. (2019). *Physiological Measurement* 40 035001 |

El resto de `data/` (`unified_v2/`, `gate/`) y todo `reports/` se generan
corriendo los scripts de la sección siguiente — no hace falta (ni conviene)
versionarlos.

## Cómo correr el pipeline (modelo gate)

Desde `desarrollo/src/`, en este orden:

```bash
# 1. Construir el dataset unificado (CirCor + ICBHI) a partir de los datos crudos
python unified_dataset/parsers/build_dataset.py

# 2. Split train/val/test por sujeto
python splits/make_splits.py

# 3. Ventanas de audio preprocesadas para el gate
python preprocessing/make_gate_windows.py

# 4. Gráficos de señal cruda vs. preprocesada (opcional, para el informe)
python preprocessing/plot_before_after.py

# 5. Entrenar el modelo gate
python gate_model/train.py

# 6. Evaluar sobre el split de test
python gate_model/evaluate.py

# 7. Validación externa con HLS-CMDS (ver "Notas" abajo)
python preprocessing/make_hls_windows.py
python gate_model/evaluate_external_hls.py

# 8. Diagrama de la arquitectura (para el informe)
python gate_model/plot_architecture.py
```

Cada script imprime un resumen por consola y además deja evidencia
guardada en `reports/<etapa>/` (CSVs y gráficos) — pensado para poder
citarlos directamente en el informe final sin tener que volver a correr
nada.

Los pasos 5-7 (entrenar + evaluar + validar en HLS-CMDS) se pueden lanzar
en una sola corrida desatendida con `python gate_model/run_pipeline.py`
(útil para dejar la PC corriendo sin tener que estar pendiente de cuándo
termina el entrenamiento). Para correr el experimento del gate v5 (MFCC en
vez de log-Mel) en vez del gate vigente: `python gate_model/run_v5_mfcc.py`
— guarda checkpoint y reportes en carpetas separadas, no pisa los de v4.

## Cómo correr el modelo pulmonar (sonidos adventicios)

Desde `desarrollo/src/`, después de construir el dataset unificado (paso 1
de arriba):

```bash
# 1. Un WAV por ciclo respiratorio anotado (data/pulmonary/cycles/)
python preprocessing/make_pulmonary_cycles.py

# 2. 5 folds por paciente, estratificados por diagnóstico
python splits/make_pulmonary_kfold.py

# 3. Entrenar + evaluar los 5 folds (reports/pulmonary_adventitious/<run>/)
python pulmonary_adventitious_model/run_kfold.py
```

`run_kfold.py` acepta `--folds 0 --epochs 2` para una prueba rápida y
`--feature mfcc --run-name <nombre>` para comparar variantes sin pisar
corridas anteriores. Además de las métricas por fold, guarda las
predicciones out-of-fold de todos los ciclos y el resultado desglosado por
equipo y por grupo de edad (chequeo de atajos).

## Notas y limitaciones conocidas

- **`sound_type` correlacionado con `source_db`**: en el dataset de
  entrenamiento, 100% de los audios cardíacos vienen de CirCor y 100% de
  los pulmonares de ICBHI. Existe el riesgo de que el modelo gate aprenda a
  distinguir la firma del equipo/entorno de grabación en vez de la
  diferencia acústica real. Por eso se agregó `evaluate_external_hls.py`:
  HLS-CMDS es el único dataset con ambos tipos de sonido grabados con el
  mismo equipo, y sirve como chequeo de que el modelo generalizó lo
  correcto (ver docstring de `preprocessing/make_hls_windows.py` para el
  detalle completo).
- **`sample_rate_hz` del CSV maestro**: en ICBHI se lee del header del WAV
  (antes era una inferencia por equipo, con errores); de todos modos el
  pipeline de audio siempre lee el sample rate real del WAV, nunca esa
  columna (ver `avances/3_preliminar_eda.md`).
- **Fe de erratas de ICBHI**: 91 grabaciones vienen con `Meditron` en el
  nombre cuando en realidad son AKGC417L/LittC2SE
  (`data/raw/ICBHI/filename_differences.txt`). `parse_icbhi.py` corrige la
  columna `equipment` sin renombrar archivos (ver
  `avances/12_icbhi_fe_de_erratas_equipo.md`).
- **Clases con muy pocos sujetos** (`pneumonia`, `bronchiectasis`,
  `bronchiolitis_asthma` en ICBHI): van enteras a train en el split actual,
  no hay suficientes sujetos para repartir de forma significativa.

## Próximos pasos

Etapa 2: modelos de patología cardíaca y pulmonar, usando segmentación por
ciclo (latido S1→S1 en CirCor vía los `.tsv`, ciclo respiratorio directo en
ICBHI vía los `.txt`) en vez de ventaneo de duración fija — ver `avances/`
para el detalle de esta decisión y sus alternativas consideradas.

**Pendiente de decidir antes de entrenar** — revisar `pathology_label`: se
diseñó para el split conjunto del gate (cardíaco+pulmonar a la vez), y
colapsa algo de granularidad que ya no hace falta mantener ahora que cada
modelo de patología va a tener su propio split independiente (ej.: el
detalle de murmullo de CirCor no se usa hoy; ICBHI agrupa
URTI+LRTI y Bronchiolitis+Asthma). Ver `avances/9_revision_clases_etapa2.md`
para el análisis completo y la recomendación por dominio.
