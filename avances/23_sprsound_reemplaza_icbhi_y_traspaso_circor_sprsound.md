# SPRSound: exploración, entrenamiento y decisión de reemplazar a ICBHI; traspaso para el modelo CirCor + SPRSound

Continúa `avances/22` (§8: sumar una base pulmonar nueva). Se eligió
**SPRSound** (pediátrica, eventos anotados con tiempos, CC BY 4.0) y se
siguió el plan de cinco pasos acordado con el usuario: (1) explorar los
datos, (2) verificar el preprocesamiento, (3) entrenar un modelo solo
pulmonar con SPRSound, (4) integrarla con ICBHI y (5) sacar conclusiones.
Al final (§8) está el traspaso para la próxima etapa: **el modelo de 6
clases con CirCor (cardíaco) + SPRSound (pulmonar)**. Fecha: 2026-10-29.

Resumen en cuatro líneas:

- **SPRSound tiene señal aprendible**: la rama pulmonar chica entrenada solo
  con ella separa grabaciones normales de anormales con AUC 0.767 (ICBHI →
  ICBHI: 0.568), con crackle/wheeze por ventana 0.74/0.87.
- **No hay transferencia entre bases en ninguna dirección**: SPRSound →
  ICBHI 0.539 y ICBHI → SPRSound 0.524. Crackle no transfiere nada.
- **Entrenar con las dos juntas no ayuda a ICBHI** (score 0.490, igual que
  solo ICBHI) **y mete un atajo**: el modelo aprende "suena a ICBHI ⇒
  anormal" (ICBHI 69% anormal, SPRSound 29%).
- **Decisión: SPRSound reemplaza a ICBHI como base pulmonar de
  entrenamiento; ICBHI pasa a validación externa** (como HLS). Las dos
  bases de entrenamiento, CirCor y SPRSound, son pediátricas.

## 1. Paso 1: exploración

Script: `src/eda/explore_sprsound.py` → `reports/sprsound/` (`recordings.csv`,
`events.csv`, `patients.csv`, `alt_annotations_vs_biocas.csv`,
`explore_run.txt`). Crudos en `data/raw/SPRSound/` (1.3 GB; se borraron los
WAV duplicados de Classification/Compression/Detection, verificados por CRC).

- **6567 grabaciones, todas a 8 kHz mono**, de 9 a 15 s, en 6 entregas
  (BioCAS 2022 train/inter/intra, 2023, 2024, 2025). Focos p1-p4 (2025
  agrega p5-p8).
- **952 pacientes, edad mediana 4.9 años** (0-16.8). Nombre del archivo:
  `paciente_edad_sexo_foco_n`.
  - 390 grabaciones **no traen número de paciente**: se excluyen (no se
    pueden asignar a un fold sin riesgo de fuga).
  - 162 pacientes aparecen en train2022 e intra2022: los folds se arman
    por paciente sobre todas las entregas juntas, no con los splits del
    challenge.
- **Eventos con inicio y fin en ms**: Normal 18.772, Fine Crackle 3530,
  Wheeze 1505, Wheeze+Crackle 303, Rhonchi 217, Coarse Crackle 177, Stridor
  74.
- **Etiqueta de grabación** (Normal, CAS, DAS, CAS & DAS, Poor Quality):
  2024 y 2025 no la traen, pero coincide al 100% con la que implican los
  eventos en los años que sí la tienen, así que se usa esa.
- Los JSON de Classification/ y Detection/ tienen los mismos eventos que
  BioCAS20xx: la única fuente es BioCAS20xx.
- Diagnósticos (`Patient Summary`): casi todo neumonía no severa; 204
  pacientes con audio no tienen diagnóstico. No se usan para estratificar.

## 2. Paso 2: verificación del preprocesamiento

Script: `src/eda/sprsound_preprocessing_check.py` →
`reports/sprsound/preprocessing/` (espectros, energía por banda, recortes,
huecos, simulación de ventanas, `check_run.txt`).

| Pregunta | Resultado | Decisión |
|---|---|---|
| ¿El pasa-bajos de 2048 Hz borra información? | La energía sobre 2048 Hz es ~0 en todas las clases (el audio nativo ya viene filtrado) | Preprocesamiento sin cambios (pasa-bajos 2048 Hz, 4 kHz, RMS 0.1) |
| ¿Recortes tras normalizar el RMS? | 97.5% de las grabaciones tienen alguno (ICBHI 10.4%), pero es ~0.13% de las muestras; vienen de golpes de manipulación (pico/RMS ~8-9 contra ~4 en ICBHI) | Sin cambios; una normalización robusta queda como posible mejora |
| ¿Los huecos sin anotar son normales? | No: en las grabaciones anormales los huecos tienen tantos transitorios como los eventos crackle (curtosis 12-13 contra 9) | **Ventanas normales solo de grabaciones Normal** |
| ¿Rhonchi y Stridor? | Rhonchi es continuo, como el wheeze de ICBHI; Stridor es raro (74 eventos, 10 pacientes) | Rhonchi → wheeze; las ventanas que tocan Stridor se descartan |
| ¿Umbral de superposición? | Entre 0.5 y 1 s cambian 173 ventanas | Se mantiene 1 s (o ≥ 50% del evento), igual que ICBHI |
| ¿Diferencia de dominio con ICBHI? | La normal de SPRSound tiene +7.8 dB en 150-500 Hz y −7.9 dB en 1-2 kHz respecto de la de ICBHI | Riesgo de atajo → test cruzado y métricas por base |

## 3. Datos resultantes

Script: `src/preprocessing/make_sprsound_windows.py` →
`data/sprsound/recordings.csv` y `windows_index_4s.csv`; reutiliza
`label_pulmonary` de `make_unified_windows.py` (cada evento hace de "ciclo").

Se descarta, en orden: Poor Quality (230 grabaciones), sin número de
paciente (387), ventanas más cortas que 4 s (2), ventanas sin ningún evento
(3212), ventanas que tocan Stridor (92), ventanas normales de grabaciones
anormales (1130).

| Clase | Ventanas | Pacientes | ICBHI (avances/22) |
|---|---|---|---|
| pulm_normal | 19.867 | 747 | 4.086 |
| pulm_crackle | 3.931 | 240 | 2.713 |
| pulm_wheeze | 1.639 | 165 | 1.204 |
| pulm_both | 631 | 66 | 972 |
| **Total** | **26.068** (5920 grabaciones, 854 pacientes) | | 8.975 (920 grabaciones, 126 pacientes) |

- **Folds** (`src/splits/make_sprsound_kfold.py` → `data/sprsound/kfold/`):
  5 folds por paciente, estratificados por la clase más rara de cada
  paciente (both > wheeze > crackle > normal), con el mismo esquema que
  `make_pulmonary_kfold.py`.
- **Caché de audio**: `make_unified_audio_cache.py --data sprsound` →
  `data/sprsound/audio_4k_int16.npy` (18.9 h).

## 4. Paso 3: rama pulmonar solo con SPRSound

`run_kfold.py --data sprsound`, con la receta de `abl_pulm` (base, log-Mel,
pooling promedio + máximo; 24k parámetros). Desbalance: pesos por clase
recalculados en cada fold sobre el train. **inv** es la receta base y **sqrt**
la única variante, registrada antes de entrenar; se elige por validación.

- **Gana inv** (promedio de la selección en val: 0.816 contra 0.800). En
  test empatan, y en el fold 0 sqrt eligió la época 1 (menos estable).
- Cada fold también predice el **test de ICBHI del mismo fold** (test
  cruzado): el modelo nunca vio ICBHI y sus umbrales se calibran con el val
  de SPRSound.

### 4.1 AUC por grabación con todas las ventanas

En la primera evaluación (`evaluate_sprsound.py`), el puntaje de cada
grabación (media de 1 − P(normal)) promediaba solo las ventanas del índice.
En SPRSound eso deja afuera las ventanas normales de las grabaciones
anormales, lo que sesga el resultado a favor del modelo. Por eso se rehízo
con **todas las ventanas de cada grabación**, como en uso real
(`evaluate_full_recordings.py`; las grabaciones de menos de 4 s se rellenan
con ceros). El sesgo resultó chico (menos de 0.01), porque los huecos de las
grabaciones anormales también tienen anomalías y el modelo las ve.

## 5. Paso 4: entrenamiento conjunto SPRSound + ICBHI

`run_kfold.py --data joint`:

- Train y val son la unión de los de cada base. Cada base usa sus folds; el
  **test de ICBHI es el mismo de siempre**.
- Pesos inv sobre el train conjunto.
- Checkpoint elegido por el **promedio de la selección en val de cada
  base**, para que no domine SPRSound, que tiene ~3 veces más ventanas.

### 5.1 Resultados (5 folds, media entre folds)

| Modelo | Crackle ICBHI (ventana) | Wheeze ICBHI (ventana) | Score ICBHI (Se / Sp) | AUC grab. ICBHI | AUC grab. SPRSound | AUC grab. conjunto |
|---|---|---|---|---|---|---|
| `abl_pulm` (solo ICBHI) | 0.606 | 0.747 | 0.490 (0.13 / 0.86) | 0.568 | 0.524 | 0.502 |
| `spr_inv` (solo SPRSound) | 0.546 | 0.700 | 0.406 (0.10 / 0.72) | 0.539 | **0.767** | 0.633 |
| `spr_sqrt` | 0.548 | 0.718 | 0.476 (0.06 / 0.89) | 0.550 | 0.774 | 0.653 |
| `joint_inv` (las dos) | 0.619 | 0.737 | 0.490 (0.14 / 0.84) | 0.557 | 0.760 | 0.782 |

AUC por grabación = normal contra anormal, con todas las ventanas.
"Conjunto" = los tests de las dos bases tratados como una sola base. En
SPRSound, `spr_inv` también da crackle/wheeze por ventana 0.741/0.869 y
score 0.603 (Se 0.33 / Sp 0.88).

Fuentes: `reports/sprsound/eval/summary.csv` (por ventana, score, por
año), `reports/sprsound/full_recording/summary.csv` (por grabación) y
`reports/sprsound/figures/` (curvas de aprendizaje, ROC, confusión, por
año).

- **Sumar SPRSound no mejora ICBHI**: las diferencias con `abl_pulm` (+0.013
  crackle, −0.010 wheeze, mismo score) caen dentro del ruido entre folds
  (±0.03-0.05).
- **Sumar ICBHI casi no le cuesta a SPRSound** (0.760 contra 0.767).
- **El 0.782 conjunto es un atajo**: es más alto que cada base por separado.
  Puntaje medio de anormalidad:

  | | Grabación normal | Grabación anormal |
  |---|---|---|
  | ICBHI | 0.80 | 0.85 |
  | SPRSound | 0.52 | 0.63 |

  - Una grabación normal de ICBHI puntúa más alto que una anormal de
    SPRSound.
  - El puntaje separa las dos bases con AUC 0.90: es el atajo de equipo
    que `avances/22` §8.3 pedía vigilar.
  - En `spr_inv`, que no vio ICBHI, no aparece.
- **Matriz de confusión del test cruzado**: el modelo de SPRSound casi nunca
  predice crackle en ICBHI (2%); manda el 79-94% de los crackles a normal.
- **Por año de SPRSound** (`spr_inv`): AUC entre 0.62 (2022 intra) y 0.77
  (2024). No hay ningún año sospechosamente fácil; en 2024 solo el 7% de
  las grabaciones es anormal.

## 6. Paso 5: conclusiones y decisión

1. **SPRSound es mejor base pulmonar de entrenamiento que ICBHI para esta
   tesis**:
   - **Más datos**: ~6.8 veces más pacientes (854 contra 126) y 1.4 veces
     más ventanas de crackle.
   - **Mejores anotaciones**: eventos con tiempos e ID de paciente.
   - **Misma población que CirCor (pediátrica)**: alcance coherente para
     el sistema completo.
   - **Más aprendible**: 0.77 contra 0.57.
   - Los números dentro de cada base no se pueden comparar directamente (son
     tareas distintas), así que el argumento principal es la población y el
     volumen, no la cifra.
2. **ICBHI no aporta como datos de entrenamiento** (no mejora su propio test)
   y **mete un atajo de base de origen**.
3. **Decisión (con el usuario)**: **SPRSound reemplaza a ICBHI** como base
   pulmonar de entrenamiento, e **ICBHI pasa a validación externa** (como
   HLS).
   - El resultado esperable en ICBHI externo es bajo (~0.54 por grabación;
     crackle no transfiere).
   - Se reporta como "generalización a adultos con otro equipo", una
     limitación honesta del alcance pediátrico.

## 7. Implementación (`src/`)

- `common/paths.py`: rutas `SPRSOUND_*` y `sprsound_windows_index()`.
- `eda/explore_sprsound.py`, `eda/sprsound_preprocessing_check.py`: pasos 1 y 2.
- `preprocessing/make_sprsound_windows.py`, `splits/make_sprsound_kfold.py`,
  `preprocessing/make_unified_audio_cache.py --data sprsound`: datos.
- `unified_model/dataset.py`: `open_audio_cache(cache_npy, offsets_csv)`.
- `unified_model/run_kfold.py`: `--data unified|sprsound|joint`.
  - `load_sources()` define las fuentes de ventanas (índice, caché, carpeta
    de folds, si entrena).
  - `concat_dataset()` junta las fuentes.
  - `val_metrics()` promedia la selección por fuente.
  - Los roles llevan un sufijo por fuente (`val`/`test` para la principal,
    `val_icbhi`/`test_icbhi` para ICBHI).
- `unified_model/evaluate_sprsound.py`: métricas por ventana, score ICBHI y
  AUC por grabación (con las ventanas del índice) por base, conjunto y por
  año.
- `unified_model/evaluate_full_recordings.py`: AUC por grabación con todas
  las ventanas. Guarda las probabilidades en
  `reports/sprsound/full_recording/<run>_<test>_windows.csv` y no las
  recalcula si ya están.
- `unified_model/plot_sprsound.py`: figuras.

Comandos (desde `desarrollo/src`):

    python eda/explore_sprsound.py
    python eda/sprsound_preprocessing_check.py
    python preprocessing/make_sprsound_windows.py
    python splits/make_sprsound_kfold.py
    python preprocessing/make_unified_audio_cache.py --data sprsound
    python unified_model/run_kfold.py --data sprsound --width base --feature logmel --pooling avg_max --class-weights inv --run-name spr_inv
    python unified_model/run_kfold.py --data sprsound --width base --feature logmel --pooling avg_max --class-weights sqrt --run-name spr_sqrt
    python unified_model/run_kfold.py --data joint --width base --feature logmel --pooling avg_max --class-weights inv --run-name joint_inv
    python unified_model/evaluate_sprsound.py --runs spr_inv spr_sqrt joint_inv
    python unified_model/evaluate_full_recordings.py --runs spr_inv spr_sqrt joint_inv
    python unified_model/plot_sprsound.py

Cada corrida tarda ~1.5-2 h en CPU. Los checkpoints quedan en
`src/unified_model/checkpoints_<run>/` (no versionados).

## 8. Traspaso: modelo de 6 clases con CirCor + SPRSound

Objetivo: rehacer el modelo único sin gate (objetivo primario de la tesis)
con **CirCor para cardíaco y SPRSound para pulmonar**, igual que
`avances/22` lo hizo con CirCor + ICBHI, y volver a comparar con el
pipeline separado.

### 8.1 Datos y código

- **Opción nueva `--data circor_spr` en `run_kfold.py`**: con
  `load_sources()` alcanza con agregar un caso con dos fuentes que
  entrenan.
  - Ventanas cardíacas del índice unificado (`domain == "cardiac"`, folds
    unificados, caché `data/unified_model/`).
  - Ventanas de SPRSound (sus folds, caché `data/sprsound/`).
  - El test cardíaco queda idéntico al de `avances/22`.
  - Hay que decidir cómo elegir el checkpoint: el promedio de los AUC de
    murmullo, crackle y wheeze, como en `avances/22`, sale directo de
    `window_metrics` si las dos fuentes se evalúan juntas.
- **Balance entre dominios mucho mejor que con ICBHI**: CirCor 34.523
  ventanas (29.111 normales, 5.412 con murmullo, 942 pacientes) contra
  SPRSound 26.068. Con ICBHI eran 34.5k contra 9k.
- **Separación de dominios**: CirCor y SPRSound también se distinguen por
  equipo, pero eso no es un problema, porque separar cardíaco de pulmonar
  es lo que queremos. El atajo del §5 aparecía porque las dos bases eran del
  **mismo** dominio y tenían distinta proporción de anormales; acá cada
  dominio tiene una sola base.

### 8.2 Entrenamiento y comparación

1. **Modelo de 6 clases** con la receta de `avances/22` (base, log-Mel,
   pooling promedio + máximo, pesos inv). Opcional: la variante ancha
   (95k), como en `avances/22` §4.2.
2. **Ablaciones de referencia (ya entrenadas)**: `abl_card` (solo CirCor) y
   `spr_inv` (solo SPRSound). Miden cuánto le cuesta a cada dominio
   compartir la red.
3. **Pipeline separado con SPRSound**: gate → rama cardíaca (`abl_card`) +
   rama pulmonar (`spr_inv`).
   - **El gate v5 se entrenó con ICBHI** como clase pulmonar: hay que
     reentrenarlo con CirCor + SPRSound, o usar el dominio que da la
     softmax del integrado, decidiéndolo con el usuario.
   - Con eso se rehace la comparación de metodologías de `avances/22` §4.2
     (accuracy de 6 clases en capa 1).
   - Ojo: el test de capa 1 dependía de los pacientes de test del gate;
     hay que redefinirlo con SPRSound.

### 8.3 Validación externa

- **ICBHI completo** (las 920 grabaciones pulmonares, ninguna vista al
  entrenar).
  - `evaluate_full_recordings.py` hoy infiere solo los pacientes de test de
    cada fold. Para validación externa hay que pasar todas las grabaciones
    por los 5 modelos y promediar.
  - Mismas métricas: AUC por grabación, score ICBHI con umbrales calibrados
    en el val de SPRSound, y por equipo.
- **HLS** (cardíaco y pulmonar) con `integration/external_hls.py`, como en
  `avances/20`. Mide el atajo de dominio fuera de las bases de
  entrenamiento.

### 8.4 Documentación

- **Memoria técnica, capítulo 1**: el estado del arte, los objetivos y el
  alcance (sección 1.4) se escribieron con ICBHI como base pulmonar. Hay
  que pasar a alcance pediátrico, con SPRSound como base de entrenamiento e
  ICBHI como validación externa. Se trabaja sección por sección, con el
  usuario.

## 9. Pendiente (en orden)

1. Modelo de 6 clases CirCor + SPRSound (§8.1-8.2).
2. Pipeline separado con SPRSound y gate nuevo; comparación de metodologías.
3. Validación externa: ICBHI completo y HLS (§8.3).
4. Adoptar la rama cardíaca nueva (`abl_card`) en el pipeline (pendiente de
   `avances/22` §9).
5. Actualizar la memoria técnica (§8.4).
6. Extras (no antes de lo anterior): ramas chicas, CNN6 integrada o
   fine-tuning con SPRSound, reducir parámetros, batches balanceados,
   normalización robusta a los golpes de manipulación (§2).

## 10. Estado del repositorio

Commiteado: `cb6bd57` (avances 22) y `aaa337f` (SPRSound: código y
`reports/sprsound/`). Sin commitear al cerrar:

- Este documento.
- `desarrollo/reports/sprsound_explore_run.txt`: log de una corrida fallida,
  sin contenido útil. Se puede borrar.

`data/` no se versiona: los crudos están en `data/raw/SPRSound/` y lo
derivado en `data/sprsound/` (índices, folds, caché de 18.9 h).
