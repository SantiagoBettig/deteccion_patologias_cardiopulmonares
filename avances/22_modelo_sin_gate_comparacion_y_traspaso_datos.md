# Modelo único sin gate: comparación con el pipeline separado y traspaso para ampliar los datos pulmonares

Continúa `avances/21` (§3-§4). Cierra la **comparación base entre las dos
metodologías**: un modelo único sin gate (integrado) contra el pipeline
gate → ramas (separado), que es un objetivo primario de la tesis. Al final
(§8) está el traspaso para la próxima etapa: **sumar una base de datos
pulmonar nueva** desde otra conversación. Fecha: 2026-10-28.

Resumen en cuatro líneas:

- **A igual red y a igual presupuesto de parámetros, el separado le gana al
  integrado**: accuracy del sistema en 6 clases 0.812 (tres redes chicas,
  ~72k parámetros) contra 0.758 (una red de 24k) y 0.792 (una de 95k).
- **Compartir la red le cuesta a las dos tareas**: la misma CNN entrenada
  solo con un dominio rinde más que dentro del integrado.
- **La rama cardíaca nueva es el mejor detector de murmullo hasta ahora**
  (F2 0.794, AUC 0.927), y con ella el pipeline separado llega a 0.826 en
  6 clases.
- **El pulmonar no lo arregla ni la calibración, ni el ancho, ni separar
  la red**: las redes chicas casi no distinguen una grabación normal de una
  anormal (AUC 0.54-0.57; CNN6 preentrenada 0.69). El límite está en los
  datos → próxima etapa: más datos pulmonares.

## 1. Diseño del modelo único (decidido con el usuario)

| Decisión | Elegido | Por qué |
|---|---|---|
| Salida | 6 clases softmax: card_normal, card_murmur, pulm_normal, pulm_crackle, pulm_wheeze, pulm_both | Comparable con la matriz de 6 clases de la capa 1; sin un gate escondido |
| Unidad | Ventanas fijas de 4 s, paso 2 s (las de `window_inference.py`) | Lo que ve el sistema al predecir |
| Preprocesamiento | `audio_ops.preprocess` a 4 kHz **sin pasa-altos** | El de 100 Hz borra S1/S2 |
| Etiqueta cardíaca | La de la grabación | No hay ciclos anotados |
| Etiqueta pulmonar | Superposición con los ciclos anotados de ICBHI: hay crackle (wheeze) si la ventana se superpone ≥ 1 s **o** ≥ 50% de un ciclo con crackle (wheeze). Las ventanas que no tocan ningún ciclo se descartan (54) | Etiqueta más fina que la de la grabación |
| Folds | Unión del fold *i* de murmullo y del fold *i* pulmonar | Cada paciente cae en test en el mismo fold que en las ramas: comparación paciente a paciente |
| Desbalance | Pesos por clase inversos a la frecuencia (normalizados a peso medio 1) | Primera iteración; batches balanceados quedan como extra |
| Arquitectura | CNN chica de 3 bloques, la de gate/murmullo, en CPU | CNN6 integrada queda para después |

Dataset resultante: **43.498 ventanas** de 4083 grabaciones (1068
pacientes): card_normal 29.111, card_murmur 5.412, pulm_normal 4.086,
crackle 2.713, wheeze 1.204, both 972. Entre exigir 0.5 s o 1 s de
superposición cambian solo 236 ventanas.

Implementación (`src/`):

- `splits/make_unified_kfold.py` → `data/unified_v2/unified_kfold/fold{i}.csv`.
- `preprocessing/make_unified_windows.py` → `data/unified_model/windows_index_4s.csv`.
- `preprocessing/make_unified_audio_cache.py`: las 4083 grabaciones
  preprocesadas en un solo `int16` memmap
  (`data/unified_model/audio_4k_int16.npy`, ~740 MB, 25.6 h de audio).
- `unified_model/dataset.py`: el frontend espectral corre en torch, dentro
  del modelo, y replica librosa (MFCC: diferencia máxima 7e-6; log-Mel:
  2e-6). Con librosa por ventana, el entrenamiento llevaba ~17 h.
- `unified_model/model.py` (`UnifiedCNN`), `run_kfold.py` (entrenamiento),
  `evaluate_unified.py` (por grabación, capas 1 y 2) y
  `evaluate_ablation.py` (§5).

Selección del checkpoint: promedio de tres AUC por ventana en val (murmullo
en ventanas cardíacas; crackle y wheeze en pulmonares). Adam 1e-3, batch
64, hasta 30 épocas, early stopping con paciencia 8. La evaluación por
grabación reutiliza `integration/evaluate_integration.py` (reglas
mean/p75/max/frac, calibración `pooled_fixed`), así los números son
directamente comparables con `avances/20`-`21`.

Probabilidades por grabación a partir de la softmax: P(murmullo) =
card_murmur; P(crackle) = crackle + both; P(wheeze) = wheeze + both;
dominio = P(cardíaco) media > 0.5.

## 2. Pilotos (fold 0): la receta pulmonar también vale acá

AUC por ventana, test del fold 0:

| Variante | Parámetros | Murmullo | Crackle | Wheeze | Dominio |
|---|---|---|---|---|---|
| MFCC + pooling promedio, base | 23.9k | 0.867 | **0.494** | 0.625 | 0.944 |
| MFCC + pooling promedio, ancha | 93.9k | 0.828 | 0.558 | 0.538 | 0.872 |
| **log-Mel + promedio y máximo, base** | 24.3k | 0.871 | 0.619 | 0.712 | 0.997 |
| log-Mel + promedio y máximo, ancha | 94.7k | 0.881 | 0.641 | 0.770 | 0.997 |

- El primer piloto repitió los errores de `avances/13`-`14`: MFCC y pooling
  promedio diluyen los eventos cortos, y crackle queda en el azar.
- Con log-Mel y el máximo temporal (el `time_max` de la v2 pulmonar) se
  arregla.
- Ancha y base empatan en validación (0.738 contra 0.737), así que se
  eligió la **base por parsimonia** y para que la comparación con las ramas
  chicas sea limpia. A la ancha se volvió en §4.2 para igualar el
  presupuesto.

## 3. Ajuste del integrado (folds 0 y 3, elegido solo por validación)

Antes de comparar se le dio al integrado su propia ronda de ajuste: las
ramas tuvieron muchas versiones y el integrado tenía que tener la misma
oportunidad. Hubo tres etapas, una por vez, sobre la v1_base.

| Run | Cambio | Val (promedio de 3 AUC) | Murmullo | Crackle | Wheeze |
|---|---|---|---|---|---|
| v1_base | — | 0.737 | 0.900 | 0.632 | 0.678 |
| t1_res | n_fft 256, hop 80, último bloque sin reducir tiempo | 0.740 | 0.868 | **0.662** | 0.690 |
| t1b_hop80 | hop 80 (n_fft 512), último bloque sin reducir tiempo | 0.742 | 0.890 | 0.635 | 0.700 |
| t2_sqrt | Pesos por clase en raíz cuadrada | 0.741 | **0.913** | 0.640 | 0.669 |
| t2_bal | Muestreo con la mitad de cada dominio | 0.731 | 0.899 | 0.636 | 0.658 |
| t3_sa1 | SpecAugment, 1 máscara | 0.724 | 0.908 | 0.611 | 0.654 |
| t3_sa2 | SpecAugment, 2 máscaras | 0.729 | 0.902 | 0.640 | 0.645 |

Fuente: `reports/unified_model/tuning_folds03_summary.csv`.

- **Ninguna variante le gana de forma consistente a la v1_base**
  (diferencias de −0.013 a +0.005, dentro del ruido entre folds). El
  ajuste se cerró con la v1_base.
- **Conflicto de resolución**: crackle mejora con ventana de FFT corta
  (128 → 64 ms) y murmullo empeora en la misma medida. Con una sola
  representación no se satisfacen las dos tareas. **Es un argumento a
  favor del modelo separado**: cada rama puede tener su resolución. Idea
  derivada: entrada con dos resoluciones en el integrado.
- El desbalance entre dominios no es la causa (t2_bal empeora), y más
  regularización tampoco ayuda (SpecAugment empeora).

## 4. Resultados con 5 folds (por grabación)

### 4.1 Capa 2: cada tarea sobre todos los pacientes (`pooled_fixed`, regla elegida en val)

| Modelo | Parámetros | F2 murmullo | AUC murmullo | Score ICBHI | Se / Sp | AUC crackle / wheeze* |
|---|---|---|---|---|---|---|
| Integrado base (v1_base) | 24k | 0.755 ± 0.033 | 0.906 | 0.492 ± 0.024 | 0.04 / 0.94 | 0.563 / 0.694 |
| Integrado ancho (v1_wide) | 95k | 0.765 ± 0.044 | 0.916 | 0.481 ± 0.025 | 0.09 / 0.88 | 0.559 / 0.740 |
| Solo cardíaca / solo pulmonar (§5) | 24k c/u | **0.794 ± 0.034** | **0.927** | 0.490 ± 0.024 | 0.13 / 0.86 | 0.610 / 0.735 |
| Pipeline actual: MurmurCNN / CNN6 | 24k / 4.6M | 0.725 ± 0.013 | 0.898 | **0.526 ± 0.014** | 0.30 / 0.76 | **0.661 / 0.821** |

\* Con la regla de agregación promedio para los modelos chicos; la regla
`max` para CNN6.

### 4.2 Capa 1: sistema completo, 161 pacientes de test del gate (591 grabaciones)

| Sistema | Parámetros totales | Accuracy de dominio | **Accuracy 6 clases** | F2 murmullo | Score pulmonar |
|---|---|---|---|---|---|
| Integrado base | 24k | **0.998** | 0.758 | 0.687 | 0.449 |
| Integrado ancho | 95k | **0.998** | 0.792 | 0.742 | 0.495 |
| Gate v5 + ramas chicas de la ablación | ~72k | 0.986 | 0.812 | 0.722 | 0.589 |
| Gate v5 + MurmurCNN + CNN6 (pipeline de `avances/21`) | — | 0.986 | 0.812 | 0.660 | 0.606 |
| **Gate v5 + rama cardíaca nueva + CNN6** | — | 0.986 | **0.826** | 0.722 | 0.606 |

Fuentes: `reports/unified_model/runs/{v1_base,v1_wide}/eval/`,
`reports/unified_model/ablation/layer1_comparison.csv`.

- **El integrado discrimina el dominio mejor que el gate** (0.998 contra
  0.986: 1 error contra 8). Aprender las 6 clases juntas da un "gate"
  implícito mejor que el gate entrenado solo para eso. Queda por ver en HLS
  si es por el atajo de equipo (`avances/4`).
- La parte pulmonar de la capa 1 son solo 116 grabaciones de 19 pacientes:
  las diferencias de score pulmonar en esta tabla son ruidosas. La capa 2
  es la referencia pulmonar.

## 5. Ablación por dominio: ¿cuánto cuesta compartir la red?

**Pregunta**: el integrado rinde menos en pulmonar que el pipeline, pero la
comparación mezcla dos cosas: integrado contra separado, y CNN chica contra
CNN6. La ablación saca un solo componente, el entrenamiento conjunto. Es la
misma red con la misma receta y los mismos folds, entrenada solo con las
ventanas de un dominio (`run_kfold.py --domain cardiac|pulmonary`; la
softmax queda restringida a las clases de ese dominio).

Resultado (capa 2, tabla de §4.1):

- **Murmullo**: solo cardíaca F2 0.794 / AUC 0.927, contra 0.755 / 0.906
  del integrado. Compartir la red le cuesta ~0.04 de F2, sobre todo en
  precisión (0.675 contra 0.578). Contra MurmurCNN (0.725) la ganancia
  viene de la receta nueva: log-Mel, pooling promedio + máximo, 4 s sin
  pasa-altos.
- **Pulmonar**: solo pulmonar AUC crackle/wheeze 0.610/0.735, contra
  0.563/0.694 (+0.045). La red sola ordena mejor las grabaciones, pero el
  score con umbral no cambia (0.490 contra 0.492).

**Comparación de metodologías a igual red** (capa 1): gate v5 + ramas chicas
**0.812** contra integrado **0.758**. Para descartar que sea solo por tener
el triple de parámetros (tres redes de 24k), se entrenó el integrado ancho
(95k, más que las tres juntas): **0.792**. Achica la brecha pero no la
cierra. **El separado le gana al integrado también a igual presupuesto.**

## 6. Diagnóstico pulmonar: no es el umbral, es la representación

La sensibilidad de las redes chicas es muy baja (0.04-0.13). Se probó si el
problema era de calibración, con las probabilidades guardadas (sin
reentrenar):

| Modelo | Umbral en val (grilla 0.05) | Umbral en val (grilla 0.01) | Umbral óptimo en el propio test (cota) | Argmax 4 clases + sesgo calibrado en val | **AUC normal vs anormal** |
|---|---|---|---|---|---|
| Integrado base | 0.493 | 0.484 | 0.518 | 0.459 | **0.543** |
| Integrado ancho | — | — | — | — | **0.573** |
| Solo pulmonar | 0.502 | 0.497 | 0.526 | 0.483 | **0.540** |
| CNN6 (pipeline) | 0.537 | 0.533 | 0.578 | — | **0.687** |

Score ICBHI por grabación, regla promedio, 5 folds. Fuente:
`reports/unified_model/ablation/pulmonary_calibration_diagnosis.csv`.

- **"Todo normal" ya da 0.50**: con umbrales óptimos elegidos sobre el
  propio test, las redes chicas llegan como mucho a ~0.53. Mejorar la
  calibración rinde a lo sumo ~0.03.
- **Una decisión sin umbrales tampoco ayuda**: argmax de las 4 clases,
  0.35-0.37; con sesgo por clase ajustado en val, 0.46-0.48.
- **La causa**: para separar grabaciones normales de anormales, las redes
  chicas están cerca del azar (0.54-0.57). Los AUC de crackle/wheeze que se
  ven vienen sobre todo de distinguir *qué tipo* de anomalía hay, no *si la
  hay*. La sensibilidad baja es una consecuencia: si la red no separa
  normales de anormales, el umbral óptimo termina eligiendo "casi todo
  normal".
- **CNN6 (preentrenada en AudioSet, fine-tuning en `avances/17`, 4.6M
  parámetros)** es la única que separa algo (0.69). El preentrenamiento ya
  se usó en la rama pulmonar: el transfer learning no es una palanca
  nueva para el pulmonar, salvo llevarlo al integrado (CNN6 integrada).

**Conclusión**: el techo pulmonar lo ponen los datos (≈ 126 pacientes
pulmonares en ICBHI, ~85 en train por fold, etiquetas ruidosas), no la
calibración ni la capacidad de la red chica. Para la literatura de
referencia, ICBHI es difícil incluso con redes grandes preentrenadas
(scores por ciclo ~0.55-0.65). → **Próxima etapa: más datos pulmonares
(§8).**

## 7. Conclusiones de la comparación (para la memoria)

1. **El pipeline separado es mejor que el modelo único** con la misma red y
   también a igual (o menor) presupuesto de parámetros: 0.812 contra 0.758
   / 0.792 en 6 clases.
2. Hay dos razones, medidas: **(a) interferencia entre tareas** (la
   ablación: compartir la red le cuesta a murmullo y a pulmonar) y **(b)
   conflicto de representación** (crackle quiere ventana FFT corta y
   murmullo larga; cada rama puede tener la suya).
3. **A favor del integrado**: una sola red de 24k parámetros, mejor
   discriminación de dominio que el gate (0.998 contra 0.986, a confirmar
   fuera de dominio) y no necesita un split aparte para el gate.
4. **Subproducto**: la receta del integrado mejoró la rama de murmullo
   (F2 0.725 → 0.794). El mejor sistema hasta ahora es **gate v5 + rama
   cardíaca nueva (`abl_card`) + CNN6: 0.826** en 6 clases.
5. **El pulmonar está limitado por los datos** en las dos metodologías (§6).

## 8. Traspaso: sumar una base de datos pulmonar nueva

Objetivo: más pacientes con **crackle/wheeze** para atacar el límite de §6.
El usuario está buscando candidatas en paralelo. Ejemplos públicos a
evaluar: SPRSound (pediátrico, anotaciones por evento), HF_Lung (anotaciones
por segmento, 4 kHz), KAUH (Fraiwan et al.) y RespiratoryDatabase@TR.
Verificar licencia y etiquetas de cada una.

### 8.1 Requisitos de una base útil

- **Etiquetas mapeables a crackle / wheeze** (por ciclo o por evento con
  tiempos, ideal; por grabación, aceptable). Rhonchi/stridor: decidir si
  van a wheeze (continuos) o se descartan, y documentarlo.
- **ID de paciente**, para que los folds no tengan fuga.
- **Frecuencia de muestreo ≥ 4 kHz**: todo el pipeline trabaja a 4 kHz
  con pasa-bajos de 2048 Hz.
- Equipo y población (edad, foco): ICBHI ya mostró sesgo por equipo
  (`avances/12`) y baja sensibilidad pediátrica (`avances/21` §5). Una
  base pediátrica ataca justamente esa debilidad.

### 8.2 Dónde se engancha en el código

1. **Dataset unificado**: hoy `data/unified_v2/unified_dataset_v2.csv`
   (`MASTER_CSV_V2`), construido con `src/unified_dataset/`
   (`parsers/parse_{circor,icbhi,hls}.py` y `parsers/build_dataset.py`,
   rutas en `config.py`). Agregar un `parse_<base>.py` con las mismas
   columnas: `file_id`, `source_db`, `original_filename`, `subject_id`,
   `subject_type`, `sex`, `age_category`, `age_years`, …,
   `sound_type`, `pathology_label`, `auscultation_location`, `equipment`,
   `sample_rate_hz`, `duration_sec`, `has_crackles`, `has_wheezes`. Conviene una v3 nueva, sin pisar la v2, así
   todos los resultados anteriores siguen siendo reproducibles.
2. **Anotaciones temporales**: hoy `data/pulmonary/cycles_index.csv` (de
   `preprocessing/make_pulmonary_cycles.py`, formato ICBHI: inicio, fin,
   crackle, wheeze). La base nueva tiene que llegar al mismo formato.
   - Si trae eventos (no ciclos), una ventana es positiva si se
     superpone con el evento.
   - Si solo trae la etiqueta de la grabación, ese caso hay que marcarlo
     (etiqueta débil).
3. **Ventanas y caché**: `make_unified_windows.py` (regla de superposición)
   y `make_unified_audio_cache.py` (memmap). Hay que regenerarlos.
4. **Folds**: `splits/make_pulmonary_kfold.py` / `make_unified_kfold.py`.

### 8.3 Cómo evaluar sin romper las comparaciones (importante)

- **Mantener el test de ICBHI idéntico**: mismos pacientes en el mismo
  fold. La base nueva entra **solo en train (y val)** de cada fold, así
  todo lo anterior (0.526 de CNN6, 0.490 de la rama chica, capa 1) sigue
  siendo comparable paciente a paciente. Aparte, reportar en un test
  propio de la base nueva (sus pacientes repartidos en 5 folds), como
  validación cruzada entre bases.
- Primero con la **rama pulmonar chica** (`run_kfold.py --domain
  pulmonary`: es rápida, ~1-1.5 h los 5 folds en CPU). Si mejora, recién ahí
  CNN6 (Colab) y el integrado.
- Métricas a mirar: el score ICBHI de capa 2 y, sobre todo, el **AUC normal
  vs anormal por grabación** (§6), que es lo que tiene que moverse.
- **Riesgo de atajo**: si la base nueva tiene una proporción de anormales
  distinta y un equipo propio, la red puede aprender "equipo X ⇒ anormal".
  Revisar las métricas por fuente y por equipo
  (`pulmonary_by_equipment.csv`), y usar HLS como test externo.

## 9. Pendiente (en orden)

1. **Base pulmonar nueva** (§8), en otra conversación.
2. **HLS externo** del integrado y de la rama cardíaca nueva: medir el
   atajo de dominio (`integration/external_hls.py`, modelos de los 5
   folds, configuración congelada).
3. Adoptar la **rama cardíaca nueva** en el pipeline (por ahora es
   `abl_card`, checkpoints en `src/unified_model/checkpoints_abl_card/`) y
   rehacer los números definitivos.
4. Extras (no antes de lo anterior): CNN6 integrada; reducción de
   parámetros de la variante elegida; entrada con dos resoluciones en el
   integrado y una resolución propia por rama; batches balanceados.

## 10. Estado del repositorio

Sin commitear al cerrar (el usuario commitea por su cuenta):

- `src/common/paths.py` (rutas del modelo único).
- `src/preprocessing/make_unified_audio_cache.py`.
- `src/unified_model/` (`dataset.py`, `model.py`, `run_kfold.py`,
  `evaluate_unified.py`, `evaluate_ablation.py`). Excluir los
  `checkpoints_*/`.
- `reports/unified_model/` (pilotos, ajuste, `runs/*`, `ablation/`, logs).
- Este documento.

`data/` no se versiona; el caché de audio (~740 MB) está en
`data/unified_model/`. Los scripts del diagnóstico de §6 quedaron fuera
del repo; si se quieren reproducir, conviene llevarlos a
`src/unified_model/`.

Comandos para reproducir (desde `desarrollo/src`):

    python unified_model/run_kfold.py --width base --feature logmel --pooling avg_max --run-name v1_base
    python unified_model/run_kfold.py --width wide --feature logmel --pooling avg_max --run-name v1_wide
    python unified_model/run_kfold.py --width base --feature logmel --pooling avg_max --domain cardiac --run-name abl_card
    python unified_model/run_kfold.py --width base --feature logmel --pooling avg_max --domain pulmonary --run-name abl_pulm
    python unified_model/evaluate_unified.py --run-name v1_base
    python unified_model/evaluate_ablation.py
