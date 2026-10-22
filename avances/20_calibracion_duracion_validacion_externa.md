# Etapa 3 — Calibración, duración de ventana y validación externa (HLS-CMDS)

Continúa `avances/19`. Son los puntos 1-3 del orden acordado con el
usuario: estabilizar la calibración, elegir la duración de ventana y hacer
la validación externa de punta a punta con la configuración congelada.
Fechas: 2026-10-20/21.

## 1. Calibración de la agregación por grabación

En `avances/19`, el umbral de grabación (y la regla de agregación) se
calibraba con los pacientes de validación de cada fold. En pulmonar son 16
pacientes por fold: la regla elegida cambiaba de fold a fold y la Sp variaba
± 0.14. Se agregaron cuatro estrategias a `evaluate_integration.py`
(`--strategy`):

| Estrategia | Con qué se calibra | Regla |
|---|---|---|
| `fold_auto` (la de avances/19) | validación del fold | una por fold |
| `fixed_rule` | validación del fold | la misma en todos los folds (mejor score medio de validación) |
| `pooled` | validación de **todos** los folds juntos, menos los pacientes que son test del fold evaluado | una por fold |
| `pooled_fixed` | igual que `pooled` | la misma en todos los folds |

En `pooled`, cada probabilidad de calibración viene de un modelo que no vio
a ese paciente, y los pacientes de test del fold evaluado quedan afuera. No
hay fuga, y pulmonar pasa de 16 a ~100 pacientes de calibración.

Resultados por grabación con ventanas fijas (media ± desvío en 5 folds):

| Estrategia | Murmullo: val F2 | Murmullo: test F2 | Pulmonar: val score | Pulmonar: test score | Pulmonar: Sp test |
|---|---|---|---|---|---|
| `fold_auto` | 0.784 | 0.725 ± 0.017 | 0.580 | 0.516 ± 0.053 | 0.68 ± 0.14 |
| `fixed_rule` | 0.778 | 0.724 ± 0.008 | 0.565 | 0.527 ± 0.036 | 0.77 ± 0.12 |
| `pooled` | 0.759 | 0.727 ± 0.015 | 0.530 | 0.532 ± 0.026 | 0.81 ± 0.08 |
| **`pooled_fixed`** | 0.759 | **0.725 ± 0.013** | 0.526 | **0.526 ± 0.014** | 0.76 ± 0.06 |

- En murmullo las cuatro estrategias dan lo mismo: la calibración ya era
  estable con 113 pacientes de validación por fold.
- En pulmonar, calibrar con más pacientes **reduce el desvío entre folds a
  menos de un tercio** (± 0.053 → ± 0.014).
- Otra señal a favor de `pooled`: su score de validación **anticipa el de
  test** (0.53 contra 0.53). Con 16 pacientes era optimista (0.58 contra
  0.52): los umbrales se ajustaban a esos pocos pacientes.
- **Se adopta `pooled_fixed`** (es el default del script). La regla que
  elige es `frac` en murmullo y `max` en pulmonar. Que pulmonar se quede con
  `max` tiene sentido: la etiqueta de la grabación es "hay al menos un ciclo
  con el evento", y eso es justamente un máximo. En murmullo, `max` sigue
  siendo la peor regla (`avances/10` §14).

Con `pooled_fixed`, la comparación con el oráculo de `avances/19` §4 cambia
algo en pulmonar:

| | Ventanas fijas | Oráculo |
|---|---|---|
| Murmullo, F2 | 0.725 ± 0.013 | 0.734 ± 0.029 |
| Pulmonar, score ICBHI | 0.526 ± 0.014 | 0.542 ± 0.010 |
| Pulmonar, AUC crackle / wheeze | 0.661 / 0.821 | 0.669 / 0.824 |

En `avances/19` las ventanas fijas parecían igualar o superar al oráculo en
pulmonar. Eso venía en parte del ruido de calibración. Con una calibración
estable, **no tener las anotaciones cuesta ~0.01 en murmullo y ~0.016 en
pulmonar**, con AUC prácticamente iguales. La conclusión no cambia: la
pérdida es chica y no justifica reentrenar con ventanas fijas (opción "3b").

**Capa 1 con `pooled_fixed`** (161 pacientes de test del gate):

| | Ruteo perfecto | Sistema (gate v5) |
|---|---|---|
| Murmullo, F2 | 0.671 | 0.660 |
| Pulmonar, score ICBHI | 0.655 | 0.606 |
| Accuracy en 6 clases | 0.824 | 0.812 |

Sin cambios en el desglose pulmonar: los pediátricos siguen en Se 0.125
(Sp 0.955).

## 2. Duración de ventana

Inferencia nueva: murmullo con 2, 3 y 5 s, y pulmonar con 3 y 6 s
(`window_probs/*_<d>s.csv`). En las ventanas de 6 s no se recorta a 4 s:
entran enteras y el pooling global de la CNN6 las resume (nuevo parámetro
`max_frames` de `pulmonary_batch`). **La duración se elige por el score de
validación** (`duration_sweep.csv`), no por el de test.

Con `pooled_fixed`:

| Rama | Ventana | Val | Test |
|---|---|---|---|
| Murmullo (F2) | 2.0 s | 0.749 | 0.704 |
| | **2.5 s** | **0.759** | 0.725 |
| | 3.0 s | 0.750 | 0.722 |
| | 5.0 s | 0.754 | 0.718 |
| Pulmonar (score ICBHI) | 3 s | 0.5263 | 0.529 |
| | **4 s** | **0.5264** | 0.526 |
| | 6 s | 0.522 | 0.532 |

- Se mantienen **2.5 s** para murmullo y **4 s** para pulmonar, las que
  venían de imitar la duración que vio cada modelo al entrenar.
- La duración casi no influye: todo entra en ± 0.02, dentro del desvío entre
  folds.
- En pulmonar, 3 s y 4 s empatan en validación. Con 3 s la regla cambia a
  `mean`, y las AUC de 4 s son un poco mejores (wheeze 0.821 contra 0.790).

**Configuración congelada para la validación externa:** calibración
`pooled_fixed`; murmullo con ventanas de 2.5 s, regla `frac` y umbral de
grabación 0.40 en los 5 folds; pulmonar con ventanas de 4 s, regla `max`,
umbral de crackle 0.70 y umbral de wheeze 0.90-0.95 según el fold
(`external_hls/frozen_config.csv`).

## 3. Validación externa de punta a punta con HLS-CMDS

Script nuevo: `integration/external_hls.py`. HLS-CMDS es un maniquí clínico
grabado con un solo estetoscopio (Littmann CORE). Ningún modelo vio esos
audios, y acá no se elige nada: la configuración es la de §2. Cada rama se
evalúa con sus 5 modelos de fold (media ± desvío entre modelos). Conjuntos:

- **HS**: 50 grabaciones cardíacas, 24 con murmullo. Fibrilación auricular,
  S3, S4, taquicardia y bloqueo AV cuentan como "sin murmullo", igual que
  `abnormal_heart_unspecified` al entrenar.
- **LS**: 50 pulmonares. Las métricas principales usan las 33 con
  equivalente en ICBHI: normal, crackles finos/gruesos y wheezing. Rhonchi y
  pleural rub se analizan aparte.
- **Mix**: 145 tríos de mezcla corazón + pulmón (M) con sus dos componentes
  por separado (H, L). Permiten medir cuánto afecta a cada rama tener el
  otro sonido de fondo.

### 3.1 Gate

| Conjunto | Accuracy por grabación |
|---|---|
| HS (cardíacas) | 0.80 |
| LS (pulmonares) | 1.00 |
| Mix-H / Mix-L | 0.79 / 0.97 |

Se repite la debilidad conocida (`avances/6`): en este dominio el gate
manda a pulmonar el 20% de las grabaciones cardíacas. Las mezclas van a
cardíaco en el 77% de los casos.

### 3.2 Ramas con ruteo perfecto

**Pulmonar (LS, 33 grabaciones):**
- Score ICBHI 0.648 ± 0.049 (Se 0.36, Sp 0.93). Es más alto que en ICBHI
  (0.526), porque los sonidos del maniquí son más "limpios".
- **Wheeze: AUC 1.00 y recall 1.00.** El 70% de los rhonchi se predice como
  wheeze, lo cual es coherente porque un rhonchus es una sibilancia grave.
  Pleural rub queda como normal en el 91% de los casos.
- **Crackle: AUC 0.865, pero recall 0.04.** El modelo *ordena* bien los
  crackles: la AUC es mejor que en ICBHI. Pero en este dominio sus
  probabilidades quedan por debajo del umbral congelado (0.70): el 92-93% de
  los crackles sale como "normal".

**Murmullo (HS, 50 grabaciones):**
- Predice murmullo en casi todo: 89% de las normales y 87-100% de los demás
  tipos. Da recall 0.99 con precisión 0.50, y la AUC es apenas 0.667.
- **El F2 de 0.83 no significa nada acá.** Con 48% de murmullos,
  predecir "murmullo" siempre ya da F2 = 0.82.
- El problema es doble: los scores corridos hacia arriba (umbral) y poca
  capacidad de separar (AUC). Hay que tener en cuenta que el maniquí
  reproduce murmullos sintéticos de adulto, y el modelo se entrenó con
  CirCor, que es pediátrico.

**Efecto del otro sonido de fondo (Mix):**

| | Componente solo | Mezcla |
|---|---|---|
| Pulmonar, AUC crackle | 0.78 | 0.65 |
| Pulmonar, Sp | 0.94 | 0.82 |
| Murmullo, AUC | 0.65 | 0.58 |

El sonido cardíaco superpuesto degrada la detección de crackles. Es
coherente con lo que motivó el pasa-altos de 100 Hz (`avances/14`): el
filtro atenúa S1/S2 pero no los elimina. La de wheeze casi no cambia (AUC
0.99).

### 3.3 Sistema completo (HS + LS)

- Accuracy en 6 clases: 0.504 ± 0.038 (0.537 con ruteo perfecto).
- Murmullo, F2: 0.81 (0.83 con ruteo perfecto). Las dos cifras están
  infladas por los positivos en todo.
- Pulmonar, score ICBHI: 0.648, igual con ruteo perfecto: el gate no se
  equivoca en LS.

### 3.4 Lectura

Fuera de su dominio, la mayor parte del problema está en la
**calibración**, no en la representación. En pulmonar, el orden de los
scores se transfiere muy bien (AUC 0.87 y 1.00), pero los umbrales no. En
murmullo fallan las dos cosas, y el gate pierde el 20% de las cardíacas.
Para la tesis:

1. Las métricas que dependen del umbral **no se transfieren entre
   dominios**. Las AUC sí, en pulmonar.
2. Para usarlo con otro equipo hace falta **recalibrar** con unas pocas
   grabaciones etiquetadas de ese equipo. No necesariamente reentrenar.
3. El sonido cardíaco de fondo es un factor de error medible en la rama
   pulmonar (AUC de crackle 0.78 → 0.65).

Advertencia: HLS es un maniquí (sonidos sintéticos, reproducidos por un
parlante) con 50 + 50 grabaciones, que además son de 15 s y de adulto. Es
una prueba de robustez, no una estimación del rendimiento clínico.

## 4. Archivos

- `src/integration/evaluate_integration.py`: estrategias de calibración
  (`--strategy`, default `pooled_fixed`), `calibrate_folds()` y barrido de
  duración.
- `src/integration/window_inference.py`: `max_frames` para ventanas
  pulmonares de más de 4 s.
- `src/integration/external_hls.py`: validación externa. Con `--smoke` corre
  una prueba del código con 3 audios por conjunto, sin mirar el resultado.
- `reports/integration/`:
  - `duration_sweep.csv`;
  - `{murmur,pulmonary}_summary.csv`, ahora con columnas de estrategia y
    val_score;
  - `evaluate_integration_run.txt`;
  - `window_probs/{murmur_2s,murmur_3s,murmur_5s,pulmonary_3s,pulmonary_6s}.csv`;
  - `window_probs/hls_*.csv`;
  - `external_hls/` (configuración congelada, gate, ramas, tipos de sonido,
    sistema) y `external_hls_run.txt`.

## 5. Para decidir cómo seguir

- **Recalibración por dominio** (lo que sugiere §3.4): recalibrar los
  umbrales con una parte de HLS y evaluar con el resto (por ejemplo, por
  ubicación o por sexo del maniquí). Cuantifica cuánto se recupera sin
  reentrenar. Es barato: no corre modelos.
- **Gate:** su error externo (20% de las cardíacas) sigue siendo el punto
  débil del sistema fuera de dominio.
- **Murmullo fuera de dominio:** la AUC de 0.67 apunta a la representación
  y no solo al umbral. Esto se cruza con el sesgo pediátrico de CirCor.
- **Rama pulmonar:** el sonido cardíaco de fondo y la sensibilidad
  pediátrica (0.125) son los dos frentes concretos para el ajuste fino (en
  Colab).
