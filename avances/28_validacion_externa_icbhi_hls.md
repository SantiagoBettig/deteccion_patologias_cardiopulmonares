# Validación externa del modelo sin gate: ICBHI completo y HLS-CMDS

Continúa `avances/27` §8.1 (etapa 2 del plan hacia el 2026-12-21), con el
protocolo de `avances/23` §8.3. Se evaluaron afuera los dos modelos de 6
clases elegidos (`cnn6_ours64`, 4.57 M de parámetros, y `cs_wide`, 95k) y,
como referencia, el pipeline separado con SPRSound (gate `kfold_cs` →
`abl_card` + `spr_inv`). Ninguno vio ICBHI ni HLS. Fecha: 2026-11-15.

Resumen:

- **La decisión de dominio no se transfiere**: `cs_wide` manda a cardíaco
  el 99.9% de las 920 grabaciones pulmonares de ICBHI y el gate nuevo el
  94%. En HLS, las dos mandan a cardíaco casi todas las pulmonares. Con
  una sola base por dominio, el modelo aprendió "suena a SPRSound ⇒
  pulmonar". La CNN6 falla menos en ICBHI (68% bien), pero en HLS manda
  el 64% de las cardíacas a pulmonar.
- **Las cabezas de tarea sí transfieren en parte** (con ruteo perfecto):
  wheeze transfiere bien (ICBHI 0.79, HLS 0.98), crackle transfiere en HLS
  (0.92) pero no en ICBHI (0.54), y el murmullo del maniquí se ordena muy
  bien con la CNN6 (AUC 0.91, contra 0.63 de `cs_wide`).
- **La CNN6 es la más robusta afuera**, con diferencias significativas
  contra `cs_wide` y el pipeline: AUC normal contra anormal en ICBHI +0.08
  y +0.06, y AUC de murmullo en HLS +0.28 y +0.30. Coincide con
  `avances/27`: la ventaja viene del preentrenamiento en AudioSet.
- **Los umbrales no se transfieren** (el recall de crackle es ~0 en las
  dos bases), igual que en `avances/20`. Recalibrarlos con unas pocas
  grabaciones del equipo nuevo recupera HLS, pero no ICBHI (§7).
- **Con el equipo de CirCor (PASCAL B y KAUH, §12) las tareas
  generalizan al nivel del test interno**: murmullo con AUC 0.94 y
  normal-anormal pulmonar 0.87-0.90 con la CNN6.
  - Lo que se cae afuera es el dominio. `cs_wide` manda a cardíaco hasta
    las grabaciones pulmonares hechas con el estetoscopio de CirCor:
    aprendió "no es el Yunting de SPRSound ⇒ cardíaco" (§10).
- **La edad no explica la caída en ICBHI; el equipo sí**: con el Littmann
  3200, la especificidad cae a 0.04-0.16.

## 1. Protocolo

Decidido con el usuario antes de evaluar:

- **Modelos**: `cnn6_ours64` y `cs_wide` con el criterio B
  (`fold{i}_best.pt`), y el pipeline separado de `avances/25` como
  referencia.
- **Configuración congelada**: la de `evaluate_circor_spr.py`, calculada
  solo con la validación interna de cada fold (CirCor y SPRSound):
  - umbral por ventana de cada tarea;
  - regla de agregación y umbrales por grabación, con la estrategia
    `pooled_fixed`.

  La regla elegida es `frac` en todos los modelos y tareas
  (`frozen_config.csv`). Con ICBHI y HLS no se elige nada.
- **Ventanas**: 4 s con paso 2 s sobre la grabación completa, como en
  `evaluate_full_recordings.py`. El gate usa sus ventanas de 5 s.
- **Cómo se combinan los 5 folds**:
  - **por fold** (principal): cada modelo con su configuración, y se
    reporta media ± desvío, como en `avances/20`;
  - **ensamble** (complemento, lo que se desplegaría): las probabilidades
    por ventana promediadas, la regla común y la media de los umbrales.

  Las dos formas dan cifras muy parecidas. Abajo va el ensamble, salvo que
  se indique otra cosa; las tablas con media ± desvío están en
  `summary_*.csv`.
- **AUC con la media de las probabilidades por ventana**: no depende de
  umbrales. La regla `frac` puede valer 0 en todas las grabaciones fuera
  de dominio, y entonces la AUC sale 0.5 por empate.
- **Dos variantes de las salidas de tarea de los modelos de 6 clases**:
  - **absolutas**: p(crackle) = p₃ + p₅, como al entrenar;
  - **condicionales** (`--conditional`): p(crackle | pulmonar) =
    p(crackle) / P(pulmonar), y lo mismo con wheeze, pulm_normal y
    murmullo (dividido por P(cardíaco)). Es el ruteo perfecto de verdad:
    si el modelo pone toda la masa en el dominio equivocado, la cabeza del
    otro queda aplastada.

  La condicional también se aplica al calibrar en la validación interna.
  Ahí casi no cambia nada, porque P(dominio correcto) ≈ 1: el único umbral
  que se movió es el de wheeze de `cs_wide`, de 0.33 a 0.39. Las ramas del
  pipeline ya son condicionales.
- **Diferencias apareadas** del ensamble con IC95 bootstrap (2000
  repeticiones), por paciente en ICBHI y por grabación en HLS.

## 2. Implementación

- **`common/paths.py`**: rutas `HLS_*` y `EXTERNAL_REPORTS_DIR`.
- **`preprocessing/make_unified_audio_cache.py --data hls`**: caché de los
  535 audios de HLS (HS 50, LS 50 y Mix 3 × 145; 2.2 h), con el mismo
  preprocesamiento. La tabla de etiquetas es la de
  `external_hls.load_hls` → `data/hls/recordings.csv`. ICBHI ya estaba en
  la caché del modelo único.
- **`unified_model/external_inference.py`**: inferencia común para la
  `UnifiedCNN` (`build_model`), la CNN6 (`Cnn6Classifier`) y el gate. Usa
  `AudioBank` y `predict` de `run_cnn6_gpu.py`. Guarda las probabilidades
  por ventana de los 5 folds y retoma si se corta.
  - **Controles**: reproduce las probabilidades ya guardadas de `spr_inv`
    sobre ICBHI (diferencia máxima 1e-7) y las del gate `kfold_cs` sobre
    CirCor y SPRSound (3e-8).
  - **Tiempos en CPU**: la CNN6 tarda ~6.5 min por fold en ICBHI (9029
    ventanas) y ~2.5 min en HLS; el resto, segundos.
- **`unified_model/external_validation.py`**: configuración congelada,
  predicción por grabación de los tres sistemas, métricas por fold y del
  ensamble, grupos (equipo, edad) y diferencias apareadas.

## 3. Dominio: el punto débil

Fracción de grabaciones mandadas al dominio correcto (ensamble; por fold
entre paréntesis):

| Conjunto | `cnn6_ours64` | `cs_wide` | Pipeline (gate `kfold_cs`) | Gate v5 (`avances/20`) |
|---|---|---|---|---|
| ICBHI (920, pulmonares) | **0.680** (0.67 ± 0.07) | 0.001 | 0.058 | n/c (lo vio al entrenar) |
| HLS-HS (50, cardíacas) | 0.360 | 1.000 | 1.000 | 0.80 |
| HLS-LS (50, pulmonares) | 0.545 | 0.030 | 0.000 | 1.00 |
| Mix-M a cardíaco | 0.70 | 1.00 | 0.97 | 0.77 |

- **No es un error de código**: en HLS-LS, P(cardíaco) tiene una mediana
  de 0.93 en `cs_wide` y de 0.99 en el gate. Las dos fallas son estables
  entre folds.
- **Lectura**: CirCor y SPRSound son las únicas bases de cada dominio, así
  que separar cardíaco de pulmonar es lo mismo que separar los dos equipos.
  `avances/23` §8.1 lo daba por bueno ("separar cardíaco de pulmonar es lo
  que queremos"), pero afuera el modelo usa el equipo: todo lo que no suena
  a SPRSound va a cardíaco.
  - **Indicio en el espectro** (energía relativa en 500-1000 Hz):
    SPRSound −19 dB; CirCor −7.7; HLS-LS −7.9.
  - HLS además no tiene energía por encima de 1 kHz (−42 dB, contra −28
    de SPRSound).
- **Afecta igual al modelo sin gate y al pipeline**, así que no es una
  propiedad de la metodología: es de los datos de entrenamiento. El gate
  v5, entrenado con CirCor + ICBHI (cuatro equipos pulmonares), acertaba
  todo HLS-LS.
- **La CNN6 es la menos sesgada**: manda bien el 68% de ICBHI, con 92% en
  el Littmann 3200 y 65% en el AKG. Pero en HLS confunde para los dos
  lados, así que su decisión de dominio tampoco es confiable afuera.

## 4. ICBHI completo (920 grabaciones, 126 pacientes)

### 4.1 Por grabación (ensamble)

| | `cnn6_ours64` | `cs_wide` | Pipeline |
|---|---|---|---|
| AUC normal-anormal, absoluta | 0.510 | 0.462 | 0.548 |
| AUC normal-anormal, condicional | **0.602** | 0.522 | 0.548 |
| AUC crackle (cond.) | 0.544 | 0.505 | 0.507 |
| AUC wheeze (cond.) | **0.789** | 0.699 | 0.723 |
| Score ICBHI, ruteo perfecto (cond.) | 0.493 (Se 0.12 / Sp 0.87) | 0.477 (0.07 / 0.89) | 0.475 (0.09 / 0.86) |
| Score ICBHI del sistema | 0.311 | 0.000 | 0.034 |
| Accuracy de 6 clases del sistema | 0.220 | 0.000 | 0.027 |

- **El pipeline coincide con lo esperado**: `spr_inv` da 0.548, contra
  0.539 en el test cruzado de `avances/23`.
- **Diferencias apareadas** (AUC normal-anormal condicional, bootstrap por
  paciente):

  | Diferencia | Valor | IC95 |
  |---|---|---|
  | CNN6 − `cs_wide` | +0.080 | [+0.032, +0.140] |
  | CNN6 − pipeline | +0.055 | [+0.004, +0.112] |
  | `cs_wide` − pipeline | −0.026 | [−0.066, +0.012] |

- **Wheeze transfiere y crackle no**: lo mismo que en `avances/23` (el
  crackle de SPRSound no transfiere a ICBHI).
- **Los umbrales congelados dejan Se ~0.1**: el recall de crackle es
  0.01-0.02 y el de wheeze 0.20-0.47.

### 4.2 Por equipo y por edad (CNN6, condicional, ensamble)

| Grupo | Grabaciones | Dominio | AUC normal-anormal | AUC wheeze | Sp |
|---|---|---|---|---|---|
| Littmann Classic II SE | 141 | 0.70 | 0.776 | 0.843 | 0.95 |
| Meditron | 36 | 0.78 | 0.786 | 0.747 | 1.00 |
| AKG C417L | 683 | 0.65 | 0.613 | 0.796 | 0.93 |
| Littmann 3200 | 60 | 0.92 | 0.597 | 0.535 | **0.16** |
| AKG + LittC2SE, adultos | 748 | 0.67 | 0.668 | 0.810 | 0.94 |
| AKG + LittC2SE, menores de 18 | 76 (49 pacientes) | 0.57 | 0.614 | 0.756 | 0.93 |

- **La edad no explica la caída**. En los dos equipos que tienen niños, los
  menores no salen mejor que los adultos:
  - CNN6: 0.61 contra 0.67;
  - `cs_wide`: 0.60 contra 0.60;
  - pipeline: 0.71 contra 0.60 (es el único que mejora).

  Con 49 pacientes, ninguna de estas diferencias es concluyente.
- **El equipo pesa más que la edad**:
  - El Littmann 3200 hace que todos los sistemas predigan wheeze en casi
    todo: Sp 0.04 a 0.16 y recall de wheeze 0.84-0.93.
  - En el AKG, que tiene el 74% de las grabaciones, el rendimiento es
    mediocre.

## 5. HLS-CMDS (maniquí, Littmann CORE)

### 5.1 Cardíaco (HS, 50 grabaciones, 24 con murmullo)

Condicional, ensamble:

| | `cnn6_ours64` | `cs_wide` | Pipeline | `avances/20` |
|---|---|---|---|---|
| AUC murmullo | **0.909** | 0.625 | 0.609 | 0.667 |
| Recall / Precisión | 0.79 / 0.86 | 1.00 / 0.52 | 0.96 / 0.59 | 0.99 / 0.50 |
| Predice murmullo en | 44% | 92% | 78% | ~90% |
| F2 | 0.805 | 0.845 | 0.852 | 0.83 |
| F2 del sistema (con dominio) | 0.385 | 0.845 | 0.852 | 0.81 |

- **La CNN6 es el único modelo que separa murmullos en el maniquí**:
  - CNN6 − `cs_wide`: +0.28 [+0.14, +0.46];
  - CNN6 − pipeline: +0.30 [+0.14, +0.46].
- **El F2 alto de `cs_wide` y del pipeline no vale nada**: predicen
  murmullo en casi todo, y con 48% de murmullos eso ya da F2 ≈ 0.82, igual
  que en `avances/20`.
- **El sistema con la CNN6 pierde lo que gana**, porque manda el 64% de HS
  a pulmonar.
- **Con las salidas absolutas**, la CNN6 da recall 0.21 y AUC 0.886: el
  modelo pone masa pulmonar en estos audios.

### 5.2 Pulmonar (LS, las 33 con equivalente en ICBHI)

Condicional, ensamble:

| | `cnn6_ours64` | `cs_wide` | Pipeline | `avances/20` |
|---|---|---|---|---|
| AUC normal-anormal | **0.996** | 0.810 | 0.849 | — |
| AUC crackle | **0.921** | 0.820 | 0.778 | 0.865 |
| AUC wheeze | 0.984 | 1.000 | 1.000 | 1.00 |
| Score ICBHI (Se / Sp) | 0.667 (0.33 / 1.00) | 0.667 (0.33 / 1.00) | 0.667 (0.33 / 1.00) | 0.648 (0.36 / 0.93) |
| Recall crackle / wheeze | 0.00 / 1.00 | 0.00 / 1.00 | 0.00 / 1.00 | 0.04 / 1.00 |

- **El orden se transfiere y los umbrales no**:
  - wheeze se detecta con los tres sistemas;
  - crackle se ordena bien (CNN6 0.92), pero ningún crackle supera el
    umbral congelado.

  Es la misma lectura que en `avances/20` §3.4, ahora con modelos
  entrenados en otra base.
- **Rhonchi y pleural rub**: el pipeline manda el 75% de los rhonchi a
  wheeze. Los modelos de 6 clases, con las salidas absolutas, los mandan
  a cardíaco (`types_hls.csv`).

### 5.3 Efecto del sonido cardíaco de fondo (Mix, condicional, ensamble)

| | Componente solo | Mezcla |
|---|---|---|
| AUC crackle: CNN6 / `cs_wide` / pipeline | 0.94 / 0.93 / 0.89 | 0.72 / 0.79 / 0.82 |
| Sp pulmonar: CNN6 / `cs_wide` / pipeline | 1.00 / 1.00 / 0.86 | 0.39 / 0.57 / 0.61 |
| AUC murmullo: CNN6 / `cs_wide` / pipeline | 0.85 / 0.55 / 0.61 | 0.83 / 0.64 / 0.70 |

- **El corazón de fondo sigue degradando crackle** y hace caer la
  especificidad pulmonar, como en `avances/20`.
- **El murmullo casi no se afecta** por el pulmón de fondo.

### 5.4 Sistema (HS + las 33 de LS)

Accuracy de 6 clases:

| Sistema | Accuracy |
|---|---|
| Pipeline | 0.398 |
| `cs_wide` | 0.361 |
| CNN6 | 0.313 (condicional 0.349) |
| Pipeline viejo (`avances/20`) | 0.504 |

Las diferencias entre los tres no son significativas (los IC95 tocan el
cero). Todos quedan por debajo del pipeline viejo, por el dominio.

## 6. Test interno contra externo

El test interno son los pacientes de test de cada fold de CirCor y
SPRSound (`avances/27` §4), y es la estimación principal del rendimiento
en la población de entrenamiento. La validación externa mide cuánto de eso
se generaliza a otros equipos y poblaciones. En la memoria van los dos
niveles.

Media ± desvío de 5 folds; afuera, las salidas condicionales:

| Métrica | Base | `cnn6_ours64` | `cs_wide` | Pipeline |
|---|---|---|---|---|
| Accuracy de dominio | Interno (9083 grabaciones) | 0.9997 | 0.9992 | 0.9912 |
| | ICBHI | 0.67 | 0.004 | 0.07 |
| | HLS (HS + LS) | 0.46 | 0.62 | 0.65 |
| Accuracy de 6 clases del sistema | Interno | **0.800** | 0.776 | 0.771 |
| | ICBHI | 0.22 | 0.00 | 0.03 |
| | HLS (HS + 33 de LS) | 0.35 | 0.37 | 0.40 |
| AUC murmullo | CirCor (test) | 0.927 | 0.933 | 0.927 |
| | HLS-HS | **0.89** | 0.62 | 0.60 |
| AUC normal-anormal pulmonar | SPRSound (test) | 0.844 | 0.793 | 0.767 |
| | ICBHI | **0.60** | 0.53 | 0.54 |
| | HLS-LS | **0.96** | 0.76 | 0.83 |
| AUC crackle | SPRSound (test) | 0.778 | 0.686 | 0.686 |
| | ICBHI | 0.55 | 0.51 | 0.52 |
| | HLS-LS | **0.92** | 0.77 | 0.75 |
| AUC wheeze | SPRSound (test) | 0.828 | 0.797 | 0.783 |
| | ICBHI | **0.78** | 0.68 | 0.69 |
| | HLS-LS | 0.92 | 0.96 | 0.99 |

- **Adentro**, los tres sistemas separan el dominio casi sin errores. La
  CNN6 es la mejor en el sistema (+2.4 puntos, `avances/27`), y la ventaja
  es toda pulmonar.
- **Afuera**, el dominio se cae en los tres. En las tareas, la ventaja de
  la CNN6 se mantiene o crece:
  - en murmullo, adentro empatan los tres, y en HLS la CNN6 saca casi 0.3
    de ventaja;
  - en pulmonar, la CNN6 sigue siendo la mejor en ICBHI y en HLS.
- **HLS-LS supera al test interno** en varias AUC, porque los sonidos del
  maniquí son limpios y prototípicos (`avances/20`). No es una estimación
  del rendimiento clínico.

## 7. Recalibración de umbrales (sin reentrenar)

Script: `unified_model/recalibrate_external.py`. Lee las predicciones
condicionales y no corre modelos. Mismo planteo que `avances/21`: cuánto se
recupera con unas pocas grabaciones etiquetadas del equipo nuevo. Se
decidió con el usuario:

- **Regla `mean` con un umbral por tarea**. La regla congelada, `frac`,
  depende también del umbral por ventana y afuera vale 0 en casi todas
  las grabaciones: recalibrar solo el umbral por grabación no la arregla.
- Grillas de 0.01 a 0.99 con paso 0.01.
- **HLS: un maniquí contra el otro** (F ↔ M, en las dos direcciones). Se
  recalibran:
  - el umbral de dominio (máxima accuracy balanceada en HS + LS);
  - el de murmullo (máximo F2 y, aparte, máxima accuracy balanceada);
  - crackle y wheeze juntos (máximo score ICBHI).
- **ICBHI: por equipo, con el 20% de los pacientes**. Los pacientes se
  reparten al azar en 5 partes; cada paciente queda en la misma parte en
  todos los equipos. Se calibra con una parte y se evalúa con las otras
  cuatro, rotando. Solo crackle y wheeze: con una sola clase no se puede
  recalibrar el dominio.

### 7.1 HLS (ensamble, media de las dos direcciones)

| | `cnn6_ours64` | `cs_wide` | Pipeline |
|---|---|---|---|
| Dominio, accuracy balanceada: congelado → recalibrado | 0.52 → 0.46 | 0.51 → **0.93** | 0.50 → 0.76 |
| Murmullo, accuracy balanceada (criterio de accuracy balanceada) | 0.84 → 0.80 | 0.58 → 0.67 | 0.68 → 0.61 |
| Score ICBHI pulmonar | 0.65 → 0.76 | 0.65 → **0.79** | 0.65 → 0.68 |
| Recall de crackle | 0.00 → 0.86 | 0.00 → 0.50 | 0.00 → 0.43 |
| Accuracy de 6 clases del sistema | 0.35 → 0.32 | 0.35 → **0.65** | 0.40 → 0.47 |

Por modelo de fold, `cs_wide` sube de 0.37 ± 0.04 a 0.56 ± 0.09.

- **Con el mismo equipo, el dominio de `cs_wide` se recupera moviendo el
  umbral**. Su P(cardíaco) ordena bien HS contra LS, pero está corrida
  hacia arriba. Con ~40 grabaciones del maniquí, el sistema pasa de 0.35 a
  0.65. Es más que el pipeline viejo recalibrado (0.56, `avances/21`).
- **El dominio de la CNN6 no se arregla con el umbral** (0.46): en HLS su
  P(cardíaco) no separa los dos dominios. Pasa lo contrario que con las
  tareas, donde la CNN6 es la que mejor ordena.
- **Pulmonar: el problema era de umbral, como en `avances/21`**. El recall
  de crackle pasa de 0 a 0.5-0.86.
- **Murmullo: la recalibración no mejora a la CNN6**, que ya era la mejor
  con los umbrales congelados (0.84). Con ~25 grabaciones de calibración
  el umbral sobreajusta.

### 7.2 ICBHI (ensamble, media de las 5 partes)

| | `cnn6_ours64` | `cs_wide` | Pipeline |
|---|---|---|---|
| Score ICBHI (ruteo perfecto): congelado → recalibrado | 0.49 → 0.47 | 0.48 → 0.45 | 0.47 → 0.44 |
| Se / Sp recalibrado | 0.16 / 0.78 | 0.14 / 0.76 | 0.14 / 0.75 |
| Recall de crackle | 0.01 → 0.14 | 0.02 → 0.17 | 0.00 → 0.16 |

- **En ICBHI, recalibrar no ayuda**: cambia sensibilidad por especificidad
  y el score baja un poco. Las AUC son 0.55 en crackle y 0.60 en normal
  contra anormal, así que no hay umbral que salve la representación. Pasa
  en todos los equipos (`icbhi_by_device.csv`).
- **El dominio de ICBHI no se pudo recalibrar** (una sola clase). Por lo
  visto en HLS, para `cs_wide` sería esperable que un umbral propio del
  equipo ayudara, pero con estos datos no se puede medir.

### 7.3 Lectura

- **Para un equipo nuevo, la recalibración liviana sirve si la
  representación ordena bien**: es el caso de HLS, donde con `cs_wide` se
  recupera hasta el dominio.
- **No sirve cuando no ordena bien**: es el caso de ICBHI, con sus
  grabaciones clínicas de adultos.
- **Ningún sistema gana en todo**:
  - la CNN6 tiene la mejor representación de las tareas, pero su decisión
    de dominio no se puede recalibrar;
  - `cs_wide` tiene peores tareas, pero un dominio que se arregla con el
    umbral.

## 8. Conclusiones

1. **El cuello de botella fuera de dominio es la decisión de dominio, y
   viene de los datos, no de la metodología**:
   - con una base por dominio, "cardíaco contra pulmonar" se aprende como
     "CirCor contra SPRSound";
   - le pasa igual al modelo sin gate y al gate del pipeline;
   - el gate v5, con cuatro equipos pulmonares, no tenía el problema.

   Es la principal limitación a reportar, y corrige lo que `avances/23`
   §8.1 daba por resuelto.
2. **La representación de las tareas transfiere en parte, y más con la
   CNN6**. La CNN6 mejora significativamente a `cs_wide` y al pipeline en
   el orden de anormalidad en ICBHI y en el de murmullo en HLS. La ventaja
   interna de la CNN6 (`avances/27`) se mantiene afuera, y crece en
   murmullo. Refuerza que la CNN6 sea el modelo principal.
3. **Wheeze transfiere a todas las bases. Crackle transfiere al maniquí,
   pero no a ICBHI**.
4. **Los umbrales no transfieren**. Las métricas con umbral afuera (score
   ICBHI, F2, accuracy de 6 clases) reflejan sobre todo la calibración y
   el dominio. Las AUC son la medida honesta de la representación.
5. **Alcance pediátrico**: en ICBHI los niños no salen mejor que los
   adultos con el mismo equipo. La diferencia de equipo domina sobre la de
   población.
6. **Recalibrar los umbrales con unas pocas grabaciones del equipo nuevo
   sirve solo si la representación ordena bien** (§7):
   - en HLS recupera lo pulmonar en los tres sistemas, y también el dominio
     de `cs_wide` (accuracy de 6 clases de 0.35 a 0.65);
   - en ICBHI no mejora nada.

   La limitación queda así: el sistema se puede llevar a otro equipo con
   una recalibración liviana cuando el equipo y los sonidos se parecen a
   los del entrenamiento. Con grabaciones clínicas de adultos y otros
   equipos (ICBHI) haría falta reentrenar o tener más diversidad de datos.

## 9. Comandos

Desde `desarrollo/src` (en Windows, con `PYTHONIOENCODING=utf-8`):

    python preprocessing/make_unified_audio_cache.py --data hls
    python unified_model/external_inference.py --models gate_kfold_cs abl_card spr_inv cs_wide cnn6_ours64 --data HLS ICBHI
    python unified_model/external_validation.py
    python unified_model/external_validation.py --conditional
    python unified_model/recalibrate_external.py

Salidas en `reports/external_validation/`, con la variante condicional en
`condicional/`:

- `frozen_config.csv`;
- `predictions_{ICBHI,HLS}.csv`;
- `metrics_*.csv` y `summary_*.csv` (media ± desvío y ensamble);
- `paired_{ICBHI,HLS}.csv`;
- `types_hls.csv`;
- `window_probs/` (probabilidades por ventana: no se recalculan si ya
  están);
- logs `external_inference_run.txt` y `external_validation_run.txt`;
- `recalibration/`: `hls_{per_split,summary}.csv`,
  `icbhi_{per_split,summary}.csv`, `icbhi_by_device*.csv` y
  `recalibration_run.txt`.

## 10. Qué diferencia a un equipo de otro

Script: `eda/device_characterization.py` → `reports/external_validation/devices/`
(`features.csv`, `group_summary*.csv`, `rule*.csv`, `ltas.png`, `bands.png`,
`rule_vs_models.png`).

- **Qué se mide**: el audio que ve el modelo (cachés preprocesadas), por
  equipo y dominio.
- **Rasgos**: espectro promedio, fracción de energía por banda, centroide,
  rolloff del 95%, piso de ruido y recortes.
- **Control de contenido**: para separar el efecto del equipo del de lo que
  se graba, se repite solo con grabaciones pulmonares normales.
- **Equipos de entrenamiento**, según los papers de cada base:
  - CirCor: Littmann 3200 con el DigiScope Collector, a 4 kHz (Oliveira et
    al., 2022);
  - SPRSound: estetoscopio Yunting modelo II, a 8 kHz (Zhang et al., 2022).

### 10.1 La firma del Yunting II

Mediana de cada grupo:

| Grupo | Energía 20-100 Hz | 100-200 Hz | 800-1600 Hz | Centroide | Rolloff 95% |
|---|---|---|---|---|---|
| SPRSound (Yunting II), pulmonar | **−11.1 dB** | **−1.9** | −32.0 | 170 Hz | 297 Hz |
| CirCor (Littmann 3200), cardíaco | −4.8 | −7.7 | **−12.2** | 293 Hz | 859 Hz |
| ICBHI Littmann 3200, pulmonar | −1.0 | −8.4 | −41.8 | 70 Hz | 191 Hz |
| ICBHI Littmann Classic II SE | −0.2 | −14.6 | −40.5 | 41 Hz | 94 Hz |
| ICBHI AKG C417L | −0.3 | −12.4 | −35.0 | 51 Hz | 117 Hz |
| ICBHI Meditron | −0.2 | −15.5 | −29.9 | 43 Hz | 86 Hz |
| HLS (Littmann CORE), cardíaco | −1.0 | −10.6 | −35.9 | 94 Hz | 231 Hz |
| HLS (Littmann CORE), pulmonar | −1.7 | −15.8 | −32.6 | 158 Hz | 539 Hz |

- **El Yunting II funciona como un pasa-banda**: corta por debajo de
  ~100 Hz y tiene un pico en ~130 Hz.
  - En todos los demás equipos (los dos Littmann, los micrófonos de ICBHI,
    el Littmann CORE) la mayor parte de la energía está debajo de 100 Hz,
    igual que en CirCor.
  - **No es por el contenido**: con solo las grabaciones pulmonares
    normales, la banda de 20-100 Hz da −10.9 dB en SPRSound contra −0.1 a
    −1.3 dB en los cuatro equipos de ICBHI (`ltas.png`, panel derecho).
- **Consecuencia**: en el entrenamiento, "poca energía debajo de 100 Hz y
  pico en 130 Hz" es lo mismo que "pulmonar". Cualquier grabación pulmonar
  de otro equipo se parece más a CirCor que a SPRSound.
- **El mismo modelo de estetoscopio no garantiza la misma señal**:
  - CirCor tiene mucha más energía en 800-1600 Hz que las grabaciones
    pulmonares de ICBHI hechas con el mismo Littmann 3200 (−12 contra
    −42 dB).
  - Puede venir del modo de filtro del Littmann (campana, diafragma o
    extendido), de la cadena del DigiScope Collector o del ruido ambiente
    de una pesquisa ambulatoria con niños. Con estos datos no se puede
    separar.
- **El piso de ruido y los recortes no distinguen equipos**: el piso queda
  entre −5 y −7 dB en todos los grupos, y la mediana de recortes es 0.

### 10.2 Una regla espectral reproduce la decisión de dominio

- **La regla**: una regresión logística con los 9 rasgos espectrales,
  entrenada solo para separar CirCor de SPRSound.
- **Adentro separa casi perfecto**: 0.9975 de accuracy con 5 folds por
  paciente. El espectro del equipo solo ya alcanza para resolver el
  problema de dominio del entrenamiento: el atajo estaba disponible.
- **Afuera, la regla manda a cardíaco**:
  - entre el 73% y el 100% de las grabaciones pulmonares de ICBHI (según
    el equipo);
  - el 96% de HLS-LS.

  Es lo mismo que hacen `cs_wide` (98-100%) y el gate (77-100%)
  (`rule_vs_models.png`).
- **Correlación con la regla** (Spearman, ICBHI): 0.48 con `cs_wide`, 0.29
  con el gate y 0.24 con la CNN6.
- **La CNN6 no sigue la regla**: manda a cardíaco solo el 8-35% de las
  pulmonares. Su preentrenamiento en AudioSet le da otras pistas además
  del espectro, aunque en HLS-HS se equivoca para el otro lado.

**Lectura para la tesis**: la falla de dominio afuera se explica por una
diferencia física concreta y medible. El estetoscopio de la única base
pulmonar filtra las bajas frecuencias, y los modelos chicos (y el gate)
aprendieron ese filtro como si fuera la definición de "pulmonar". Esto
también orienta la augmentation de equipo: pasa-altos y realces de graves
aleatorios (de 50 a 150 Hz) para que la energía debajo de 100 Hz deje de
servir para decidir el dominio.

## 11. Bases externas con los mismos equipos del entrenamiento

Búsqueda del 2026-11-15:

| Base | Equipo | Dominio | Qué aporta | Acceso |
|---|---|---|---|---|
| **KAUH** (Fraiwan et al., 2021; King Abdullah University Hospital, Jordania) | Littmann 3200 | Pulmonar, 112 sujetos, en su mayoría adultos (Table 1 incluye desde 12 años) | **Mismo equipo que CirCor, otro dominio**: la prueba directa del atajo. Trae cada grabación con los tres modos de filtro del 3200 (campana, diafragma, extendido) | Mendeley Data, CC BY 4.0 |
| **PASCAL, Dataset B** (2011; Real Hospital Português, Recife) | DigiScope (el mismo sistema de CirCor; que sea el Littmann 3200 es probable, pero falta confirmarlo) | Cardíaco, 656 grabaciones (normal, murmullo, extrasístole), niños y adultos | Mismo equipo que CirCor, mismo dominio: valida murmullo sin cambio de equipo | Público (challenge PASCAL) |
| **HF_Lung V1/V2** (Taiwán) | Littmann 3200 + un equipo propio (HF-Type-1) | Pulmonar, adultos en UCI | Mismo equipo que CirCor, con eventos anotados; la parte del 3200 se puede separar | GitLab (licencia a confirmar) |
| **RespiratoryDatabase@TR** (Turquía) | Littmann 3200 | Pulmonar y cardíaco, EPOC | También tiene grabaciones cardíacas | Por pedido |
| Otra base con el **Yunting II** | — | — | No se encontró ninguna pública: SPRSound es la única | — |

- **ICBHI ya trae 60 grabaciones pulmonares con el Littmann 3200** (§4.2).
  Ahí `cs_wide` manda el 100% a cardíaco y el gate el 78%, pero la CNN6
  solo el 8%. KAUH tiene más casos de ese mismo cruce (Littmann 3200,
  pulmonar), y además permite medir el efecto del modo de filtro.
- **Ninguna base pública reproduce el Yunting II**, así que no se puede
  validar la rama pulmonar afuera sin cambio de equipo. Es parte de la
  limitación.

## 12. Validación con el equipo de CirCor: PASCAL B y KAUH

Descargadas con el permiso del usuario el 2026-11-15.

- **KAUH**: de Mendeley Data (jwyy9np4gv v3, CC BY 4.0). Los SHA-256 de
  `Audio Files.zip` y del Excel coinciden con los que publica Mendeley.
- **PASCAL B**: de la página del challenge (`Btraining_{normal,murmur,extrasystole}.zip`).
  El test sin etiquetas no se usa.
- **Dónde quedan**:
  - crudos en `data/raw/{KAUH,PASCAL}/`;
  - tablas en `data/{kauh,pascal}/recordings.csv`
    (`preprocessing/make_external_recordings.py`);
  - cachés con `make_unified_audio_cache.py --data kauh|pascal`.
- **Inferencia y métricas**: las mismas de las otras bases, con la
  configuración congelada.
- **Cambio en la inferencia del gate**: para que el pipeline cubra todas
  las grabaciones, las de menos de 2.5 s, que el gate no cortaba en
  ventanas, ahora reciben una sola ventana rellenada con ceros. En ICBHI,
  HLS y KAUH no cambia nada (no hay grabaciones tan cortas).

### 12.1 PASCAL B: mismo equipo y mismo dominio que CirCor (cardíaco)

- **Datos**: 461 grabaciones de 165 pacientes, a 4 kHz, con mediana de
  5 s.
  - 95 tienen murmullo (21%).
  - Las 46 extrasístoles cuentan como "sin murmullo".
  - 149 son de las carpetas "ruidosas".

Condicional, ensamble:

| | `cnn6_ours64` | `cs_wide` | Pipeline | Interno (CirCor, test) |
|---|---|---|---|---|
| Dominio | **0.993** | 0.918 | 0.896 | ≥ 0.99 |
| AUC murmullo | **0.936** | 0.878 | 0.893 | 0.927 / 0.933 / 0.927 |
| Recall / Precisión | 0.84 / 0.60 | 0.97 / 0.33 | 0.93 / 0.36 | |
| Predice murmullo en | 29% | 61% | 54% | |
| F2 | **0.778** | 0.695 | 0.701 | 0.782 / 0.789 / 0.794 |
| Accuracy de 6 clases del sistema | **0.846** | 0.573 | 0.627 | |

- **Con el mismo equipo, el murmullo generaliza al nivel del test
  interno**. La CNN6 iguala su F2 interno (0.78) y su AUC (0.94 contra
  0.93). Las grabaciones ruidosas le cuestan poco: AUC 0.944 en las limpias
  y 0.919 en las ruidosas.
- **`cs_wide` y el pipeline ordenan bien, pero predicen murmullo de más**:
  marcan el 54-61% de las grabaciones, cuando hay 21%. Es un corrimiento de
  calibración del mismo tipo que el de HLS.
- **Diferencias apareadas** (bootstrap por paciente):

  | Métrica | CNN6 − `cs_wide` | CNN6 − pipeline |
  |---|---|---|
  | AUC de murmullo | +0.058 [+0.023, +0.092] | +0.044 [+0.014, +0.074] |
  | Accuracy del sistema | +0.27 [+0.20, +0.34] | +0.22 [+0.15, +0.28] |

- **Lectura**: la caída del murmullo en HLS (AUC 0.62 con los modelos
  chicos) venía del equipo y del maniquí, no de la tarea.

### 12.2 KAUH: el equipo de CirCor en el otro dominio (pulmonar)

- **Datos**: 112 sujetos (2 menores de 18 años), cada uno grabado con los
  tres modos de filtro del Littmann 3200.
  - Tipos de sonido: 35 normales, 35 con crackle, 43 con wheeze y 2 con
    los dos.
  - Una grabación solo bronquial queda afuera de las métricas.

Ensamble (dominio por modo; tareas condicionales):

| | Modo | `cnn6_ours64` | `cs_wide` | Pipeline |
|---|---|---|---|---|
| Dominio (a pulmonar) | campana | **1.00** | 0.01 | **1.00** |
| | diafragma | 0.72 | 0.00 | 0.44 |
| | extendido | 0.68 | 0.00 | 0.03 |
| AUC normal-anormal | campana | **0.87** | 0.77 | 0.78 |
| | diafragma | **0.90** | 0.87 | 0.82 |
| | extendido | **0.90** | 0.85 | 0.84 |
| AUC crackle | diafragma | **0.80** | 0.70 | 0.71 |
| AUC wheeze | diafragma | **0.80** | 0.77 | 0.77 |
| Score ICBHI | diafragma | **0.71** | 0.63 | 0.36 |
| Accuracy de 6 clases del sistema | diafragma | **0.49** | 0.00 | 0.21 |

- **Prueba directa del atajo**: `cs_wide` manda a cardíaco el 99-100% de
  las grabaciones pulmonares hechas con el estetoscopio de CirCor. No
  aprendió "Littmann 3200 ⇒ cardíaco", sino "no es el Yunting ⇒
  cardíaco", que es lo que predice §10.
- **El modo de filtro cambia la decisión de dominio**:
  - en modo campana (corta todo lo que pasa de ~200 Hz; 800-1600 Hz a
    −76 dB) la CNN6 y el gate aciertan el 100%;
  - en modo diafragma y extendido, que dejan pasar más agudos, caen.
- **Las tareas pulmonares transfieren al nivel del test interno**:
  - la AUC normal-anormal de KAUH (0.87-0.90 con la CNN6) iguala o supera
    la del test de SPRSound (0.84);
  - crackle (0.80) y wheeze (0.80) también;
  - ICBHI es la base que se cae, no "los adultos" en general.
- **Diferencias apareadas de la AUC normal-anormal** (bootstrap por
  sujeto):

  | Modo | CNN6 − `cs_wide` | CNN6 − pipeline |
  |---|---|---|
  | Campana | +0.10 [+0.03, +0.17] | +0.09 [+0.02, +0.16] |
  | Diafragma | +0.03 (n.s.) | +0.09 [+0.02, +0.16] |

### 12.3 Qué agregan al panorama

- **El problema de dominio se confirma con el equipo de CirCor**: el
  límite lo pone la única base pulmonar (el Yunting II), no el
  estetoscopio cardíaco.
  - En el espectro, PASCAL B se parece a CirCor (la regla espectral lo
    manda a cardíaco el 98%, bien).
  - KAUH, aunque se grabó con el mismo Littmann 3200, se parece más a
    ICBHI: casi toda la energía debajo de 100 Hz y casi nada arriba de
    800 Hz (`ltas_littmann3200.png`).
  - El modo de filtro y la cadena de adquisición (el DigiScope) pesan
    tanto como el modelo de estetoscopio.
- **La representación de las tareas generaliza mejor de lo que sugerían
  ICBHI y HLS**:
  - en PASCAL B y KAUH, la CNN6 queda al nivel de su test interno;
  - lo que se cae afuera es el dominio y la calibración de los umbrales.
- **La CNN6 es la mejor en las cuatro bases externas**, con diferencias
  significativas en casi todas las comparaciones.

Figuras actualizadas:

- `figures/dominio.png` y `figures/auc_interno_externo.png`, ahora con
  PASCAL B y KAUH;
- `devices/ltas_littmann3200.png`, nueva;
- `devices/bands.png` y `devices/rule_vs_models.png`, con los grupos
  nuevos.

## 13. Pendiente (a decidir con el usuario)

1. **Robustez del dominio** (lo que más pesa afuera). Opciones, de la más
   barata a la más cara:
   - **(a) Augmentation de equipo**: filtros aleatorios de ecualización y
     de ancho de banda sobre la onda, para que el dominio no se pueda
     decidir por el espectro del equipo. Según §10, sobre todo pasa-altos
     y realces de graves entre 50 y 150 Hz.
   - **(b) Sumar al entrenamiento una base de otro equipo por dominio**:
     por ejemplo, PhysioNet 2016 (cardíaco) o una pulmonar que no sea
     ICBHI, que tiene que seguir siendo externa.
   - **(c) Dejarlo como limitación documentada**, con la recalibración
     de §7. Decidido con el usuario: primero (c), y después se decide si
     se prueba (a) con una regla fijada antes.
2. **Figuras** (hechas): `unified_model/plot_external.py` →
   `reports/external_validation/figures/`:
   - `dominio.png`;
   - `p_cardiaco.png`;
   - `auc_interno_externo.png`;
   - `roc.png`;
   - `icbhi_equipo.png`;
   - `recalibracion.png`.
3. **Bases con el mismo equipo** (§11): hecho (§12).
4. **Memoria**: sección de validación externa en el Cap. 4, y la
   limitación de dominio en las conclusiones.

## 14. Estado del repositorio

Sin commitear al cerrar:

- Código:
  - `common/paths.py`;
  - `preprocessing/make_unified_audio_cache.py`;
  - `unified_model/external_inference.py`;
  - `unified_model/external_validation.py`;
  - `unified_model/recalibrate_external.py`;
  - `unified_model/plot_external.py`;
  - `eda/device_characterization.py`;
  - `preprocessing/make_external_recordings.py`.
- `reports/external_validation/`: `window_probs/` pesa ~19 MB y los
  `predictions_*.csv` ~11 MB en total. Se pueden versionar (permiten
  rehacer las métricas sin correr modelos) o dejar solo las tablas y los
  logs.
- Este documento.

`data/` no se versiona: `data/{hls,kauh,pascal}/` (cachés y tablas) y los
crudos de `data/raw/{KAUH,PASCAL}/`.
