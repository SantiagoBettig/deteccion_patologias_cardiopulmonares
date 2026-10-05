# Modelo cardíaco — de 3 clases a dos tareas binarias independientes

## 1. Primer intento: clasificador único de 3 clases

Primer baseline del modelo cardíaco: `cardiac_model/` (CNN chica, igual bloque
convolucional que el gate, sobre MFCC), 3 clases (`normal_heart`,
`abnormal_heart_unspecified`, `heart_murmur`), ventanas de 4 ciclos
cardíacos (ver `preprocessing/make_cardiac_cycle_windows.py`).

Resultados en test (`reports/cardiac/test_metrics.csv`,
`reports/cardiac/confusion_matrix.png`):

| Clase | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| normal_heart | 0.593 | 0.736 | 0.657 | 2149 |
| abnormal_heart_unspecified | 0.416 | 0.286 | 0.339 | 1410 |
| heart_murmur | 0.735 | 0.632 | 0.680 | 557 |

Accuracy 0.568, F1 macro 0.558.

Matriz de confusión (filas=real, columnas=predicho):

```
                              pred_normal  pred_abnormal  pred_murmur
true normal_heart                  1155           929           65
true abnormal_unspecified            519           848           43
true heart_murmur                    124           131          302
```

## 2. Diagnóstico

Durante el entrenamiento, el recall de validación de `normal_heart` y
`abnormal_heart_unspecified` oscilaba fuerte y de forma errática entre
épocas (ej. una época 0.98/0.04, la siguiente 0.27/0.72), mientras que
`train_loss` bajaba de forma monótona y suave. Se probó bajar el learning
rate 5x (1e-3 → 2e-4, 20 épocas) — la oscilación persistió, con F1 macro
ligeramente peor (0.600 vs. 0.626). Como el `heart_murmur` es
comparativamente estable en ambas corridas, esto no parece ser un problema
de optimización.

La matriz de confusión confirma la hipótesis: la confusión
`normal_heart` ↔ `abnormal_heart_unspecified` es **simétrica** (43% de
`normal_heart` real predicho `abnormal`, 37% de `abnormal_heart_unspecified`
real predicho `normal`), mientras que `heart_murmur` está bastante más
separado de ambas (solo 108 fugas sobre 3559 casos normal+abnormal).

**Causa más probable**: `abnormal_heart_unspecified` = `Outcome=Abnormal`
sin murmullo audible en esa ubicación. Por diseño, estos sujetos *suenan*
igual que `normal_heart` — el único murmullo que distinguiría a un sujeto
abnormal ya se etiqueta aparte como `heart_murmur`. Si el resto de los
diagnósticos anormales de CirCor (el dataset no da más detalle que
`Outcome`) no tienen necesariamente una firma acústica en el sonido
cardíaco (ej. confirmados por eco, no por auscultación), no hay señal de
la que el modelo pueda aprender esa frontera específica.

Esto coincide con cómo CirCor fue pensado originalmente: el reto público
de PhysioNet/CirCor (2022) separa **dos tareas independientes** — detección
de murmullo, y clasificación de outcome clínico — no una clasificación
conjunta de 3 clases.

## 3. Decisión

Se reemplaza el clasificador de 3 clases por **dos modelos binarios
independientes**, sobre las mismas ventanas de 4 ciclos:

- `cardiac_murmur_model/`: murmullo presente/ausente (`pathology_label ==
  "heart_murmur"` ya alcanza para esta etiqueta — es correcta tal como
  está).
- `cardiac_outcome_model/`: outcome normal/abnormal (`Outcome` crudo de
  CirCor). Requiere agregar una columna nueva al dataset unificado —
  `pathology_label` no sirve para esto: colapsa los 29 sujetos con
  murmullo pero `Outcome=Normal` dentro de `heart_murmur`, perdiendo la
  distinción. Se agrega `clinical_outcome` (`normal`/`abnormal`,
  independiente de si el murmullo es audible en esa ubicación) — ver
  `unified_dataset/parsers/parse_circor.py`.

Cada modelo queda en su propia carpeta con su propio `model.py` (mismo
bloque convolucional de partida que el gate, para tener una comparación
limpia — divergen desde ahí según lo que muestre la evidencia, no a
priori), `dataset.py`, `train.py` y `evaluate.py` — no comparten código
entre sí, para poder ajustar la arquitectura/hiperparámetros de cada uno
sin afectar al otro.

`cardiac_model/` (el intento de 3 clases) se deja tal cual, como evidencia
del experimento — no se borra ni se reemplaza en el lugar.

Se agrega también `clinical_outcome` (normal/abnormal) al dataset unificado
(`unified_dataset/parsers/parse_circor.py`, `config.py`) — el Outcome
crudo de CirCor, constante por sujeto. No se puede derivar de
`pathology_label`: 29 de los 179 sujetos con murmullo tienen
`Outcome=Normal` (murmullo inocente), y `pathology_label` los etiqueta
`heart_murmur` en las ubicaciones donde se oye, perdiendo esa distinción.

## 4. Resultados de los dos modelos binarios (baseline)

Mismo bloque convolucional de partida en los dos (`cardiac_murmur_model/`,
`cardiac_outcome_model/`), MFCC, class weighting, 15 épocas, checkpoint
elegido por mejor F1 de validación:

| Modelo | Accuracy (test) | Precision | Recall | F1 |
|---|---|---|---|---|
| Murmullo (presente/ausente) | 0.927 | 0.913 | 0.508 | 0.653 |
| Outcome (normal/abnormal) | 0.578 | 0.521 | 0.455 | 0.486 |

- **Murmullo**: mucho más estable que el intento de 3 clases (sin la
  oscilación errática). Sí se ve sobreajuste después de la época ~5
  (`reports/cardiac_murmur/training_curves.png`) y una caída importante de
  recall entre validación (0.73 en la época elegida) y test (0.51) — el
  modelo es demasiado conservador (alta precisión, recall bajo) para el
  criterio clínico acordado (priorizar recall). Pendiente: elegir
  checkpoint por recall/F2 en vez de F1, y/o bajar el umbral de decisión
  por debajo de 0.5.
- **Outcome**: accuracy 0.578 con clases casi balanceadas (train:
  9184 abnormal / 9339 normal) — apenas por encima del azar. Confirma con
  fuerza la hipótesis de la sección 2: gran parte de los casos `abnormal`
  sin murmullo no tienen firma acústica aprovechable en el sonido
  cardíaco. No se esperaría que aumentar la capacidad del modelo cambie
  esto de forma sustancial — es una limitación de información disponible,
  no de capacidad — pero queda como línea abierta si se quiere confirmar
  con un experimento.

## 5. Ajustes sobre el baseline

**Murmullo** — el objetivo era subir el recall (criterio clínico: un falso
negativo importa más que un falso positivo), sin tocar la arquitectura
salvo por sobreajuste:

- `pos_weight` de la loss subido 1.4x sobre el valor natural (4.77 → 6.68).
- Selección del mejor checkpoint por **F2** en vez de F1 durante el
  entrenamiento — F2 pesa el recall 4x más que la precisión en la fórmula
  ($F_\beta = (1+\beta^2)\frac{P \cdot R}{\beta^2 P + R}$), así que el
  entrenamiento prefiere épocas con más recall aunque bajen algo de
  precisión.
- Dropout (p=0.3) antes de la capa final, por el sobreajuste visible
  después de la época 5 en el baseline.
- Umbral de decisión calibrado sobre validación (buscando el máximo F2 en
  una grilla), en vez de 0.5 fijo — calibración pura sobre el checkpoint ya
  entrenado, sin reentrenar.

Resultado en test:

| Configuración | Umbral | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| Baseline (F1, umbral 0.5) | 0.50 | 0.927 | 0.913 | 0.508 | 0.653 | — |
| +pos_weight+F2+dropout, umbral 0.5 | 0.50 | 0.903 | 0.624 | 0.704 | 0.662 | 0.686 |
| +pos_weight+F2+dropout, umbral calibrado | 0.45 | 0.895 | 0.589 | **0.749** | 0.659 | 0.710 |

Recall subió de 0.51 a 0.75 (casi se duplicó), a costa de la precisión
(0.91 → 0.59) — el trade-off esperado y buscado. F1 se mantiene
prácticamente igual (0.65-0.66 en las tres configuraciones) porque F1 no
distingue esta mejora — es exactamente por lo que se optimizó F2 en vez de
F1 para elegir el checkpoint.

**Outcome** — se agregó un 4to bloque convolucional (64→128 canales, ver
`cardiac_outcome_model/model.py`) y se subieron las épocas de 15 a 30
(el baseline no había convergido: train_loss seguía bajando en la última
época).

Resultado: con la capacidad extra, el modelo **sí llega a memorizar el
train** (train_acc 0.928 en la época 30, vs. 0.727 del baseline con 3
bloques y 15 épocas) — pero `val_loss` diverge desde la época ~17 en
adelante (0.66 → 1.44) y el F1 de validación no supera ~0.65 en ninguna
época, prácticamente el mismo techo que el modelo más chico. En test:

| Configuración | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (3 bloques, 15 épocas) | 0.578 | 0.521 | 0.455 | 0.486 |
| 4 bloques, 30 épocas | 0.504 | 0.453 | 0.635 | 0.529 |

Accuracy 0.504 — exactamente en el nivel del azar (clases ~50/50 en
train). El F1 algo más alto es solo un corrimiento del punto
precisión/recall, no una mejora real de separabilidad.

**Conclusión (outcome)**: la capacidad no es el techo — más profundidad
solo le dio al modelo más lugar para sobreajustar (memoriza train, no
generaliza), sin mover la aguja en test. Esto respalda con bastante fuerza
la hipótesis de la sección 2: para una porción sustancial de los casos
`abnormal` de CirCor, no hay señal acústica aprovechable en el sonido
cardíaco (el `Outcome` probablemente viene de hallazgos no auscultables).
Se documenta como limitación conocida de esta tarea con este dataset, no
se sigue iterando la arquitectura — otras vías posibles a futuro (fuera de
alcance por ahora): usar features distintas a MFCC que capturen algo no
espectral (ej. variabilidad de intervalos, ver
`avances/8_gate_v6_periodicity.md` para el precedente de por qué esa vía
tampoco fue trivial), o aceptar que esta tarea específica no es resoluble
solo con fonocardiograma.

## 6. Murmullo — mel spectrogram vs. MFCC

Antes de seguir iterando el modelo de murmullo, se comparó log-Mel (v2)
contra MFCC (v1, sección 5) — misma receta en todo lo demás (pos_weight
x1.4, checkpoint por F2, dropout, umbral calibrado sobre validación).
Motivación: a diferencia del gate (donde MFCC ganó por robustez entre
dominios distintos, CirCor vs. HLS-CMDS), acá no hay ese problema de
dominio cruzado, y un murmullo tiene contenido de alta frecuencia que MFCC
podría estar recortando al comprimir a 20 coeficientes.

v1 (MFCC) y v2 (log-Mel) quedan preservados en carpetas separadas —
`cardiac_murmur_model/checkpoints_v1_mfcc/` /
`reports/cardiac_murmur/runs/v1_mfcc/` y
`checkpoints_v2_logmel/` / `reports/cardiac_murmur/runs/v2_logmel/` — antes
de seguir experimentando, para poder citar ambos.

| Versión | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| v1 — MFCC | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** |
| v2 — log-Mel | 0.60 | 0.881 | 0.542 | 0.756 | 0.631 | 0.701 |

Prácticamente empatan — log-Mel da 0.7 puntos más de recall, pero pierde en
precisión, F1 y F2 (nuestro criterio principal). La hipótesis de que
log-Mel retendría información útil que MFCC recorta no se confirmó en la
práctica. **Se sigue con MFCC (v1)** como base para las próximas
iteraciones.

## 7. Murmullo — learning rate scheduler

Motivación: en v1/v2, `val_loss` mostraba picos fuertes en varias épocas
(ej. 1.75, 1.53 en v1) — se probó si un `ReduceLROnPlateau` sobre `val_f2`
(baja el LR cuando F2 deja de mejorar 2 épocas seguidas) amortiguaba ese
ruido. v3 = MFCC (igual que v1) + scheduler — ver
`cardiac_murmur_model/run_v3_lr_scheduler.py`, preservado en
`checkpoints_v3_lr_scheduler/` / `reports/cardiac_murmur/runs/v3_lr_scheduler/`.

| Versión | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| v1 — MFCC, sin scheduler | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** |
| v3 — MFCC + LR scheduler | 0.40 | 0.852 | 0.471 | 0.750 | 0.579 | 0.671 |

El LR bajó de 1e-3 a 1.25e-4 a lo largo de las 15 épocas, pero `val_loss`
siguió con picos igual de fuertes en varias épocas — la hipótesis de que
el ruido venía de la magnitud del LR no se sostiene. v3 queda claramente
peor que v1 (mismo recall, pero mucho peor precisión, F1 y F2). **Se
descarta el scheduler y se mantiene v1** como la mejor versión del modelo
de murmullo hasta ahora.

## 8. Murmullo — metadata del paciente (v4)

Se agrega sexo, edad, altura y peso como entrada adicional al modelo de
murmullo, fusionada con el embedding de audio antes de la capa final
("late fusion" — ver `cardiac_murmur_model/model.py:MurmurCNNMultimodal`),
sobre la base de v1 (MFCC, sigue siendo la mejor versión, secciones 6-7).

**Análisis previo** (antes de entrenar nada): correlación cruda entre
metadata y murmullo en el dataset — proporción de murmullo por edad casi
plana (adolescente 17.8%, child 16.3%, infant 18.2%), por sexo casi
idéntica (F 15.5%, M 16.1%). Altura/peso muestran una diferencia modesta
(111.9cm/22.9kg en murmullo vs. 115.6cm/25.3kg sin murmullo) probablemente
confundida con edad. Conclusión: no se espera una mejora grande, pero
tampoco hay mucho riesgo de que el modelo aprenda un atajo demográfico en
vez de escuchar el audio — la correlación cruda es débil.

**Chequeo de atajo** (mismo tipo que `gate_model/diagnostics_source_db.py`,
que detectó el sesgo por `source_db` en el gate v1): antes de confiar en
cualquier mejora del modelo multimodal, se entrena un clasificador que ve
**solo** la metadata (sin audio) — regresión logística sobre el mismo
vector de metadata, ver `cardiac_murmur_model/diagnostics_metadata_only.py`.

**Problema de datos encontrado en el camino**: `Weight` del sujeto 84784 en
el CSV crudo de CirCor viene corrupto
(`"28.160.000.000.000.000"`, inconsistente además con su altura de 47cm) —
eso hacía que pandas infiriera la columna `weight_kg` completa como texto
en vez de numérica en el dataset unificado, rompiendo cualquier cálculo de
media/std. Corregido en `unified_dataset/parsers/parse_circor.py`: se
fuerza `Height`/`Weight` a numérico al leer el CSV crudo, los valores no
parseables quedan `NaN` (ya se manejan como faltantes). Pipeline
re-corrido (`build_dataset.py` → `make_cardiac_splits.py` →
`make_cardiac_cycle_windows.py`) — no afecta ninguna clase ni split, solo
agrega ~2 NaN nuevos en `weight_kg`.

## 9. Murmullo — agregación de predicciones a nivel sujeto

Idea del usuario: en vez de evaluar ventana por ventana, agregar las
probabilidades de TODAS las ventanas de un mismo sujeto (todas sus
ubicaciones y ciclos) antes de decidir — más parecido a cómo se
diagnostica en la práctica (varios puntos de auscultación, no un clip de
2s), y potencialmente más robusto al ruido de una ventana individual. No
requiere reentrenar — es una forma distinta de usar las probabilidades ya
calculadas por el checkpoint de v1 (ver
`cardiac_murmur_model/evaluate_subject_level.py`).

Ground truth por sujeto: 1 si tiene al menos una ventana `heart_murmur`
(mismo criterio que `splits/split_utils.py:pick_subject_pathology`).
Agregación probada de dos formas — máximo (si CUALQUIER ventana sugiere
murmullo, se marca al sujeto) y media:

| Agregación | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| Ventana (v1, referencia) | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | 0.710 |
| Sujeto — máximo | 0.85 | 0.887 | 0.704 | 0.704 | **0.704** | 0.704 |
| Sujeto — media | 0.50 | 0.909 | 0.889 | 0.593 | 0.711 | 0.635 |

La agregación por **máximo** da el mejor balance: F1 sube de 0.659 a 0.704
(la precisión mejora mucho, 0.589→0.704, con recall prácticamente igual),
y F2 se mantiene equivalente al de ventana individual (0.704 vs 0.710) —
confirma que buena parte del "ruido" de precisión a nivel ventana se debe
a falsos positivos aislados que se filtran al mirar el conjunto de
ventanas de un sujeto. La agregación por media da mejor precisión pero
recall mucho más bajo — no conviene dado el criterio clínico. **Se adopta
la agregación por máximo como la forma de reportar/usar el modelo en la
práctica**, sobre cualquier checkpoint que termine siendo el mejor.

**Resultado del diagnóstico** (`diagnostics_metadata_only.py`, regresión
logística solo-metadata): accuracy 0.482 en test — **muy por debajo** de
la referencia trivial de predecir siempre "sin murmullo" (0.865).
Precision 0.142, básicamente ruido. Confirma la expectativa: la metadata
sola no alcanza para predecir murmullo, bajo riesgo de atajo demográfico.

**Resultado del modelo multimodal (v4)**:

| Versión | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| v1 — MFCC (solo audio) | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** |
| v4 — MFCC + metadata | 0.45 | 0.794 | 0.376 | 0.788 | 0.509 | 0.646 |

v4 es claramente **peor** que v1 en accuracy, precisión, F1 y F2 — el
recall sube apenas (0.788 vs. 0.749), no compensa el resto. Consistente
con el diagnóstico: como la metadata casi no tiene señal propia, agregarla
como entrada no aporta información real y solo suma parámetros/ruido a un
modelo que ya entrena de forma inestable (`val_loss` con picos, ver
secciones 5-7) — empeora en vez de ayudar.

## 10. Murmullo — focal loss (v5)

Motivación: en v1-v4, `val_loss`/`val_f2` mostraban oscilaciones fuertes
entre épocas. FocalLoss (ver `cardiac_murmur_model/losses.py`,
alpha=0.75, gamma=2.0) concentra el gradiente en los ejemplos difíciles en
vez de escalar uniformemente los positivos (lo que hace `pos_weight`) —
se probó si esto estabilizaba el entrenamiento y/o mejoraba el resultado,
sobre la receta de v1 (MFCC, dropout, sin scheduler).

| Versión | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| v1 — BCE + pos_weight | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** |
| v5 — FocalLoss | 0.50 | 0.895 | 0.600 | 0.675 | 0.635 | 0.659 |

v5 queda por debajo de v1 en recall, F1 y F2 (precisión casi igual). La
oscilación de `val_f2` entre épocas tampoco se redujo (0.72→0.84→0.67→
0.80→0.62 en las últimas 5 épocas) — FocalLoss no resuelve la
inestabilidad ni mejora el resultado. **Se descarta, se mantiene v1**.

## 11. Murmullo — ubicación de auscultación como feature (v6)

A diferencia de la metadata demográfica (v4, sección 8), la ubicación
(`auscultation_location`, one-hot de 5 categorías — ver
`cardiac_murmur_model/dataset.py:encode_location`) está mecánicamente
ligada a si un murmullo se escucha, así que se esperaba mejor resultado
que v4. Misma arquitectura de fusión que v4 (`MurmurCNNMultimodal`), sobre
la receta de v1.

| Versión | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| v1 — solo audio | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** |
| v6 — audio + ubicación | 0.35 | 0.843 | 0.451 | 0.740 | 0.561 | 0.656 |

Igual que v4, v6 queda por debajo de v1 en todo menos en recall (0.740 vs.
0.749, prácticamente empatado) — ni siquiera con una feature bien
fundamentada la fusión multimodal mejora sobre el audio solo. **Se
descarta, se mantiene v1**.

**Patrón que se repite en v3/v4/v5/v6**: agregar cualquier cosa a la
receta base de v1 (scheduler, metadata, focal loss, ubicación) empeoró el
resultado o no cambió nada. Lo único que mejoró sobre v1 hasta ahora fue
**calibrar cómo se usa el modelo ya entrenado** (umbral, agregación por
sujeto — secciones 5 y 9), no cambiar el modelo en sí. Posible explicación:
con ~179 sujetos con murmullo, cualquier rama/parámetro nuevo compite por
gradiente con la señal de audio (que ya es débil por el volumen de datos)
en vez de sumar.

## 12. Murmullo — ventanas de train más densas (v7)

Última idea de la tanda: en vez de agregar una feature o cambiar la loss,
generar más ventanas de entrenamiento por grabación — `TRAIN_CYCLE_HOP=1`
ciclo en vez de 2 (más solapamiento, solo en train; val/test quedan
idénticas), ver `preprocessing/make_cardiac_cycle_windows_dense_train.py`.
Train pasó de 18523 a 35963 ventanas (murmullo: 3210→6246). Resto de la
receta igual a v1 (MFCC, pos_weight×1.4, checkpoint por F2, dropout).

| Versión | Umbral calibrado | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| v1 — hop=2 en train (igual que val/test) | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** |
| v7 — hop=1 en train | 0.40 | 0.854 | 0.476 | 0.752 | 0.583 | 0.674 |

Tampoco mejora — recall prácticamente empatado (0.752 vs. 0.749), pero
peor en todo lo demás. Confirma lo que se anticipó al proponer la idea:
las ventanas extra son recortes muy correlacionados del mismo puñado de
~179 sujetos, no información nueva — no alcanzan para mover la aguja.
**Se descarta, se mantiene v1.**

## 14. Validación k-fold de la receta de v1

Antes de cerrar, se corrió una validación k-fold (K=5, estratificado por
sujeto — ver `splits/make_cardiac_murmur_kfold.py`) sobre la receta de v1,
midiendo tanto a nivel ventana como a nivel sujeto (agregación por
máximo, sección 9) en cada fold — ver `cardiac_murmur_model/run_kfold.py`.
Motivación: con un solo split, test tenía solo ~27-36 sujetos con
murmullo; una diferencia entre versiones (o entre formas de evaluar)
podía deberse a qué sujetos tocaron en esa partición particular, no a un
efecto real.

Cada uno de los ~179 sujetos con murmullo pasó por el rol de test
exactamente una vez a lo largo de los 5 folds. Resultado (media ± desvío
entre folds):

| Nivel | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|
| Ventana (k-fold) | 0.869 ± 0.023 | 0.606 ± 0.062 | 0.782 ± 0.026 | 0.681 ± 0.039 | **0.737 ± 0.022** |
| Ventana (v1, split único, referencia) | 0.895 | 0.589 | 0.749 | 0.659 | 0.710 |
| Sujeto — máximo (k-fold) | 0.492 ± 0.079 | 0.268 ± 0.034 | 0.938 ± 0.024 | 0.416 ± 0.041 | 0.623 ± 0.036 |
| Sujeto — máximo (split único, referencia, sección 9) | 0.887 | 0.704 | 0.704 | 0.704 | 0.704 |

**Hallazgo importante — la ventana se sostiene, la agregación por sujeto no**:

- El resultado **a nivel ventana** del k-fold (F2=0.737±0.022) es
  consistente con el del split único de v1 (F2=0.710, dentro de 1 desvío)
  — la performance de v1 es reproducible, no fue suerte del split.
- El resultado **a nivel sujeto** del k-fold es muy distinto al del split
  único: F1 cae de 0.704 a **0.416** (precisión 0.704→0.268), muchísimo
  peor que a nivel ventana en vez de mejor. La mejora que se había visto
  con la agregación por máximo (sección 9) **no se reprodujo** — fue en
  buena parte un efecto favorable de esa partición particular, exactamente
  el tipo de espejismo que el k-fold está pensado para detectar. Con ~28
  ventanas por sujeto en promedio, agregar por máximo acumula falsos
  positivos ("si alguna de 28 ventanas se pasa del umbral, se marca al
  sujeto") en vez de cancelar ruido — empeora cuando el clasificador de
  ventana es imperfecto, no lo compensa.

## 15. Conclusión final — modelo de murmullo

De siete versiones de modelo probadas (v1-v7), **v1 (MFCC + pos_weight×1.4
+ checkpoint por F2 + dropout + umbral calibrado) sigue siendo la mejor**,
y el k-fold confirma que su performance a nivel ventana es reproducible:
**F2 ≈ 0.71-0.74, recall ≈ 0.75-0.78**. Ninguna modificación al modelo o
al entrenamiento (log-Mel, LR scheduler, metadata, focal loss, ubicación,
más datos de train) la superó.

Se **revierte la recomendación de la sección 9**: la agregación de
predicciones por sujeto (máximo) no se sostiene bajo k-fold — se reporta
como hallazgo negativo, no como mejora. **Se cierra esta línea de trabajo
con v1, evaluado a nivel ventana, como versión final** — sin agregación
por sujeto salvo que se investigue una forma de agregar que no amplifique
falsos positivos (ej. votación por mayoría en vez de máximo, o un umbral
recalibrado específicamente para la métrica agregada — quedan como líneas
abiertas, no se investigó más en esta etapa).
