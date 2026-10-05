# Proyecto Final CEIA — Modelo gate v1: baseline y sesgo detectado

## 1. Qué se implementó

Pipeline completo de la etapa 1 (gate cardíaco/pulmonar), primera de las dos
etapas del diseño "pipeline de 3 modelos" (ver `3_preliminar_eda.md`,
sección 4):

- Split train/val/test por `subject_id`, estratificado por `pathology_label`
  (`splits/make_splits.py`).
- Preprocesamiento de audio compartido: filtrado pasa-bajos (~2048 Hz) →
  resample a 4000 Hz → normalización RMS (`preprocessing/audio_ops.py`).
- Ventaneo fijo (5 s, 50% overlap) sobre CirCor + ICBHI para generar el
  dataset de entrenamiento del gate (`preprocessing/make_gate_windows.py`).
- CNN 2D pequeña (3 bloques conv+pool + global average pooling, 23.585
  parámetros) sobre espectrogramas log-Mel, entrenada con PyTorch
  (`gate_model/model.py`, `train.py`).
- Validación externa con HLS-CMDS: mismo preprocesamiento y ventaneo
  aplicado a un dataset que el modelo nunca vio en entrenamiento
  (`preprocessing/make_hls_windows.py`, `gate_model/evaluate_external_hls.py`).
  Motivo: en el dataset de entrenamiento, `sound_type` está perfectamente
  correlacionado con `source_db` (100% cardíaco = CirCor, 100% pulmonar =
  ICBHI), así que un resultado de test alto no alcanza para confiar en que
  el modelo generalice — HLS-CMDS es el único dataset con ambos tipos de
  sonido grabados con el mismo equipo/entorno.

Artefactos de esta corrida (checkpoint + reportes) guardados en
`desarrollo/reports/gate/runs/v1_baseline_sin_mitigacion/` para poder
comparar contra las siguientes iteraciones.

## 2. Resultados

| Métrica | Test (CirCor + ICBHI) | HLS-CMDS (validación externa) |
|---|---|---|
| Accuracy | 0.9998 | 0.545 |
| Precision | 0.9998 | 0.708 |
| Recall | 1.0000 | 0.153 |
| F1 | 0.9999 | 0.252 |

Matrices de confusión:

**Test** — pulmonary: 992/993 correctos · cardiac: 4199/4199 correctos
(prácticamente perfecto).

**HLS-CMDS** — pulmonary: 281/300 correctos (93.7%) · **cardiac: solo
46/300 correctos (15.3%)**, 254 casos clasificados como pulmonares.

El entrenamiento además convergió a >99% de accuracy de validación desde la
época 2-3 (`training_curves.png`), una velocidad de convergencia atípica
para una tarea genuinamente difícil.

## 3. Análisis — hipótesis de shortcut learning

El resultado de test (~100%) combinado con el fracaso marcado y asimétrico
en HLS-CMDS (falla sistemáticamente en cardíaco, no en pulmonar) sugiere que
el modelo no aprendió a distinguir sonido cardíaco de pulmonar por su
contenido acústico, sino que explota el confound `sound_type` ≈ `source_db`
del dataset de entrenamiento — probablemente algún artefacto remanente del
proceso de resample (más marcado cuanto más agresivo fue el downsampling
original).

Detalle que apoya esta hipótesis: HLS-CMDS es nativo a 22.050 Hz, más
cercano al rango nativo de ICBHI (10k/44.1k Hz, resampleado fuerte) que al
de CirCor (ya nativo a ~4000 Hz, resample casi nulo). Esto podría explicar
por qué el modelo "ve" a HLS más parecido a ICBHI (pulmonar) sin importar la
clase real.

> ⚠️ **Corrección (ver sección 7)**: se verificó con `soundfile` que
> HLS-CMDS es en realidad nativo a **4000 Hz**, no a 22.050 Hz — el dato
> original venía de la documentación del dataset, no de una medición
> directa. Esta hipótesis del sample rate queda **descartada**; el EDA de
> la sección 7 identifica una causa más concreta y medible.

## 4. Diagnóstico cuantitativo — confirmado

Se entrenó la misma arquitectura (`GateCNN`), sobre las mismas ventanas del
gate, prediciendo `source_db` (CirCor vs. ICBHI) en vez de `sound_type`
(`gate_model/diagnostics_source_db.py`).

| Métrica | Diagnóstico source_db (test) |
|---|---|
| Accuracy | 0.9994 |
| Precision | 1.0000 |
| Recall | 0.9993 |
| F1 | 0.9996 |

Resultado prácticamente idéntico al del gate real (0.9998 accuracy), con la
misma curva de convergencia casi instantánea (>99.8% ya en la época 2 —
ver `reports/gate_diagnostics/source_db/training_curves.png`). Esto
**confirma cuantitativamente** la hipótesis de la sección 3: la red separa
el origen del dataset con la misma facilidad con la que "resolvía"
cardíaco/pulmonar, evidencia directa de que el gate v1 aprendió el atajo
`source_db` y no la diferencia acústica real.

## 5. Mitigación — gate v2

Cambios respecto a v1, en `gate_model/dataset.py` (ver también
`GateDataset.__init__`/`_augment_waveform`/`_normalize`):

- **Normalización por espectrograma (z-score)**: se aplica siempre —
  train/val/test y también en la validación externa con HLS-CMDS — resta
  la media y divide por el desvío de cada espectrograma log-Mel
  individual, para reducir diferencias de nivel/offset atribuibles al
  dataset de origen.
- **Data augmentation (solo en train)**: por cada muestra, ganancia
  aleatoria (`gain_db_range=(-3.0, 3.0)` dB) + ruido gaussiano aditivo
  (`noise_std=0.005`) sobre la forma de onda, antes de calcular el
  espectrograma. No se aplica en val/test/HLS (la evaluación debe ser
  determinística).
- Sin cambios en arquitectura (`GateCNN`), hiperparámetros de
  entrenamiento (15 épocas, Adam, `lr=1e-3`, `batch_size=32`) ni en el
  split o las ventanas de audio — así el efecto medido es atribuible solo
  a estos dos cambios.

Artefactos de esta corrida (checkpoint + reportes) guardados en
`desarrollo/reports/gate/runs/v2_normalizacion_augmentation/`.

### Resultados — v1 (baseline) vs. v2 (normalización + augmentation)

| Métrica | Test v1 | Test v2 | HLS-CMDS v1 | HLS-CMDS v2 |
|---|---|---|---|---|
| Accuracy | 0.9998 | 0.9961 | 0.545 | **0.818** |
| Precision | 0.9998 | 0.9953 | 0.708 | **0.852** |
| Recall | 1.0000 | 1.0000 | 0.153 | **0.770** |
| F1 | 0.9999 | 0.9976 | 0.252 | **0.809** |

Matriz de confusión HLS-CMDS v2: pulmonary 260/300 correctos (86.7%),
**cardiac 231/300 correctos (77.0%)** — antes 46/300 (15.3%). El error dejó
de ser un sesgo sistemático hacia "pulmonar" y pasó a un error más
parejo entre clases.

El accuracy de test bajó levemente (0.9998 → 0.9961), esperable: el modelo
ya no puede apoyarse tan fácilmente en el atajo de origen de dataset, así
que la tarea en test también se volvió (correctamente) un poco más difícil.
La curva de val loss en v2 es más ruidosa época a época
(`training_curves.png`) — consistente con estar entrenando sobre una señal
más genuina y menos con un atajo trivial.

**Conclusión**: la mitigación funcionó — mejora sustancial y verificable en
el dataset nunca visto, no solo en el propio held-out del mismo origen.
Sigue habiendo un ~20-23% de error por clase en HLS-CMDS, con margen de
mejora (ver sección 6).

## 6. Mitigación — gate v3 (resultado negativo)

Sobre v2, en `gate_model/dataset.py`:

- **SpecAugment**: 2 máscaras de frecuencia (ancho aleatorio hasta 8 bandas
  de 64) + 2 máscaras de tiempo (ancho aleatorio hasta 15 frames de 126),
  aplicadas sobre el espectrograma ya normalizado (se enmascara con 0 =
  la media tras z-score). Solo en train.
- **Augmentation de forma de onda más fuerte**: `gain_db_range` de
  `(-3, 3)` a `(-6, 6)` dB, `noise_std` de `0.005` a `0.01`.

Artefactos en `desarrollo/reports/gate/runs/v3_specaugment/`.

| Métrica | Test v2 | Test v3 | HLS-CMDS v2 | HLS-CMDS v3 |
|---|---|---|---|---|
| Accuracy | 0.9961 | 0.9857 | 0.818 | 0.793 |
| Precision | 0.9953 | 0.9947 | 0.852 | **0.994** |
| Recall | 1.0000 | 0.9876 | 0.770 | 0.590 |
| F1 | 0.9976 | 0.9912 | 0.809 | 0.741 |

**Resultado negativo**: v3 empeoró tanto en test como en HLS-CMDS respecto
a v2 (F1 en HLS: 0.809 → 0.741, recall cardíaco: 77.0% → 59.0%). También
tardó más en entrenar. La precisión en HLS subió mucho (0.994) pero a costa
de un recall bajo — el modelo se volvió más conservador prediciendo
"cardíaco", probablemente porque el enmascarado de tiempo/frecuencia,
combinado con una señal cardíaca ya de por sí más sutil que la del atajo
original, resultó demasiado agresivo para la capacidad del modelo (23.585
parámetros). Se descarta este camino; queda documentado como resultado
negativo (información igual de válida para la memoria técnica).

## 7. EDA comparativo de fuentes — hallazgo real

A pedido del usuario, se investigó (a) si había una diferencia concreta y
medible en los datos crudos de HLS-CMDS que explicara el fallo, y (b) si el
desbalance de clases (cardíaco:pulmonar ≈ 3.5:1 en las ventanas de
entrenamiento — 27.361 vs. 7.857) podía ser la causa.
(`preprocessing/compare_sources_eda.py`, resultados en
`reports/preprocessing/source_comparison_stats.csv` y
`source_comparison_boxplots.png`).

**Sample rate nativo de HLS-CMDS**: verificado con `soundfile.info()`
directamente sobre los WAV — **4000 Hz**, no 22.050 Hz. Descarta la
hipótesis de la sección 3.

**Centroide espectral (brillo del sonido), cardíaco vs. pulmonar dentro de
cada fuente** — ver `reports/preprocessing/spectral_centroid_inversion.png`:

| Fuente | Cardíaco | Pulmonar | Relación |
|---|---|---|---|
| CirCor + ICBHI (entrenamiento) | 407 Hz | 119 Hz | cardíaco mucho más brillante |
| HLS-CMDS | 208 Hz | 262 Hz | **pulmonar más brillante — invertido** |

El centroide espectral es la diferencia más grande y fácil de explotar
entre clases en el dataset de entrenamiento. En HLS-CMDS esa relación está
**invertida**: un cardíaco de HLS es espectralmente más parecido a un
pulmonar de CirCor/ICBHI que a un cardíaco de CirCor. Si el modelo se apoya
en el brillo espectral como señal principal, esto explica directamente el
patrón de error observado (cardíaco de HLS confundido con pulmonar) en
v1-v3.

**Hipótesis de fondo**: HLS-CMDS se graba sobre un maniquí clínico
(reproducción mecánica/sintética), no un paciente real. La función de
transferencia acústica del maniquí podría diferir de la de un tórax humano
real lo suficiente como para alterar la relación espectral natural entre
sonido cardíaco y pulmonar. De ser así, parte de la brecha con HLS-CMDS
sería una **brecha de dominio genuina** (sintético vs. real), no solo un
artefacto de preprocesamiento — limitación a documentar honestamente en el
informe final, no necesariamente resoluble por completo con más
regularización.

**Sobre el desbalance de clases**: confirmado (3.5:1 cardíaco:pulmonar),
pero **probablemente no es la causa principal** del fallo con HLS-CMDS: un
desbalance sesgaría al modelo hacia la clase mayoritaria (cardíaco) al
fallar, y se observa justo lo contrario (sesga hacia pulmonar, la
minoritaria) en v1 y v3. Sigue siendo una corrección válida por buenas
prácticas, independiente del problema de generalización con HLS.

## 8. Mitigación — gate v4 (class weighting)

Sobre v2 (se revierte el augmentation de v3 — SpecAugment y ruido/ganancia
más fuertes descartados), en `gate_model/train.py`:

- `pos_weight` en `BCEWithLogitsLoss`, calculado del split de train real:
  19.031 positivos (cardiac) / 5.411 negativos (pulmonary) →
  `pos_weight = 5411/19031 ≈ 0.284`. Downweightea la clase mayoritaria para
  que la loss no la favorezca.
- Sin cambios en arquitectura, augmentation (vuelve a `gain_db_range=(-3,3)`,
  `noise_std=0.005`, sin SpecAugment) ni en split/ventanas — aísla el
  efecto del class weighting del resto de las variables.

Artefactos en `desarrollo/reports/gate/runs/v4_class_weighting/`.

### Comparación completa v1 → v4

| Métrica | v1 (baseline) | v2 (norm+aug) | v3 (SpecAugment) | v4 (class weighting) |
|---|---|---|---|---|
| Test accuracy | 0.9998 | 0.9961 | 0.9857 | 0.9931 |
| Test F1 | 0.9999 | 0.9976 | 0.9912 | 0.9957 |
| HLS accuracy | 0.545 | 0.818 | 0.793 | **0.828** |
| HLS precision | 0.708 | 0.852 | 0.994 | 0.880 |
| HLS recall | 0.153 | 0.770 | 0.590 | 0.760 |
| HLS F1 | 0.252 | 0.809 | 0.741 | **0.816** |

**Conclusión**: v4 es prácticamente equivalente a v2 (F1 en HLS: 0.809 vs.
0.816, diferencia dentro del ruido esperable entre corridas). Esto confirma
la lectura de la sección 7: el desbalance de clases **no era la causa
principal** del problema de generalización con HLS-CMDS — corregirlo no
perjudica pero tampoco resuelve la brecha. La causa más probable sigue
siendo la inversión del centroide espectral documentada en la sección 7,
que podría ser una diferencia de dominio genuina (maniquí vs. paciente
real) más difícil de cerrar solo con regularización o balanceo.

**v4 queda como versión vigente del gate** (mejor F1 en HLS de las cuatro,
aunque por un margen mínimo sobre v2) — checkpoint en
`gate_model/checkpoints/best.pt`.

## 9. Próximos pasos

- Evaluar agregación por grabación completa (voto mayoritario entre las
  ventanas de un mismo `file_id`) en vez de accuracy por ventana individual
  — más representativo del caso de uso real y ya previsto en el charter
  (evaluación "por etapa y end-to-end"). No requiere reentrenar.
- Si se busca cerrar más la brecha con HLS-CMDS: explorar features menos
  sensibles a la brecha de dominio maniquí/paciente real (ej. MFCC en vez
  de espectrograma Mel crudo, que comprime la envolvente espectral y podría
  ser más robusto a la inversión de brillo encontrada), o aceptar la
  brecha como limitación documentada y priorizar avanzar a la etapa 2.
- Una vez conforme con el gate, avanzar a la etapa 2 (modelos de patología
  cardíaca y pulmonar con segmentación por ciclo).
