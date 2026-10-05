# Proyecto Final CEIA — Modelo gate v5: MFCC en vez de espectrograma log-Mel

## 1. Motivación

`4_gate_model_baseline_y_sesgo.md` (sección 7) encontró que la causa más
probable del bajo desempeño del gate v4 en HLS-CMDS (F1 0.816, frente a
0.996 en test propio) es una inversión del centroide espectral (brillo)
cardíaco/pulmonar entre el dataset de entrenamiento (CirCor+ICBHI) y
HLS-CMDS — posiblemente una brecha de dominio real (maniquí vs. paciente).

La sección 9 de ese mismo documento propone, como siguiente paso, probar
MFCC en vez del espectrograma log-Mel crudo: la DCT final de MFCC comprime
la envolvente espectral en pocos coeficientes, lo que en teoría podría
hacerlo menos sensible a esa inversión de brillo que un espectrograma
log-Mel completo (que preserva el detalle banda por banda).

Nota: esto es una hipótesis distinta a la discutida y descartada en
`5_intento_google_colab.md` (sección 5) — ahí se descartó MFCC por motivos
de **cómputo** (no ahorra tiempo, calcula el mismo log-Mel + DCT). Acá el
objetivo es **generalización/robustez**, no velocidad.

## 2. Qué se implementó

Gate v5 construido **sobre v4** — se mantiene todo lo que ya funciona
(normalización z-score por instancia, augmentation de audio sobre la forma
de onda, sin SpecAugment (descartado en v3), class weighting, split,
arquitectura `GateCNN`, hiperparámetros de entrenamiento) y se cambia
únicamente la extracción de features:

- `gate_model/dataset.py` (`GateDataset`): nuevo parámetro `feature_type`
  (`"logmel"` default | `"mfcc"`) y `n_mfcc` (default 20). Con
  `feature_type="mfcc"`, se calcula `librosa.feature.mfcc(...)` en vez de
  `melspectrogram` + `power_to_db`, reutilizando los mismos `n_mels`/
  `n_fft`/`hop_length` como parámetros del banco de filtros Mel interno. El
  resto del pipeline (z-score, augmentation) opera igual sobre el resultado.
- `gate_model/train.py`, `evaluate.py`, `evaluate_external_hls.py`: aceptan
  `feature_type`/`n_mfcc` (default `"logmel"`, no cambia el comportamiento
  de v1-v4 si se corren sin argumentos).
- `gate_model/model.py`: sin cambios — `GateCNN` usa
  `AdaptiveAvgPool2d`, no depende de la cantidad de filas de entrada (64
  bandas Mel o 20 coeficientes MFCC).
- Elección de `n_mfcc=20` y sin coeficientes delta/delta-delta, para
  mantener la comparación acotada a MFCC vs. log-Mel a igualdad de todo lo
  demás.
- Nuevo script `gate_model/run_v5_mfcc.py` (wrapper de
  `gate_model/run_pipeline.py`, ver sección 3) que corre entrenamiento +
  evaluación + validación externa con `feature_type="mfcc"`, guardando en
  carpetas separadas (`gate_model/checkpoints_v5_mfcc/`,
  `reports/gate/runs/v5_mfcc/`) para no pisar el checkpoint vigente de v4.

## 3. Extra — `run_pipeline.py`

A pedido del usuario, además se agregó `gate_model/run_pipeline.py`: corre
`train()` + `evaluate()` + `evaluate_external_hls()` en una sola llamada,
para poder lanzar el entrenamiento y dejar la PC corriendo sin tener que
estar pendiente de cuándo termina cada paso para lanzar el siguiente a
mano. Con los defaults (`feature_type="logmel"`) reemplaza correr a mano
los pasos 5-7 del README para el gate vigente; `run_v5_mfcc.py` es un
wrapper de una línea sobre esta misma función.

## 4. Resultados

| Métrica | Test v4 | Test v5 (MFCC) | HLS-CMDS v4 | HLS-CMDS v5 (MFCC) |
|---|---|---|---|---|
| Accuracy | 0.9931 | 0.9753 | 0.828 | **0.867** |
| Precision | 0.9950 | 0.9825 | 0.880 | **0.991** |
| Recall | 0.9964 | 0.9871 | 0.760 | 0.740 |
| F1 | 0.9957 | 0.9848 | 0.816 | **0.847** |

Matriz de confusión HLS-CMDS v5 (300 ventanas por clase, inferida de
precision/recall): pulmonary ≈298/300 correctas (99.3%), **cardiac
222/300 correctas (74.0%)** — 78 ventanas cardíacas clasificadas como
pulmonares. Comparado con v4 (cardiac 228/300, 76.0%), el recall de
cardíaco **no mejoró — bajó levemente** (76.0% → 74.0%); la mejora de F1 se
explica por una precisión mucho más alta (menos pulmonares mal
clasificados como cardíacos), no por resolver la debilidad en cardíaco.

## 5. Conclusión

MFCC mejora el F1 global en HLS-CMDS (0.816 → 0.847) pero **no resuelve el
problema central**: la clase cardíaca sigue siendo el punto débil, con
prácticamente el mismo recall que v4 (de hecho, levemente peor). El patrón
es el mismo que en v3 (SpecAugment): el modelo se vuelve más conservador
(precisión sube, recall baja), en vez de aprender a reconocer mejor el
cardíaco de HLS-CMDS. Esto sugiere que comprimir la envolvente espectral
con la DCT no es suficiente para neutralizar la inversión de brillo
detectada en la sección 7 de `4_gate_model_baseline_y_sesgo.md`: el
problema no es solo "cuánta resolución espectral ve el modelo", sino que la
señal principal que aprendió (brillo espectral) sigue estando invertida en
el dominio de HLS-CMDS, la comprima como la comprima.

Aun así, **v5 pasa a ser la versión vigente** del gate por tener mejor F1
en HLS-CMDS (0.847 vs. 0.816) — checkpoint en
`gate_model/checkpoints_v5_mfcc/best.pt`. Pero queda documentado que este
resultado no invalida el diagnóstico de dominio de la sección 7: la
brecha probablemente no se cierra con más ingeniería de features sobre la
misma señal de brillo espectral, sino que requeriría (a) una feature menos
dependiente del timbre absoluto (ver `avances/7_...md`, ideas de
periodicidad/ritmo) o (b) confirmar/descartar si el patrón encontrado es
un artefacto específico del manikin HLS-CMDS y no una limitación general
del enfoque — ver discusión en `avances/7_...md`.

## 6. Próximos pasos

- Ver `avances/7_investigacion_debilidad_cardiaca.md` para el análisis de
  causas (protocolo de grabación de HLS-CMDS) y alternativas de features
  discutidas con el usuario tras este resultado.
- Independiente de esa investigación: evaluar agregación por grabación
  completa (voto mayoritario entre ventanas de un mismo `file_id`) y avanzar
  a la etapa 2 (modelos de patología), como ya estaba previsto en
  `4_gate_model_baseline_y_sesgo.md` sección 9.
