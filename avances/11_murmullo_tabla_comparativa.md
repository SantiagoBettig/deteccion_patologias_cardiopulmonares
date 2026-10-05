# Modelo de murmullo cardíaco — tabla comparativa de versiones

Resumen de todas las versiones probadas para el modelo binario de
murmullo (`cardiac_murmur_model/`), con qué se modificó en cada una y
dónde encontrar el detalle completo. El detalle de cada experimento
(motivación, hipótesis, diagnóstico) está en
[`avances/10_cardiaco_split_dos_tareas.md`](10_cardiaco_split_dos_tareas.md)
— este documento es solo la tabla resumen para cotejar de un vistazo.

No incluye el modelo de outcome (`cardiac_outcome_model/`) ni el intento
original de 3 clases (`cardiac_model/`) — esos quedaron cerrados como
limitación conocida y están documentados aparte en el mismo
`avances/10` (secciones 1-4).

## Tabla comparativa (métricas en test, ventana por ventana salvo que se indique lo contrario)

| Versión | Qué se probó / modificó respecto a la anterior | Umbral | Accuracy | Precision | Recall | F1 | F2 | Referencia |
|---|---|---|---|---|---|---|---|---|
| Baseline inicial | MFCC, `BCEWithLogitsLoss` con `pos_weight` natural (4.77x), checkpoint elegido por mejor F1 de validación, sin dropout, umbral fijo 0.5. Punto de partida antes de ajustar nada. | 0.50 | 0.927 | 0.913 | 0.508 | 0.653 | — | `avances/10` §4 |
| **v1** | Mismo audio (MFCC) que el baseline, pero: `pos_weight`×1.4 sobre el natural, checkpoint elegido por **F2** (no F1) — prioriza recall, dropout (p=0.3) por el sobreajuste visto en el baseline, umbral de decisión calibrado sobre validación (grilla, máximo F2) en vez de 0.5 fijo. **Mejor versión hasta ahora.** | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | **0.710** | `avances/10` §5 · `checkpoints_v1_mfcc/` · `reports/cardiac_murmur/runs/v1_mfcc/` |
| v2 | Espectrograma **log-Mel** en vez de MFCC — resto de la receta igual a v1. Hipótesis: log-Mel retiene más contenido de alta frecuencia que MFCC recorta. No se confirmó. | 0.60 | 0.881 | 0.542 | 0.756 | 0.631 | 0.701 | `avances/10` §6 · `checkpoints_v2_logmel/` · `.../runs/v2_logmel/` |
| v3 | `ReduceLROnPlateau` sobre `val_f2` (baja el LR cuando F2 deja de mejorar 2 épocas) — resto igual a v1 (MFCC). Hipótesis: los picos de `val_loss` venían de un LR muy alto. No se confirmó — el LR bajó pero los picos siguieron igual. | 0.40 | 0.852 | 0.471 | 0.750 | 0.579 | 0.671 | `avances/10` §7 · `checkpoints_v3_lr_scheduler/` · `.../runs/v3_lr_scheduler/` |
| v4 | Metadata del paciente (sexo, edad, altura, peso) agregada como entrada, fusionada por concatenación con el embedding de audio antes de la capa final (`MurmurCNNMultimodal`) — resto igual a v1. Diagnóstico previo: regresión logística solo-metadata da accuracy 0.482 en test (por debajo de la referencia trivial 0.865) — confirma que la metadata casi no tiene señal propia. | 0.45 | 0.794 | 0.376 | 0.788 | 0.509 | 0.646 | `avances/10` §8 · `checkpoints_v4_metadata/` · `.../runs/v4_metadata/` · diagnóstico: `diagnostics_metadata_only.py` |
| v5 | **FocalLoss** (alpha=0.75, gamma=2.0, ver `losses.py`) en vez de `BCEWithLogitsLoss`+`pos_weight` — resto igual a v1 (MFCC). Hipótesis: concentrar el gradiente en ejemplos difíciles en vez de escalar uniformemente los positivos podría estabilizar el entrenamiento. No se confirmó — la oscilación de `val_f2` entre épocas siguió igual de fuerte. | 0.50 | 0.895 | 0.600 | 0.675 | 0.635 | 0.659 | `avances/10` §10 · `checkpoints_v5_focal_loss/` · `.../runs/v5_focal_loss/` |
| v6 | **Ubicación de auscultación** (one-hot, 5 categorías) agregada como entrada, misma fusión tardía que v4 (`MurmurCNNMultimodal`) — resto igual a v1. Hipótesis: a diferencia de la metadata demográfica, la ubicación está mecánicamente ligada a si el murmullo se escucha, mejor candidato. Tampoco mejoró. | 0.35 | 0.843 | 0.451 | 0.740 | 0.561 | 0.656 | `avances/10` §11 · `checkpoints_v6_location/` · `.../runs/v6_location/` |
| v7 | Ventanas de **train** más densas (hop=1 ciclo en vez de 2 — más ejemplos correlacionados por grabación, train 18523→35963 ventanas), val/test sin cambios, sobre la receta de v1 (MFCC). | 0.40 | 0.854 | 0.476 | 0.752 | 0.583 | 0.674 | `avances/10` §12 · `checkpoints_v7_dense_train/` · `.../runs/v7_dense_train/` |

## Agregación por sujeto (no es una versión nueva — otra forma de leer las predicciones de v1)

En vez de evaluar ventana por ventana, agregar todas las probabilidades de
un mismo sujeto (todas sus ubicaciones/ciclos) antes de decidir — sin
reentrenar, aplicado sobre el checkpoint de v1:

| Agregación | Umbral (sobre prob. agregada) | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|---|
| Ventana individual (v1, referencia) | 0.45 | 0.895 | 0.589 | 0.749 | 0.659 | 0.710 |
| Por sujeto — **máximo** | 0.85 | 0.887 | 0.704 | 0.704 | **0.704** | 0.704 |
| Por sujeto — media | 0.50 | 0.909 | 0.889 | 0.593 | 0.711 | 0.635 |

Referencia: `avances/10` §9 · `evaluate_subject_level.py` ·
`reports/cardiac_murmur/test_metrics_subject_level.csv`.

## Validación k-fold (K=5, estratificado por sujeto) de la receta de v1

Antes de cerrar, se validó v1 con k-fold — cada uno de los ~179 sujetos
con murmullo pasa por el rol de test exactamente una vez a lo largo de 5
folds, en vez de depender de un solo split. Ver `avances/10` §14,
`splits/make_cardiac_murmur_kfold.py`, `cardiac_murmur_model/run_kfold.py`.

| Nivel | Accuracy | Precision | Recall | F1 | F2 |
|---|---|---|---|---|---|
| Ventana (k-fold, media±std) | 0.869±0.023 | 0.606±0.062 | 0.782±0.026 | 0.681±0.039 | **0.737±0.022** |
| Ventana (v1, split único) | 0.895 | 0.589 | 0.749 | 0.659 | 0.710 |
| Sujeto — máximo (k-fold, media±std) | 0.492±0.079 | 0.268±0.034 | 0.938±0.024 | 0.416±0.041 | 0.623±0.036 |
| Sujeto — máximo (split único) | 0.887 | 0.704 | 0.704 | **0.704** | 0.704 |

**El k-fold confirma la performance a nivel ventana** (F2 consistente entre
split único y k-fold, dentro de 1 desvío) — **pero desmiente la mejora por
agregación por sujeto**: en el split único parecía una mejora grande
(F1 0.659→0.704), pero en k-fold el promedio real es mucho **peor** que a
nivel ventana (F1=0.416), no mejor. Fue en buena parte un efecto
favorable de esa partición particular — el tipo de espejismo que el
k-fold está pensado para detectar.

## Lectura rápida (conclusión final)

- **v1 sigue siendo la mejor versión "de modelo"** (F2≈0.71-0.74, validado
  por k-fold) — ninguna de las seis modificaciones probadas después
  (v2-v7: log-Mel, LR scheduler, metadata, focal loss, ubicación, más
  datos de train) la superó.
- Patrón que se repite en v3/v4/v5/v6/v7: agregar cualquier cosa a la
  receta base (scheduler, metadata, focal loss, ubicación, más ventanas
  correlacionadas) empeoró el resultado o no cambió nada — con ~179
  sujetos con murmullo, cualquier rama/parámetro nuevo parece competir por
  gradiente con la señal de audio en vez de sumar, y más ventanas del
  mismo puñado de sujetos no suman información nueva.
- La agregación por sujeto (sección 9) **no se sostuvo** bajo k-fold — se
  descarta como mejora, queda documentada como hallazgo negativo.
- **Versión final recomendada**: v1, evaluado a nivel ventana —
  F2≈0.71-0.74, recall≈0.75-0.78 (validado por k-fold, no depende de un
  solo split favorable).
