# Modelo pulmonar — v3 (40 épocas + checkpoint por AUC) y v4 (pasa-altos 100 Hz)

Continuación de `avances/13`. Parte de v2 (CNN de 3 bloques, log-Mel,
pooling por máximo en el tiempo, 5 folds por paciente): score ICBHI
0.511 ± 0.032. v2 tenía dos problemas concretos, que se atacan acá en ese
orden:

1. **No había convergido** en 20 épocas (la train_loss seguía bajando) y la
   **elección del checkpoint por score ICBHI de val era muy ruidosa**, con
   solo 16 pacientes de validación. En fold 1 eligió la época 1
   (`avances/13` §5).
2. **Los crackles casi no se detectaban** (AUC 0.63).

## 1. v3 — 40 épocas y mejor checkpoint por AUC medio de validación

Cambios respecto a v2 (`run_kfold.py --epochs 40 --select auc`):

- 40 épocas en vez de 20.
- Mejor checkpoint = mayor **AUC medio (crackle, wheeze) de validación**,
  en vez del score ICBHI de val. El AUC no depende de ningún umbral: mide
  solo separabilidad, así que saca el ruido de calibrar umbrales sobre 16
  pacientes. Los umbrales se calibran igual, **una sola vez**, sobre el
  checkpoint ya elegido.
- Todo lo demás, igual. Con la misma semilla, la época 1 da exactamente lo
  mismo que en v2, así que la comparación es limpia.

Resultados en §3. Lectura:

- **Sube la sensibilidad** (0.197 → 0.292) y **bajan los desvíos entre
  folds** en casi todas las métricas.
- **Se resuelve el caso del fold 1**: v2 eligió la época 1 (test 0.490); v3
  elige la época 40 (test 0.561).
- **Mejor época por fold**: 37 / 40 / 40 / 32 / 21. En dos folds el mejor
  checkpoint es el último: todavía hay margen con más épocas.
- **Crackle sigue débil** (AUC 0.655). 1139 de 1864 crackles se predicen
  como normales.

## 2. v4 — pasa-altos a 100 Hz

### Motivación, medida antes de entrenar

Se midió qué fracción de la energía de cada ciclo está por debajo de
100 Hz, con 300 ciclos al azar:

| Ciclos | Mediana de la fracción de energía < 100 Hz |
|---|---|
| Preprocesamiento base (pasa-bajos 2048 Hz → 4 kHz → RMS) | **0.98** |
| Con pasa-altos 100 Hz | 0.069 (la caída gradual del filtro) |

En las grabaciones de pulmón, **casi toda la energía está por debajo de
100 Hz**: sonidos cardíacos (S1/S2, concentrados por debajo de ~150 Hz),
roce y ruido de baja frecuencia. Eso tiene dos efectos:

- La **normalización RMS por grabación termina escalando según el volumen
  del corazón**, no según el de la respiración.
- Los sonidos de interés quedan como una fracción mínima de la señal: los
  crackles son de banda ancha (~100-2000 Hz) y los wheezes suelen estar por
  encima de 100-400 Hz.

### Implementación

- `preprocessing/audio_ops.py`: nuevo `highpass_filter` (Butterworth de
  orden 4, fase cero) y parámetro opcional `preprocess(...,
  highpass_cutoff=None)`. Se aplica antes del resample. Con `None`, el
  comportamiento es el de siempre (el gate y el modelo cardíaco no cambian).
- `preprocessing/make_pulmonary_cycles.py --highpass 100` genera los ciclos
  en `data/pulmonary_hp100/` (función `common/paths.py:
  pulmonary_cycles_paths`). Los ciclos de v1-v3 quedan intactos.
- `run_kfold.py --epochs 40 --select auc --cycles-variant hp100 --run-name
  v4_hp100`: receta idéntica a v3, solo cambia el audio de entrada.

### Por qué no se subió también el techo de 2 kHz

Se evaluó volver a muestrear a más de 4 kHz para no recortar contenido de
crackles finos. **Se descartó por riesgo de atajo**: Littmann 3200 y
Meditron graban de forma nativa a 4 kHz (no tienen contenido por encima de
2 kHz), y sus pacientes son 100% COPD (`avances/12` §3). Con una banda más
ancha, "tiene o no contenido por encima de 2 kHz" delataría el equipo y, a
través de él, el diagnóstico. El recorte a 2 kHz iguala a todos los equipos.

## 3. Comparación v1 → v4 (media ± std, 5 folds, test)

| Métrica | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|
| Cambio | base (pool promedio) | pool máx. tiempo | 40 ép. + checkpoint por AUC | **+ pasa-altos 100 Hz** |
| Score ICBHI | 0.477 ± 0.014 | 0.511 ± 0.032 | 0.524 ± 0.027 | **0.559 ± 0.007** |
| Se | 0.146 ± 0.220 | 0.197 ± 0.140 | 0.292 ± 0.091 | **0.368 ± 0.083** |
| Sp | 0.808 ± 0.219 | 0.824 ± 0.101 | 0.756 ± 0.058 | 0.750 ± 0.081 |
| Accuracy 4 clases | 0.502 | 0.531 | 0.539 | **0.570** |
| AUC crackle | 0.611 ± 0.070 | 0.631 ± 0.051 | 0.655 ± 0.033 | **0.676 ± 0.039** |
| AUC wheeze | 0.606 ± 0.086 | 0.714 ± 0.061 | 0.776 ± 0.044 | **0.793 ± 0.027** |
| Recall crackle | 0.208 | 0.214 | 0.290 | **0.415** |
| Recall wheeze | 0.041 | 0.225 | 0.354 | **0.422** |
| F1 crackle | 0.189 | 0.262 | 0.340 | **0.464** |
| F1 wheeze | 0.068 | 0.252 | 0.380 | **0.475** |

Score por fold (v4): 0.564 / 0.567 / 0.558 / 0.550 / 0.556. Mejor época
por fold: 35 / 27 / 33 / 26 / 39.

Matriz de confusión OOF de v4 (filas = real, columnas = predicho):

```
         normal  crackle  wheeze  both
normal     2741      587     264    50
crackle     965      816      51    32
wheeze      389      124     314    59
both        222       95     129    60
```

(v3, para comparar: crackle→crackle 643, wheeze→wheeze 300, both→both 14.)

**Lectura de v4:**

- **Es la mejora más grande de la serie**: score 0.524 → 0.559 y
  **desvío entre folds de solo 0.007** (los 5 folds entre 0.550 y 0.567).
  El resultado ya no depende de qué pacientes tocaron en cada fold.
- **Mejora sobre todo crackle**, que era el objetivo: recall 0.290 → 0.415,
  F1 0.340 → 0.464, 816 crackles bien detectados contra 643. Wheeze sigue
  mejorando, pero menos (AUC 0.776 → 0.793).
- **"both" empieza a aparecer**: 60 de 506 aciertos, contra 14 en v3.
- Las mejoras de v2 → v4 son todas cambios de **cómo se presenta la señal**
  (pooling, audio de entrada) y de **cómo se elige el modelo**, no de su
  capacidad. Es el mismo patrón que en murmullo: en esta rama el cuello de
  botella no era el tamaño de la red.
- **Referencia externa**: la literatura reporta ~0.60-0.65 en ICBHI con
  modelos preentrenados, split oficial 60/40 y más frecuencia de muestreo.
  Nuestro 0.559 viene de una CNN chica, entrenada desde cero, en CPU y con
  k-fold por paciente. No es directamente comparable, pero ubica el orden
  de magnitud.

## 4. Chequeo de atajos (predicciones OOF)

Score por equipo:

| Equipo | Pacientes | v3 | v4 |
|---|---|---|---|
| Littmann 3200 (100% COPD) | 11 | 0.502 | 0.519 |
| Littmann Classic II SE | 54 | 0.579 | **0.622** |
| AKG C417L | 56 | 0.533 | 0.554 |
| Meditron (100% COPD) | 8 | **0.397** | 0.505 |

Score por grupo de edad (Se / Sp):

| Edad | v3 | v4 |
|---|---|---|
| Adulto (76 pacientes, 6049 ciclos) | 0.516 (0.300 / 0.733) | 0.549 (0.377 / 0.722) |
| Pediátrico (49 pacientes, 788 ciclos) | 0.548 (0.212 / 0.883) | 0.536 (0.171 / 0.900) |

- **El pasa-altos achica la brecha entre equipos.** Meditron sube de 0.397
  a 0.505, la mayor mejora individual. Es consistente con que parte de lo
  que distinguía a cada equipo estaba en la banda baja (respuesta de la
  pieza, ruido de manipulación), y el filtro lo iguala.
- **Pediátricos: sensibilidad baja** (0.171 en v4, contra 0.377 en
  adultos), y es el único grupo que no mejora con v4. Hay pocos ciclos
  anormales pediátricos (~150). Además, los ciclos anormales de train son
  en su mayoría de adultos con COPD, así que el modelo aprende sobre todo
  cómo suenan crackles y wheezes adultos. Otra posible contribución: en
  niños la frecuencia cardíaca es más alta y parte de los sonidos cardíacos
  puede quedar por encima de 100 Hz. **Queda como limitación a
  documentar**: el modelo detecta peor los sonidos adventicios en
  pacientes pediátricos.

## 5. Conclusión y próximos pasos

- **v4 es la versión vigente** del modelo pulmonar: CNN de 3 bloques,
  log-Mel, pooling por máximo en el tiempo, pasa-altos a 100 Hz, 40 épocas,
  checkpoint por AUC de val y umbrales calibrados sobre val. **Score ICBHI
  0.559 ± 0.007**, AUC crackle 0.676, AUC wheeze 0.793.
- Líneas abiertas:
  1. **Más épocas o un scheduler**: en 3 de 5 folds el mejor checkpoint es
     de la época 33 o posterior.
  2. **Probar el corte del pasa-altos** (ej. 150-200 Hz), por los sonidos
     cardíacos pediátricos (§4).
  3. **Más capacidad** (4.º bloque / más canales): la train_loss de v4
     sigue en ~0.64-0.70.
  4. **Transfer learning** (PANNs CNN6), el "modelo C" pendiente.
  5. **Repetir el análisis de murmullo con pasa-altos**: el mismo problema
     de energía dominante en banda baja podría afectar la rama cardíaca
     (aunque ahí los sonidos cardíacos son la señal, no el ruido).

Reportes: `reports/pulmonary_adventitious/v3_auc_40ep/` y `.../v4_hp100/`.
Logs: `reports/pulmonary_adventitious_v{3_auc_40ep,4_hp100}_run.log`.
Checkpoints: `pulmonary_adventitious_model/checkpoints/v{3_auc_40ep,4_hp100}/`.
