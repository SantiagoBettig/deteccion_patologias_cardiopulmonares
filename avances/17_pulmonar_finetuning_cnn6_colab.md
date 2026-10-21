# Modelo pulmonar — fine-tuning de la CNN6 (CPU parcial, Colab A/B/C) y estudio de calibración

Continuación de `avances/15` (CNN6 congelada + MLP: 0.559 ± 0.024) y
`avances/16` (ensamble). Referencia vigente hasta ahora: **v4**, la CNN de 3
bloques entrenada desde cero, con score ICBHI **0.559 ± 0.007**, AUC crackle
0.676 y AUC wheeze 0.793.

Pregunta de fondo, planteada por el usuario: **con ~85 pacientes de train
por fold, ¿adaptar la red preentrenada (fine-tuning) mejora, o se queda
corto por la cantidad de datos y solo sobreajusta?**

## 1. Fine-tuning parcial en CPU (v7)

Para responder sin GPU: los ciclos se pasan **una sola vez** por la parte
congelada (bn0 + bloques 1-3, `extract_cnn6_block3.py`, 0.94 GB en float16)
y se entrena solo el bloque 4 + fc1 + una cabeza nueva
(`run_partial_finetune.py`), a ~2 min por época contra ~16 del fine-tuning
completo.

Decisiones para pocos datos, que se mantienen en todo lo que sigue:

- learning rate 1e-4 para lo preentrenado y 1e-3 para la cabeza;
- estadísticas de BatchNorm congeladas (las de AudioSet; solo se entrenan
  escala y desplazamiento);
- dropout de PANNs y enmascarado temporal como augmentation;
- pooling enmascarado sobre los frames reales;
- mismo protocolo de siempre (5 folds por paciente, checkpoint por AUC de
  val, umbrales calibrados en val).

Resultado, folds 0-3 (30 épocas): 0.608 / 0.596 / 0.573 / 0.528. En 3 de
4 folds supera a v4 y a la CNN6 congelada. Mejor época 27-30: seguía
mejorando, **sin señales de sobreajuste**.

**El fold 4 no llegó a terminar**: la corrida se cortó tres veces desde
afuera (cierre de sesión / límite de tiempo de tareas en segundo plano, sin
error de Python). Se agregó retome por época (`--resume`, estado completo en
`fold{i}_last.pt`), pero al confirmarse con Colab que la variante A replica
este experimento (§3), no se completó. v7 queda como evidencia de 4 folds.

## 2. Preparación de Colab

El intento anterior (`avances/5`) falló por leer miles de archivos chicos
desde Drive. Ahora:

- **`pack_for_colab.py`**: un solo zip de 263 MB con los espectrogramas
  log-Mel de PANNs **ya calculados** (entrada `panns32k` sobre los ciclos
  con pasa-altos), el índice, los 5 folds y los pesos de CNN6. Colab lo
  copia una vez al disco local. No remuestrea ni calcula STFTs, y toda la
  augmentation corre en GPU sobre el espectrograma. Control: el tronco
  congelado sobre el paquete coincide con la salida del bloque 3 de v7
  (diferencia ≤ 0.002, redondeo de float16).
- **`run_finetune_gpu.py`**: red completa en el loop, variantes A/B/C,
  precisión mixta en GPU, **guardado del estado completo al final de cada
  época** en Drive (retoma solo si Colab corta la sesión; probado matando el
  proceso a mitad de un fold) y guardado de las probabilidades de
  validación.
- **`colab_pulmonary_finetune.ipynb`**: clona el repositorio público, copia
  el paquete y entrena. No instala `requirements.txt`, para no reemplazar el
  torch con GPU de Colab.

**Velocidad en GPU (T4)**: 5-7 s por época (A, B) y 11-12 s (C), contra ~2
min del fine-tuning parcial en CPU. Una variante completa (5 folds) tarda
entre 20 y 35 minutos.

## 3. Resultados de las variantes

| Variante | Qué se entrena | Épocas | Augmentation | lr preentrenado |
|---|---|---|---|---|
| A | Bloque 4 + fc1 | 60 | Enmascarado temporal | 1e-4 |
| B | Bloques 3-4 + fc1 | 40 | Ganancia + SpecAugment (tiempo y frecuencia) | 1e-4 |
| C | Toda la red (bloques 1-4 + fc1) | 30 | Ganancia + SpecAugment | 3e-5 |

Media ± std, 5 folds, test:

| Modelo | Score ICBHI | Se | Sp | AUC crackle | AUC wheeze | F1 crackle | F1 wheeze |
|---|---|---|---|---|---|---|---|
| v4 (desde cero) | 0.559 ± 0.007 | 0.368 | 0.750 | 0.676 | 0.793 | 0.464 | 0.475 |
| CNN6 congelada + MLP | 0.559 ± 0.024 | 0.357 | 0.761 | 0.675 | 0.773 | 0.449 | 0.471 |
| A | 0.563 ± 0.041 | 0.416 | 0.710 | 0.695 | **0.840** | 0.487 | **0.568** |
| **B** | **0.572 ± 0.027** | 0.414 | 0.731 | **0.708** | 0.836 | **0.506** | 0.548 |
| C | 0.555 ± 0.036 | 0.345 | 0.764 | 0.677 | 0.832 | 0.435 | 0.470 |

Score por fold:

| Fold | v4 | v7 (CPU) | A | B | C |
|---|---|---|---|---|---|
| 0 | 0.564 | 0.608 | 0.602 | 0.608 | 0.605 |
| 1 | 0.567 | 0.596 | 0.594 | 0.583 | 0.572 |
| 2 | 0.558 | 0.573 | 0.571 | 0.581 | 0.547 |
| 3 | 0.550 | 0.528 | 0.544 | 0.549 | 0.542 |
| 4 | 0.556 | — | **0.502** | 0.542 | 0.508 |

**B contra v4, fold por fold** (comparación pareada, mismos pacientes de
test):

| Métrica | Diferencia por fold | Folds donde mejora | Media |
|---|---|---|---|
| Score ICBHI | +0.044, +0.016, +0.023, −0.001, −0.014 | 3/5 | +0.014 |
| AUC crackle | +0.031, +0.039, −0.001, +0.032, +0.063 | 4/5 | +0.033 |
| AUC wheeze | +0.025, +0.037, +0.078, +0.014, +0.064 | **5/5** | +0.044 |

Matriz de confusión OOF de B (filas = real, columnas = predicho):

```
         normal  crackle  wheeze  both
normal     2680      775     167    20
crackle     851      963      32    18
wheeze      359      119     317    91
both        143      118     174    71
```

### Lectura

- **El fine-tuning mejora, y no sobreajustó**, que era la duda del usuario.
  La separabilidad (AUC, que no depende del umbral) mejora en forma
  consistente: wheeze en los 5 folds (+0.044 de media) y crackle en 4 de 5
  (+0.033). Las curvas no muestran sobreajuste: en A y B el AUC de val se
  estabiliza sin caer, y la loss de train de B queda en ~0.72 (la
  augmentation frena la memorización).
- **El score ICBHI mejora menos que el AUC** (+0.014 con B), por lo mismo
  que se vio en el ensamble: el score depende de los umbrales, calibrados
  con 16 pacientes (§4).
- **A ≈ v7**: los folds 0-3 de A replican v7 con diferencias de ~0.01,
  lo que valida el pipeline de Colab. Duplicar las épocas (30 → 60) casi no
  aportó: el AUC de val ya estaba en su techo cerca de la época 30, y la
  loss de train siguió bajando (0.63 → 0.48), un sobreajuste incipiente que
  no llega a dañar val.
- **B es la mejor variante**: entrenar también el bloque 3, con
  augmentation completa, da el mejor score, el mejor AUC de crackle y la
  menor variabilidad entre los modelos ajustados. No colapsa en el fold 4
  (0.542, contra 0.502 de A). Su mejor época cae en 35-40 de 40: todavía
  tenía margen.
- **C (toda la red) no mejoró, pero no por sobreajuste**: su loss de train
  queda alta (~0.80-0.85) y el AUC de val sigue subiendo al final. Con lr
  3e-5 y augmentation completa, en 30 épocas **aprende demasiado lento**.
  No descarta que ajustar toda la red sirva; descarta esta configuración.
- **Edad y equipo**: sin cambios de fondo. La sensibilidad pediátrica sigue
  en ~0.16-0.21 en todas las variantes (limitación de datos, `avances/14`
  §4). Littmann 3200 y Meditron siguen siendo los grupos más débiles.

## 4. Estudio de calibración de umbrales

Motivación: el fold 4 de A cayó a 0.502 con los umbrales de val, contra
0.580 con los ideales.

Herramientas:

- `export_val_probs.py` reconstruye las probabilidades de val de v4 y A
  desde sus checkpoints. Control: test reproduce exacto en v4 y con
  diferencia < 0.009 en A, por la precisión mixta de Colab.
- `calibration_study.py` compara métodos de elección de umbrales **fijados
  antes de ver resultados**, sobre 4 modelos × 5 folds = 20 casos:

| Método (solo con val) | Diferencia vs. actual | Mejor / peor | Pérdida media vs. oráculo | Peor pérdida |
|---|---|---|---|---|
| Máximo de la grilla 19×19 (actual) | — | — | 0.018 | 0.078 |
| Grilla suavizada (3×3) | −0.001 | 3 / 3 | 0.018 | 0.078 |
| Bootstrap de pacientes (mediana de 200) | +0.001 | 2 / 2 | 0.017 | 0.078 |
| Youden por etiqueta | −0.071 | 0 / 20 | 0.089 | 0.150 |
| Prevalencia por etiqueta | −0.028 | 2 / 17 | 0.046 | 0.106 |

(Oráculo = umbrales elegidos mirando test; solo como techo.)

**Conclusiones:**

- **Ningún método mejora de forma consistente al actual**: los robustos
  empatan y los que eligen cada umbral por separado son claramente peores.
  El score ICBHI exige acertar la combinación exacta, así que hay que
  optimizar los dos umbrales juntos. **Se mantiene el método actual.**
- **La pérdida típica por calibración es chica** (~0.017 de media). Lo
  preocupante son fallas puntuales, como el fold 4 de A (0.078), que
  **ningún método basado solo en val puede evitar**.
- **Causa del fold 4**: dos pacientes con COPD del test (ICB_213, 129
  ciclos, 14% con crackle; ICB_162, 89 ciclos, 8% con crackle) cuyos ciclos
  **normales** reciben probabilidades de crackle muy altas (medianas 0.62 y
  0.47; en ICB_162 incluso más altas que sus ciclos con crackle). En val de
  ese fold no hay pacientes así, y el umbral elegido (0.50) queda bajo para
  test. No es el equipo (ambos AKG, el más común). Puede ser que suenen
  "crepitantes" sin cumplir el criterio del anotador, o ruido de etiqueta
  (documentado en ICBHI); con estos datos no se puede distinguir.
- **Chequeo de atajo a nivel paciente**, sobre las predicciones OOF de A: la
  relación entre el % de crackles de un paciente y la probabilidad que
  reciben sus ciclos normales es débil (Spearman r = 0.18; 0.10 en v4). El
  AUC de crackle **dentro de cada grabación** (ciclos del mismo archivo con
  y sin crackle) es 0.667 de mediana, cerca del global (0.689). El modelo
  mayormente detecta eventos por ciclo, no reconoce pacientes.

## 5. Conclusión y próximos pasos

- **Nueva versión vigente del modelo pulmonar: variante B** — CNN6 de PANNs
  preentrenada, bloques 3-4 + fc1 ajustados, augmentation en GPU, entrada
  `panns32k` con pasa-altos. **Score ICBHI 0.572 ± 0.027, AUC crackle 0.708,
  AUC wheeze 0.836.** Mejora sobre v4 consistente en separabilidad (AUC) y
  más modesta en score.
- **Respuesta a la pregunta de fondo**: con este volumen de datos, el
  fine-tuning **sí** mejora sobre entrenar desde cero y sobre usar la red
  congelada, siempre que se adapte una parte de la red (no toda) y con
  augmentation. La mejora es real pero acotada: los datos siguen siendo el
  límite.
- Líneas abiertas, todas baratas en Colab:
  1. **B con más épocas** (su mejor época cae al final; ~60 épocas ≈ 35
     min).
  2. **C con lr más alto** (1e-4): queda abierto si ajustar toda la red
     ayuda con una configuración que llegue a converger.
  3. **Calibración con validación cruzada interna** dentro de los ~85
     pacientes de train, la única vía para reducir fallas como la del fold
     4.
- La sensibilidad pediátrica (~0.16) sigue siendo la limitación principal y
  ningún cambio de modelo la movió: viene de los datos.

Reportes: `reports/pulmonary_adventitious/v7_cnn6_partial_ft/` (4 folds),
`.../v8_cnn6_ft_{A,B,C}/`, `.../calibration_study/`. Logs de v7:
`reports/pulmonary_adventitious_v7_cnn6_partial_ft*.log`.
