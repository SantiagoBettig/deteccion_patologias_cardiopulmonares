# Modelo pulmonar — detección de sonidos adventicios por ciclo (v1 y v2)

## 1. Planteo de la tarea

Diagnosticar la enfermedad por paciente con ICBHI no es viable (ver la
revisión de clases en la sesión del 2026-09-27 y `avances/12`):

- 5 de los 6 diagnósticos agrupados tienen entre 6 y 16 pacientes (LRTI 2,
  Asthma 1 si se desagrupan).
- `bronchiolitis_asthma` junta 6 bebés con 1 adulto de 70 años.
- Hay un sesgo fuerte por edad: sanos, URTI y bronquiolitis son pediátricos;
  COPD, neumonía y bronquiectasia, adultos.

Se replantea como la **tarea oficial del challenge ICBHI 2017**: detectar
sonidos adventicios en cada ciclo respiratorio. Hay 6898 ciclos anotados de
los 126 pacientes:

| Clase (crackle, wheeze) | Ciclos |
|---|---|
| normal (0, 0) | 3642 |
| crackle (1, 0) | 1864 |
| wheeze (0, 1) | 886 |
| both (1, 1) | 506 |

- **Crackle (crepitante)**: discontinuo, transitorio explosivo de menos de
  ~20 ms. En el espectrograma se ve como una línea vertical.
- **Wheeze (sibilancia)**: continuo y musical, de más de 100-250 ms. En el
  espectrograma se ve como una línea horizontal (tono con armónicos).

Es la analogía pulmonar del murmullo cardíaco: un evento acústico, no un
diagnóstico clínico (en cardíaco, el `Outcome` clínico tampoco resultó
resoluble por audio, `avances/10` §5).

## 2. Pipeline

| Paso | Script | Detalle |
|---|---|---|
| Ciclos | `preprocessing/make_pulmonary_cycles.py` | Preprocesamiento de siempre (pasa-bajos 2048 Hz → 4 kHz → RMS por grabación) y un WAV por ciclo anotado, sin padding. 0 errores, 0 descartados. |
| Folds | `splits/make_pulmonary_kfold.py` | 5 folds por paciente, estratificados por diagnóstico. Test: 25-26 pacientes por fold. Val: 16 pacientes (15% del resto por diagnóstico). Cada paciente cae en test exactamente una vez. |
| Modelo | `pulmonary_adventitious_model/` | La misma CNN de 3 bloques del gate y el murmullo, con **2 salidas sigmoide** (crackle, wheeze): "both" son las dos activas a la vez. |
| Evaluación | `run_kfold.py` | K-fold desde la primera corrida (lección de `avances/10` §14). Guarda las predicciones out-of-fold (OOF) de los 6898 ciclos y el desglose por equipo y edad. |

**Métrica principal**: el score oficial ICBHI sobre las 4 clases. Sp = ciclos
normales predichos normales. Se = ciclos anormales predichos con la clase
exacta. Score = (Se + Sp) / 2. **Referencia trivial**: predecir todo como
"normal" da Se = 0, Sp = 1, score = **0.50**. En la literatura, con el split
oficial y modelos preentrenados, los mejores resultados rondan 0.60-0.65.
También se reporta el AUC por etiqueta, que no depende del umbral.

Decisiones distintas a las del modelo de murmullo:

- **STFT más fina**: n_fft=256 y hop=64 a 4 kHz, es decir ventana de 64 ms y
  paso de 16 ms (murmullo: 128 ms / 40 ms). Con un paso de 40 ms, un
  crackle quedaría diluido en uno o dos frames.
- **log-Mel (40 bandas) en vez de MFCC**: un wheeze es una línea tonal
  angosta, y 20 coeficientes MFCC resumen la envolvente espectral.
- **Duración fija de 4 s** (percentil ~88 de la duración de ciclo). Los
  ciclos cortos se rellenan con 0 en la feature, después del z-score. No se
  usa repeat padding, porque cada unión mete un transitorio que parece un
  crackle. Los ciclos largos se recortan: al azar en train, centrado en
  evaluación.
- **Loss**: BCE con `pos_weight` natural por etiqueta (≈2.1 crackle, ≈3.6
  wheeze). Sin el ×1.4 del murmullo: allá se buscaba recall a propósito, y
  acá el score ya pesa igual Se y Sp.
- **Checkpoint y umbrales**: mejor score ICBHI en val, con los dos umbrales
  calibrados en conjunto sobre val (grilla de 19 × 19).
- Resto igual al murmullo: Adam 1e-3, dropout 0.3, augmentation de ganancia
  y ruido, 20 épocas.

## 3. v1 — pooling por promedio global (igual que gate y murmullo)

| Métrica (media ± std, 5 folds) | v1 |
|---|---|
| Score ICBHI | 0.477 ± 0.014 |
| Se / Sp | 0.146 ± 0.220 / 0.808 ± 0.219 |
| AUC crackle / wheeze | 0.611 ± 0.070 / 0.606 ± 0.086 |
| F1 crackle / wheeze | 0.189 / 0.068 |

**Peor que la referencia trivial** (0.50). AUC ≈ 0.61: separa las clases
apenas por encima del azar.

**Diagnóstico.** A diferencia del murmullo, **no hay sobreajuste: el modelo
casi no aprende ni en train**. La train_loss baja de ~1.00 a ~0.87 en 20
épocas, en todos los folds. Para distinguir un bug de un límite de
arquitectura se corrió una prueba de control: 600 ciclos al azar, sin
augmentation, sin dropout, 30 épocas. Un pipeline sano debería
memorizarlos.

| Pooling final | AUC en train, época 30 (crackle / wheeze) |
|---|---|
| Promedio global (v1) | 0.823 / 0.846 — no memoriza ni 600 ejemplos |
| Promedio en frecuencia + **máximo en tiempo** | **0.998 / 1.000** |

Cambiando **solo** el pooling, memoriza. **Causa**: el promedio global
diluye los eventos cortos. Un crackle ocupa pocos frames de los ~250 del
ciclo, y al promediar su activación se pierde. En el murmullo no molestaba,
porque un soplo ocupa buena parte del ciclo. Fue un error de diseño por
arrastrar la arquitectura del murmullo, no un límite de los datos. (La
prueba de control mide capacidad de aprender, no de generalizar.)

## 4. v2 — máximo en el tiempo

Único cambio respecto a v1: `AdventitiousCNN(pooling="time_max")` (ver
`model.py`). Mismos datos, folds, épocas, loss y calibración. v1 sigue
reproducible con `--pooling avg`.

| Métrica (media ± std, 5 folds) | v1 (promedio) | **v2 (máx. tiempo)** |
|---|---|---|
| Score ICBHI | 0.477 ± 0.014 | **0.511 ± 0.032** |
| Se | 0.146 ± 0.220 | 0.197 ± 0.140 |
| Sp | 0.808 ± 0.219 | 0.824 ± 0.101 |
| Accuracy 4 clases | 0.502 | 0.531 |
| AUC crackle | 0.611 ± 0.070 | 0.631 ± 0.051 |
| **AUC wheeze** | 0.606 ± 0.086 | **0.714 ± 0.061** |
| F1 crackle | 0.189 | 0.262 |
| F1 wheeze | 0.068 | 0.252 |

Score por fold (v2): 0.534 / 0.490 / 0.513 / 0.549 / 0.469.

Matriz de confusión OOF de v2 (filas = real, columnas = predicho):

```
         normal  crackle  wheeze  both
normal     3034      364     235     9
crackle    1341      490      26     7
wheeze      647       80     152     7
both        338       61     106     1
```

**Lectura:**

- **Mejora real pero modesta.** El score pasa de 0.477 a 0.511: supera
  apenas la referencia trivial y queda lejos de la literatura (0.60-0.65).
- **Casi toda la mejora está en wheeze** (AUC +0.11, 0.61 → 0.71). Crackle
  casi no se mueve (0.61 → 0.63). Queda como la etiqueta difícil: 1341 de
  1864 crackles se predicen como normales.
- **"both" prácticamente nunca se acierta** (1 de 506). Suele salir como
  wheeze o como normal: el modelo detecta la sibilancia pero no el
  crepitante que la acompaña.
- **Todavía no convergió.** La train_loss sigue bajando en la época 20
  (~0.80), y en varios folds el mejor checkpoint es de las últimas épocas
  (18, 20). El AUC de val de fold 0 sigue subiendo (wheeze 0.61 → 0.74
  entre las épocas 4 y 20).

## 5. Hallazgo metodológico — validación demasiado chica para elegir

En **fold 1** el checkpoint elegido fue el de la **época 1**, tanto en v1
como en v2. En val de ese fold, el AUC de crackle queda en ~0.48-0.51
durante las 20 épocas, así que el score ICBHI de val no mejora nunca y gana
el modelo casi sin entrenar. En test ese fold da Se = 0.025.

Pasa lo mismo con los umbrales: calibrados sobre 16 pacientes, no se
trasladan bien a test. Por ejemplo, en v2 los umbrales van de 0.45 a 0.90
según el fold.

Con 16 pacientes de validación, **el score ICBHI de val es demasiado ruidoso
para elegir el checkpoint y los umbrales**. Alternativas a evaluar:

- elegir el checkpoint por AUC medio de val, que no depende del umbral;
- entrenar un número fijo de épocas;
- agrandar val.

## 6. Chequeo de atajos (predicciones OOF de v2)

| Equipo | Pacientes | Ciclos | % crackle | % wheeze | Score |
|---|---|---|---|---|---|
| Littmann 3200 | 11 | 502 | 10.8 | 27.3 | 0.424 |
| Littmann Classic II SE | 54 | 1124 | 15.5 | 24.9 | 0.530 |
| AKG C417L | 56 | 4697 | 41.7 | 19.0 | 0.522 |
| Meditron | 8 | 575 | 31.8 | 14.3 | 0.468 |

| Edad | Pacientes | Ciclos | Score (Se / Sp) |
|---|---|---|---|
| Adulto | 76 | 6049 | 0.502 (0.200 / 0.804) |
| Pediátrico | 49 | 788 | 0.556 (0.144 / 0.969) |

(Los pacientes de COPD con más de un equipo cuentan en cada uno.)

No hay una señal clara de atajo. Los equipos con peor score (Littmann 3200,
Meditron) son 100% COPD y tienen pocos pacientes, así que la diferencia se
confunde con la mezcla de casos. En pediátricos, el score algo mayor viene
de la especificidad (hay pocos ciclos anormales), no de detectar mejor. Con
un modelo tan débil este chequeo tiene poco poder; hay que repetirlo cuando
el modelo mejore.

## 7. Conclusión y próximos pasos

- **v2 (máximo en el tiempo) reemplaza a v1** como base. El pooling era el
  cuello de botella principal para aprender, no los datos.
- El modelo sigue débil (score 0.511, AUC crackle 0.63). Líneas posibles,
  en orden de costo:
  1. **Más épocas** (30-40) y **elegir el checkpoint por AUC de val** en vez
     de score ICBHI (secciones 4 y 5). Barato, y ataca dos problemas
     concretos observados.
  2. **Crackles**: probar un pasa-altos (~100 Hz) para sacar los sonidos
     cardíacos, que en los registros pulmonares dominan las bajas
     frecuencias y el nivel RMS. Revisar si el techo de 2 kHz (resample a
     4 kHz) recorta contenido útil de los crackles finos.
  3. **Más capacidad** (4.º bloque / más canales): la prueba de control
     muestra que la red de 3 bloques puede memorizar, pero la train_loss de
     la corrida completa sigue alta.
  4. **Transfer learning** desde un modelo preentrenado en AudioSet (PANNs
     CNN6, viable en CPU), el "modelo C" discutido al plantear la tarea.

Reportes: `reports/pulmonary_adventitious/v1_logmel/` y `.../v2_logmel/`
(`summary.csv`, `per_fold.csv`, `oof_predictions.csv`,
`confusion_matrix_4class.png`, `metrics_by_equipment.csv`,
`metrics_by_age_group.csv`, `training_curves.png`). Logs completos en
`reports/pulmonary_adventitious_v{1,2}_logmel_run.log`. Checkpoints en
`pulmonary_adventitious_model/checkpoints/v{1,2}_logmel/`.
