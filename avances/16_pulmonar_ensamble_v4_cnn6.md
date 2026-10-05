# Modelo pulmonar — ensamble v4 + CNN6 congelada

Continuación de `avances/15`. Dos modelos de naturaleza muy distinta llegan
al mismo score ICBHI (0.559):

- **A = v4**: CNN de 3 bloques entrenada desde cero sobre nuestros ciclos
  (`avances/14`).
- **B = CNN6 de PANNs congelada + MLP**: features genéricas de AudioSet, con
  entrada `panns32k` (`avances/15`).

Antes de decidir si vale la pena el fine-tuning en Colab, se prueba una idea
barata: **promediar sus probabilidades**. Si se equivocan en ciclos
distintos, el promedio debería mejorar a los dos.

## 1. Método (`pulmonary_adventitious_model/run_ensemble.py`)

- Mismos 5 folds por paciente. Probabilidades de val y test de A (cargando
  los checkpoints de v4) y de B (reentrenando el MLP con la misma semilla:
  es determinístico). **Control**: A y B reproducen exactamente sus
  resultados anteriores (0.559 ± 0.007 y 0.559 ± 0.024).
- **Ensamble principal**: promedio simple, p = (p_A + p_B) / 2, sin ajustar
  ningún peso, para no sobreajustar a los 16 pacientes de val.
- **Referencia secundaria**: peso w de p = w·p_A + (1 − w)·p_B elegido por
  AUC medio de val.
- Umbrales calibrados sobre val, como siempre.

## 2. Resultados (media ± std, 5 folds, test)

| Modelo | Score ICBHI | Se | Sp | AUC crackle | AUC wheeze | F1 crackle | F1 wheeze |
|---|---|---|---|---|---|---|---|
| A — v4 | 0.559 ± 0.007 | 0.368 | 0.750 | 0.676 | 0.793 | 0.464 | 0.475 |
| B — CNN6 congelada | 0.559 ± 0.024 | 0.357 | 0.761 | 0.675 | 0.773 | 0.449 | 0.471 |
| **Ensamble promedio** | 0.562 ± 0.029 | **0.417** | 0.708 | **0.691** | **0.821** | **0.503** | **0.543** |
| Ensamble con peso de val | 0.560 ± 0.027 | 0.426 | 0.694 | 0.691 | 0.818 | 0.497 | 0.555 |

Score por fold:

| Fold | A (v4) | B (CNN6) | Ensamble promedio |
|---|---|---|---|
| 0 | 0.564 | 0.593 | 0.585 |
| 1 | 0.567 | 0.567 | 0.580 |
| 2 | 0.558 | 0.564 | 0.580 |
| 3 | 0.550 | 0.533 | 0.548 |
| 4 | 0.556 | 0.539 | **0.518** |

Pesos w de v4 elegidos en val: 0.6 / 0.3 / 0.6 / 0.5 / 0.5. Rondan 0.5, así
que ajustar el peso no aporta sobre el promedio simple.

**Correlación entre las probabilidades de A y B** en test: 0.46-0.63 en
crackle y 0.44-0.60 en wheeze, según el fold. Es una correlación moderada:
los dos modelos coinciden en parte, pero se equivocan en ciclos bastante
distintos.

Matriz de confusión OOF del ensamble promedio (filas = real, columnas =
predicho). Entre corchetes, la diagonal de v4:

```
         normal  crackle  wheeze  both
normal     2622      745     220    55     [v4: 2741]
crackle     895      907      37    25     [v4:  816]
wheeze      356      109     337    84     [v4:  314]
both        150      103     163    90     [v4:   60]
```

## 3. Lectura

- **El ensamble separa mejor las clases que cualquiera de los dos por
  separado.** El AUC, que no depende del umbral, sube en las dos etiquetas:
  crackle 0.676 → 0.691, wheeze 0.793 → 0.821 (la mayor mejora individual de
  wheeze desde v3). Suben la sensibilidad (0.368 → 0.417) y los F1
  (crackle 0.464 → 0.503, wheeze 0.475 → 0.543). "both" pasa de 60 a 90
  aciertos. Con la correlación moderada, confirma que **los features
  preentrenados aportan información que v4 no captura, y viceversa**.
- **Pero el score ICBHI casi no se mueve** (0.559 → 0.562), y el desvío
  empeora (0.029). Mejora en 3 folds, empata en 1 y **cae en el fold 4**
  (0.518). La mejora en separabilidad no llega al score final porque el
  score depende de los **umbrales**, calibrados sobre solo 16 pacientes de
  val. El ensamble ordena mejor los ciclos, pero el umbral elegido en val no
  se traslada bien a test en todos los folds. Es el mismo problema que ya
  apareció en `avances/13` §5, ahora como cuello de botella.
- **Equipo y edad**: sin cambios de fondo. Littmann 3200 sigue siendo el
  peor grupo (0.514). La sensibilidad pediátrica sigue muy baja (0.137).

## 4. Conclusión

1. **El ensamble confirma que vale la pena el fine-tuning (etapa 2).** Con
   la red congelada, lo preentrenado ya aporta información complementaria
   (+0.03 de AUC en wheeze al combinar). Adaptar esas features a nuestros
   ciclos es la vía con más margen.
2. **Aparece un segundo cuello de botella: la calibración de umbrales.**
   Mientras los umbrales se elijan sobre 16 pacientes, mejoras reales en AUC
   pueden no verse en el score ICBHI. Vale la pena atacarlo, por ejemplo
   calibrando sobre más datos (validación cruzada interna dentro del train
   de cada fold) o con un criterio más suave que el máximo de una grilla.
   Aplica a cualquier modelo, incluido el que salga del fine-tuning.
3. **El ensamble no se adopta como versión vigente**: agrega complejidad
   para una mejora de score (+0.003) dentro del ruido. **v4 sigue siendo la
   referencia**, y el ensamble queda como evidencia de complementariedad.

Reportes: `reports/pulmonary_adventitious/v6_ensemble_v4_cnn6/`
(`summary.csv`, `per_fold.csv`, `correlation_v4_cnn6.csv`,
`metrics_by_*.csv`).
