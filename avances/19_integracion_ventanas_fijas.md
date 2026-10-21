# Etapa 3 — Integración del pipeline con ventanas fijas: primeros resultados

Sigue el plan de `avances/18` (opción "3a": ramas con ventanas fijas, solo
al predecir, sin reentrenar). Fecha: 2026-10-20.

## 1. Qué se hizo

Dos scripts nuevos en `desarrollo/src/integration/`:

- **`window_inference.py`**: corta el audio crudo en ventanas fijas, con el
  mismo preprocesamiento que vio cada modelo al entrenar, y guarda la
  probabilidad de cada ventana en `reports/integration/window_probs/`.
  - gate v5: 5 s, paso 2.5 s (el corte de `make_gate_windows.py`), las 4083
    grabaciones;
  - murmullo: **2.5 s**, paso 1.25 s, checkpoints k-fold;
  - pulmonar: **4 s**, paso 2 s, CNN6 variante B.
  - Si la grilla no llega al final de la grabación, la última ventana se
    alinea al final, en lugar de rellenar con ceros.
  - Cada grabación de las ramas pasa por dos modelos: el del fold donde su
    paciente es validación (para calibrar) y el del fold donde es test
    (para evaluar).
  - Además corre la "segmentación oráculo" (ventanas de 4 latidos de los
    `.tsv` y ciclos de los `.txt` de ICBHI) con los mismos modelos.
- **`evaluate_integration.py`**: agrega de ventana a grabación, calibra en
  validación, evalúa la capa 2, el oráculo y la capa 1. No corre modelos;
  tarda segundos.

Inferencia completa en CPU: unos 50 min en total (gate 4.5 min, murmullo
~8, pulmonar ~16, oráculo pulmonar ~15, oráculo murmullo ~12).

### Chequeos de que se cargó todo bien

- Gate: accuracy por ventana en test **0.9753**, igual al reporte de v5.
- Murmullo, oráculo por ventana: F2 medio **0.739** contra 0.737 de
  `per_fold.csv`. La diferencia viene de procesar cada ventana sola, sin el
  relleno del batch.
- Pulmonar, oráculo por ciclo: reproduce `oof_predictions.csv` con un error
  menor a 0.002, **una vez corregido el relleno** (ver §1.1).

### 1.1 Hallazgo: el relleno del batch cambiaba las predicciones pulmonares

En Colab, `predict()` rellena cada batch con silencio (-100 dB) hasta el
ciclo más largo del batch. El pooling enmascara el relleno, pero las
convoluciones lo ven en el borde. Así, **la probabilidad de un ciclo corto
dependía de con quién compartía batch**: procesado solo cambia hasta 0.18
(diferencia media 0.03). Al entrenar, un batch de 64 ciclos casi siempre
trae uno de ≥ 4 s, así que el modelo vio los ciclos cortos rellenados a 400
frames. Por eso la inferencia ahora **rellena siempre a 400 frames**: con
eso se reproduce el OOF y el resultado ya no depende del batch. Las
ventanas fijas de 4 s no tienen este problema, porque ya miden 400 frames.

## 2. Agregación ventana → grabación

Hay cuatro reglas candidatas: `mean`, `p75` (percentil 75), `max` y `frac`
(proporción de ventanas por encima del umbral por ventana de cada fold).
En cada fold, el umbral de grabación se calibra en los pacientes de
validación: máximo F2 en murmullo, máximo score ICBHI (los dos umbrales en
conjunto) en pulmonar. `auto` elige la mejor regla de validación en cada
fold.

- **Murmullo**: `max` vuelve a ser la peor (F2 0.668), como en
  `avances/10` §14. `mean`, `p75`, `frac` y `auto` quedan todas en 0.72-0.73.
  `auto` eligió `frac` en 3 folds, `mean` en 1 y `p75` en 1.
- **Pulmonar**: no hay una regla claramente mejor (score 0.49-0.53) y la
  elección cambia de fold a fold. Con 16 pacientes de validación por fold,
  la calibración por grabación es inestable (Sp entre folds ± 0.14-0.20).

## 3. Capa 2 — ramas con ventanas fijas sobre todos los pacientes

Media ± desvío en 5 folds, por grabación (regla `auto`).

**Murmullo** (942 pacientes, 3163 grabaciones):

| | Precisión | Recall | F1 | F2 | AUC |
|---|---|---|---|---|---|
| Ventanas fijas 2.5 s | 0.63 ± 0.13 | 0.76 ± 0.05 | 0.68 ± 0.06 | **0.725 ± 0.017** | 0.899 |
| Oráculo 4 latidos | 0.66 ± 0.08 | 0.76 ± 0.05 | 0.71 ± 0.03 | **0.738 ± 0.023** | 0.905 |

A nivel paciente (positivo si alguna grabación es positiva, OOF): recall
0.82, precisión 0.47, F2 0.71. Por edad, el resultado es parejo entre
lactantes, niños y adolescentes (F2 0.73-0.74). Los neonatos (6 pacientes)
no tienen resultados útiles.

**Pulmonar** (126 pacientes, 920 grabaciones; etiqueta por grabación
`has_crackles` / `has_wheezes`, 4 clases):

| | Se | Sp | Score ICBHI | AUC crackle | AUC wheeze |
|---|---|---|---|---|---|
| Ventanas fijas 4 s | 0.35 ± 0.08 | 0.68 ± 0.14 | **0.516 ± 0.053** | 0.656 | 0.808 |
| Oráculo ciclos | 0.31 ± 0.14 | 0.68 ± 0.23 | **0.494 ± 0.051** | 0.658 | 0.771 |

- El score por grabación (~0.52) **no es comparable** con el 0.572 por
  ciclo de `avances/17`. Es otra tarea: hay que acertar la combinación de
  la grabación entera, y el 22% de las grabaciones tiene ambos sonidos.
- Por equipo, el más débil es el AKG C417L (683 grabaciones, score 0.50).
  Le siguen Meditron (8 pacientes, 0.41) y Littmann Classic (0.59).
- **Pediátricos: Se 0.125** (Sp 0.955). Se confirma la limitación conocida
  (`avances/18` §2).

## 4. Ventanas fijas vs. segmentación oráculo

Es el resultado para la tesis: **no tener las anotaciones cuesta poco**.

- Murmullo: F2 0.725 contra 0.738 (−0.013, dentro del desvío entre folds),
  AUC 0.899 contra 0.905. Las 3162 grabaciones con `.tsv` son casi todas
  (3163), así que la comparación es sobre las mismas grabaciones.
- Pulmonar: las ventanas fijas **no pierden nada**: score 0.516 contra
  0.494 y AUC iguales o mejores. Tiene sentido: el pooling por máximo de la
  CNN6 encuentra el evento esté donde esté dentro de los 4 s, y las
  ventanas cubren también los tramos que las anotaciones dejan afuera.

Conclusión: **no hace falta pasar a la opción "3b"** (reentrenar con
ventanas fijas) por pérdida de rendimiento.

## 5. Capa 1 — sistema completo sobre el test del gate

161 pacientes (142 cardíacos, 19 pulmonares), 591 grabaciones. Gate por
grabación: probabilidad media de sus ventanas > 0.5.

- Gate: **0.9865 de accuracy por grabación** (0.9753 por ventana). Con
  votación por mayoría da 0.9814, así que se queda la media. Comete 8
  errores: 3 cardíacas mandadas a pulmonar y 5 pulmonares mandadas a
  cardíaco. 3 de las 5 pulmonares son de niños de 3, 3 y 10 años, en los que
  el corazón se escucha más en el tórax.
- Pérdida por la cascada, sobre las mismas grabaciones:

| Rama | Ruteo perfecto | Sistema (gate v5) |
|---|---|---|
| Murmullo, F2 | 0.643 | 0.632 |
| Pulmonar, score ICBHI | 0.667 | 0.638 |
| Accuracy en 6 clases (todo el sistema) | 0.812 | 0.804 |

**El gate cuesta ~1 punto en murmullo y ~3 en pulmonar.** Lo de pulmonar
son 5 grabaciones de 116, así que no alcanza para conclusiones firmes. Como
se esperaba, la pérdida por el gate es chica, así que **por ahora no hace
falta el gate con 5 folds** (`avances/18` §4.3). El F2 de murmullo en
estos 142 pacientes (0.64) queda por debajo de la media de la capa 2
(0.725). Es variación del subconjunto, no del gate: con ruteo perfecto da
lo mismo.

Matriz de confusión del sistema: `reports/integration/layer1_confusion_system.{csv,png}`.

## 6. Archivos

`reports/integration/`:
- `window_probs/{gate,murmur,pulmonary,murmur_oracle,pulmonary_oracle}.csv`
  (probabilidades por ventana o ciclo);
- `murmur_{per_fold,summary}.csv`, `pulmonary_{per_fold,summary}.csv`
  (todas las reglas, más ventanas fijas vs. oráculo);
- `murmur_recording_oof.csv`, `pulmonary_recording_oof.csv`
  (predicción por grabación, regla `auto`);
- desgloses `murmur_by_{age_category,auscultation_location}.csv` y
  `pulmonary_by_{equipment,age_group}.csv`, más el nivel paciente
  `murmur_patient_level.csv`;
- `layer1_{system,branch_metrics,recordings,confusion_system}.*`;
- `check_murmur_oracle_window_level.csv`, `evaluate_integration_run.txt`
  (salida de consola).

Para regenerar:
```
python src/integration/window_inference.py --branch <gate|murmur|pulmonary|murmur_oracle|pulmonary_oracle>
python src/integration/evaluate_integration.py
```

## 7. Próximos pasos posibles

1. Sensibilidad a la duración de ventana: murmullo con 2.0, 3.0 y 5 s, y
   pulmonar con 3 y 6 s (`--murmur-sec` / `--pulmonary-sec` + `--suffix`).
2. Calibración pulmonar por grabación más estable: elegir una sola regla
   fija para todos los folds, o calibrar con validación + train del fold.
3. Validación externa de punta a punta con HLS-CMDS.
4. Líneas de `avances/18` §5: modelo único sin gate, gate con 5 folds para
   los números definitivos, ajuste fino de cada modelo (en especial la
   sensibilidad pediátrica pulmonar).
