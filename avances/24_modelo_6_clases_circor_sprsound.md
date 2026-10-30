# Modelo de 6 clases con CirCor + SPRSound

Continúa `avances/23` §8 (traspaso). Se rehízo el modelo único sin gate
(objetivo primario de la tesis) con **CirCor para cardíaco y SPRSound para
pulmonar**, con la receta de `avances/22`, y se lo evaluó por ventana y por
grabación frente a las ablaciones por dominio con la misma red. Después se
entrenó un gate nuevo con CirCor + SPRSound y se comparó el pipeline
separado contra el modelo de 6 clases en capa 1 (§5). Fecha: 2026-10-29/30.

Resumen en cinco líneas:

- **Compartir la red cuesta poco**: frente a las ablaciones (`abl_card`,
  `spr_inv`), el modelo de 6 clases pierde entre 0.009 y 0.015 de AUC por
  ventana y ~0.02 por grabación (F2 murmullo 0.772 contra 0.794, score
  pulmonar 0.573 contra 0.588), dentro del ruido entre folds.
- **El dominio se separa casi perfecto**: 99.92% por grabación (7 errores
  en 9083, todos de SPRSound hacia cardíaco).
- **Sistema de punta a punta: 76.6% de accuracy de 6 clases** por grabación
  (1796 pacientes). Lo que falta está en las clases pulmonares anormales,
  que el modelo manda mucho a normal (igual que la ablación).
- **El checkpoint se elige con el criterio B** (cada dominio pesa lo mismo),
  decidido antes de ver el test. El criterio A da prácticamente lo mismo.
- **El pipeline separado empata con el modelo de 6 clases** en capa 1
  (0.771 contra 0.766, IC95 de la diferencia [−1.2, +0.4] puntos). Gana en
  las tareas, pero el gate nuevo erra más el dominio (80 errores contra 7).
  **Se sigue con el modelo integrado** (decisión con el usuario).

## 1. Criterio de selección del checkpoint

En `avances/22` el checkpoint se elegía con la media de los tres AUC de val
(murmullo, crackle, wheeze). Con el usuario se revisó qué implica:

| | A: `select` | B: `select_domain` |
|---|---|---|
| Fórmula | (AUC murm + AUC crk + AUC whz) / 3 | (AUC murm + (AUC crk + AUC whz) / 2) / 2 |
| Peso cardíaco / pulmonar | 1/3 / 2/3 | 1/2 / 1/2 |

- Los AUC se calculan dentro de cada dominio, así que el desbalance de
  ventanas entre bases no influye en ninguno de los dos.
- A le da el doble de peso a lo pulmonar solo porque tiene dos AUC; es una
  consecuencia de cómo se cuentan las tareas, no una decisión.
- **Decisión (con el usuario): B es el criterio principal.** El objetivo es
  un clasificador de patologías que detecte tan bien unas como otras, y la
  comparación con etapas anteriores pesa poco.
- **El criterio se fijó antes de ver el test.** Elegirlo después mirando el
  test sería fuga, y elegirlo por val es circular (cada criterio es el
  máximo de sí mismo en val). A queda como **análisis de sensibilidad**: se
  guarda su checkpoint y se evalúa igual.

## 2. Implementación (`src/unified_model/`)

- `run_kfold.py`:
  - **`--data circor_spr`**: dos fuentes que entrenan. Ventanas cardíacas
    del índice unificado (folds unificados: el test cardíaco es el de
    `avances/22`) y ventanas de SPRSound (sus folds).
  - Las fuentes llevan un **sufijo de rol**: las del mismo sufijo se
    evalúan juntas. En `circor_spr` las dos van sin sufijo, con los roles
    `val`/`test` de siempre; en `joint`, ICBHI sigue con `_icbhi`.
  - **`--select select|select_domain`** elige el criterio principal
    (default `select`, para que los runs anteriores se reproduzcan igual).
  - **Se guarda el mejor checkpoint de cada criterio**: `fold{i}_best.pt`
    (principal) y `fold{i}_best_<otro>.pt`, con sus
    `window_probs_fold{i}[_<otro>].csv`. En `per_fold.csv`, las columnas
    del otro criterio llevan el prefijo `<otro>_`.
  - **Early stopping**: corta cuando ningún criterio mejoró en `patience`
    épocas.
- `evaluate_unified.load_window_probs(run_dir, criterion)`: filtra los
  archivos por criterio (sin esto, el glob mezclaba los dos).
- `evaluate_full_recordings.infer(..., criterion, cols)`: inferencia sobre
  grabaciones completas con el checkpoint de cualquier criterio.
- `evaluate_circor_spr.py`: evaluación por grabación (§4).
- `plot_circor_spr.py`: figuras (§5).

## 3. Entrenamiento y métricas por ventana

`cs_base`: base (24k parámetros), log-Mel, pooling promedio + máximo, pesos
inv sobre el train conjunto (~41k ventanas por fold: CirCor 34.5k y
SPRSound 26k en total). Mejores épocas (B): 14, 18, 9, 21, 26. ~2 min por
época en CPU.

AUC por ventana en test, media ± desvío en 5 folds:

| Modelo | Murmullo | Crackle | Wheeze | Dominio |
|---|---|---|---|---|
| **`cs_base` (B)** | **0.903** ± 0.025 | **0.726** ± 0.055 | **0.860** ± 0.020 | 0.999 |
| `cs_base` (A) | 0.900 | 0.727 | 0.863 | 0.999 |
| `abl_card` (solo CirCor) | 0.913 ± 0.021 | — | — | — |
| `spr_inv` (solo SPRSound) | — | 0.741 ± 0.049 | 0.869 ± 0.024 | — |

- Los tests son los mismos pacientes que los de las ablaciones: la
  comparación es apareada.
- **A y B eligieron la misma época en 4 de 5 folds.** En el fold 1, B
  eligió la 18 y A la 10 (+0.016 murmullo, −0.005/−0.012 crackle/wheeze).
- Como referencia, `v1_base` (CirCor + ICBHI, `avances/22`) daba 0.887 /
  0.565 / 0.699; las tareas pulmonares no son comparables (otra base).

## 4. Evaluación por grabación

Script: `evaluate_circor_spr.py` → `reports/unified_model/circor_spr/`.
Mismo procedimiento que `evaluate_unified.py` (reglas de agregación,
estrategia `pooled_fixed`, umbrales por ventana calibrados en val) con
**todas las ventanas de cada grabación**:

- CirCor: el índice ya tiene todas las ventanas de cada grabación.
- SPRSound: las grabaciones completas pasan por el checkpoint de cada fold
  (`avances/23` §4.1).

Las ablaciones se evalúan con el mismo código. Control: `spr_inv` da 0.767
de AUC normal-anormal, igual que en `avances/23`, y `abl_card` coincide con
`reports/unified_model/ablation/`.

### 4.1 Capa 2: cada tarea por grabación

| Modelo | F2 murm. | AUC murm. | Precisión murm. | Score pulm. (Se / Sp) | AUC crk / whz | AUC normal-anormal |
|---|---|---|---|---|---|---|
| **`cs_base` (B)** | **0.772** ± 0.044 | **0.919** | 0.605 | **0.573** (0.27 / 0.88) | 0.658 / 0.803 | **0.767** ± 0.037 |
| `cs_base` (A) | 0.774 | 0.918 | 0.610 | 0.573 (0.26 / 0.89) | 0.653 / 0.806 | 0.767 |
| `abl_card` / `spr_inv` | 0.794 ± 0.034 | 0.927 | 0.675 | 0.588 (0.31 / 0.87) | 0.686 / 0.783 | 0.767 ± 0.035 |

- **Murmullo**: −0.022 de F2, casi todo por **precisión** (0.605 contra
  0.675; el recall es igual, 0.83). En val la diferencia es más clara (0.773
  contra 0.811) que en test.
- **Pulmonar**: −0.015 de score; wheeze por grabación algo mejor (+0.020) y
  crackle algo peor (−0.028).
- **La AUC normal-anormal de SPRSound no pierde nada** (0.767 las dos).
- Las dos tareas bajan en la misma dirección, aunque cada diferencia cae
  dentro del desvío entre folds.

### 4.2 Dominio y sistema de punta a punta (capa 1)

Sobre todos los pacientes (9083 grabaciones, 1796 pacientes). El test del
gate de `avances/22` no aplica: sus pacientes pulmonares eran de ICBHI.

| | `cs_base` (B) | `cs_base` (A) |
|---|---|---|
| Accuracy de dominio | 0.9992 | 0.9992 |
| Errores cardíaco → pulmonar / pulmonar → cardíaco | 0 / 7 | 0 / 7 |
| **Accuracy de 6 clases del sistema** | **0.766** | 0.768 |
| Accuracy de 6 clases con dominio perfecto | 0.767 | 0.769 |
| F2 murmullo del sistema | 0.774 | 0.777 |
| Score pulmonar del sistema | 0.572 | 0.571 |

- Los **7 errores de dominio** son de 3 pacientes de SPRSound, todos del
  fold 2 y de 3-4 años (`40801342`, `41101309`, `65114631`), con P(cardíaco)
  0.77-0.90. Hay que escucharlos: puede que el corazón domine la grabación.
- El sistema pierde sobre todo en las clases pulmonares anormales:
  crackle 25%, wheeze 31% y both 24% de aciertos, la mayoría mandados a
  normal (63%, 43% y 42%). Es el mismo patrón de la ablación pulmonar
  (sensibilidad 0.27-0.31), no un costo de compartir la red.

## 5. Pipeline separado con gate nuevo

### 5.1 Gate CirCor + SPRSound (`src/gate_model/run_kfold_circor_spr.py`)

Misma receta que el gate v5 (`avances/22`): GateCNN (3 bloques conv
16/32/64 + GAP + lineal), MFCC de 20 coeficientes con z-score por ventana,
ventanas de 5 s con salto de 2.5 s (cola con padding si dura ≥ 2.5 s),
aumento de ganancia ±3 dB y ruido, BCE con `pos_weight` = neg/pos, Adam
1e-3, batch 32, 15 épocas, mejor época por F1 de val por ventana.

- **Los folds son los del modelo de 6 clases**: CirCor con los folds
  unificados y SPRSound con los suyos. Así el test de cada paciente es el
  mismo en los dos sistemas y la comparación es apareada.
- El audio sale de las mismas cachés (4 kHz, `audio_ops.preprocess`, el
  preprocesamiento del gate v5). El corte de ventanas se verificó contra
  `make_gate_windows._slice_windows` en largos de borde.
- Datos: CirCor 3163 grabaciones (27,361 ventanas), SPRSound 5920
  (24,445 ventanas). Ninguna grabación dura menos de 2.5 s.
- ~73 s por época en CPU (~1.75 h los 5 folds).

Resultados en test por grabación (media de P(cardíaco) > 0.5):

| Fold | Mejor época | Accuracy | Errores |
|---|---|---|---|
| 0 | 9 | 0.998 | 3 |
| 1 | 2 | 0.999 | 1 |
| 2 | 9 | **0.959** | **75** |
| 3 | 3 | 1.000 | 0 |
| 4 | 1 | 0.999 | 1 |

- **El entrenamiento es inestable**: la loss de train baja a ~0.001 pero el
  accuracy de val salta entre 0.88 y 0.99 de una época a otra. La mejor
  época cae temprano (1-3 en tres folds) y depende del azar.
- **Los 80 errores son todos de SPRSound hacia cardíaco**, de 40
  pacientes, casi todos del fold 2 (75). No se concentran por edad ni por
  ubicación de auscultación.
- **Son errores dudosos**: 61 de 80 tienen P(cardíaco) entre 0.5 y 0.9.
  Apunta a calibración y elección de época, no a que el gate no pueda
  separar los dominios.
- Dos de los pacientes con error de dominio en `cs_base` (`41101309`,
  `65114631`, fold 2) también fallan acá con P(cardíaco) ≈ 1: son los
  candidatos más claros para escuchar.

### 5.2 Capa 1: pipeline separado contra modelo de 6 clases

Pipeline separado = gate `kfold_cs` + `abl_card` + `spr_inv`, con la misma
evaluación por grabación de §4. Todos los pacientes (9083 grabaciones, 1796
pacientes):

| | `cs_base` (B) | Pipeline separado |
|---|---|---|
| Accuracy de dominio | **0.9992** (7 errores) | 0.9912 (80 errores) |
| **Accuracy de 6 clases del sistema** | 0.766 | **0.771** |
| Accuracy de 6 clases con dominio perfecto | 0.767 | 0.778 |
| F2 murmullo del sistema | 0.774 | 0.795 |
| Score pulmonar del sistema | 0.572 | 0.579 |

Diferencia apareada (`layer1_paired_difference.csv`, bootstrap por paciente,
2000 repeticiones): **+0.44 puntos para el pipeline separado, IC95
[−1.2, +0.4]**, que incluye el cero. 434 grabaciones las acierta solo
`cs_base` y 474 solo el pipeline.

- **La ventaja del pipeline viene de las ramas dedicadas, no del gate**:
  con dominio perfecto llegaría a 0.778, y el gate le resta 0.7 puntos.
  El modelo de 6 clases separa el dominio mejor que el gate dedicado.
- Con un gate estabilizado (pocos errores en el fold 2), el pipeline
  quedaría cerca de 0.777: ~1 punto sobre `cs_base`, todavía en el orden
  del ruido.
- **Decisión (con el usuario): no se sigue mejorando el gate.** Se trabaja
  sobre el modelo integrado, que es el objetivo primario, más simple, y ya
  empata. El pipeline separado queda como referencia de comparación.

Salidas en `reports/unified_model/circor_spr/`: `pipeline_kfold_cs/`
(dominio y capa 1 por grabación), la columna del pipeline en
`layer1_system.csv` y `layer1_paired_difference.csv`. El gate en
`reports/gate/runs/kfold_cs/` (historias, probabilidades por ventana,
`per_fold.csv`).

## 6. Figuras

`plot_circor_spr.py` → `reports/unified_model/circor_spr/figures/`:

- `learning_curves.png`: loss, selección B y A y AUC de cada tarea en val,
  por época y fold.
  - La loss de val es ruidosa y no baja mientras los AUC suben: la pérdida
    pesada (inv) es muy sensible a las clases raras; por eso se elige por
    AUC.
  - Crackle es la tarea más inestable entre épocas y entre folds (el fold 1
    está ~0.07 por encima del resto).
- `roc_windows.png`: ROC por ventana (promedio de las ROC de cada fold,
  banda = rango entre folds) contra las ablaciones. Las curvas casi se
  superponen.
  - Si se juntan los folds en una sola ROC, `cs_base` sale mejor en wheeze
    (0.834 contra 0.770): **sus puntajes son más parecidos entre folds**
    que los de `spr_inv`, lo que sirve con umbrales comunes como
    `pooled_fixed`. Pasa lo mismo con la AUC normal-anormal agrupada
    (0.715 contra 0.688). No se usa como métrica principal.
- `confusion_windows.png`: 6 clases por ventana (argmax, test).
- `confusion_system.png`: capa 1 por grabación, con la columna "dominio
  erróneo".
- `recording_metrics.png`: métricas por grabación contra las ablaciones,
  con el valor de cada fold.
- `domain_separation.png`: P(cardíaco) media por grabación según el dominio
  real (escala log).

## 7. Conclusiones

1. **El modelo de 6 clases con CirCor + SPRSound funciona como un sistema
   único**: separa los dominios casi sin error y cada tarea queda a ~0.02
   de su rama dedicada con la misma red.
2. **El costo de compartir la red es chico y no significativo con 5
   folds**, aunque va en la misma dirección en las dos tareas. Por
   grabación, en murmullo se concentra en la precisión.
3. **El cuello de botella es la tarea pulmonar**, no la integración: la
   sensibilidad a las anomalías pulmonares es baja (~0.27) también en la
   ablación.
4. **El criterio de selección no cambia ninguna conclusión** (A ≈ B).
5. **El modelo integrado empata con el pipeline separado** en el sistema
   de punta a punta y separa el dominio mejor que un gate dedicado. Lo que
   el pipeline gana en las tareas lo pierde en el gate. Esto respalda
   seguir con el modelo sin gate y concentrar el trabajo en mejorarlo.

## 8. Comandos

Desde `desarrollo/src`:

    python unified_model/run_kfold.py --data circor_spr --width base --feature logmel --pooling avg_max --class-weights inv --select select_domain --run-name cs_base
    python gate_model/run_kfold_circor_spr.py --run-name kfold_cs
    python unified_model/evaluate_circor_spr.py --gate-run kfold_cs
    python unified_model/plot_circor_spr.py

El entrenamiento tarda ~4 h en CPU (124 épocas en total). Checkpoints en
`src/unified_model/checkpoints_cs_base/` (no versionados). La inferencia
sobre grabaciones completas de SPRSound queda en caché en
`reports/sprsound/full_recording/cs_base[_select]_SPRSound_windows.csv`.
El gate tarda ~1.75 h; checkpoints en `src/gate_model/checkpoints_kfold_cs/`
(no versionados). `evaluate_circor_spr.py` sin `--gate-run` evalúa solo los
modelos de §4.

## 9. Pendiente y traspaso (en orden)

1. **Mejorar el modelo integrado** (próxima etapa). Punto de partida:
   `cs_base` (B), con 0.766 de accuracy de 6 clases en capa 1, F2 de
   murmullo 0.772 y score pulmonar 0.573. Referencias a superar: las ramas
   dedicadas (0.794 / 0.588) y el pipeline separado (0.771).
   - **Lo que más rinde es la tarea pulmonar**: la sensibilidad por
     grabación es ~0.27 y las clases anormales se van a normal (§4.2).
   - **En murmullo, la pérdida frente a `abl_card` es de precisión**
     (0.605 contra 0.675).
   - Candidatos ya anotados: variante ancha (95k), batches balanceados por
     dominio y clase, CNN6, normalización robusta, ramas chicas, menos
     parámetros.
   - Mantener el protocolo: mismos folds, criterio B fijado antes del test,
     A como sensibilidad, evaluación con `evaluate_circor_spr.py`.
2. Validación externa: ICBHI completo y HLS (`avances/23` §8.3).
3. Escuchar los pacientes con error de dominio (§4.2 y §5.1), empezando por
   `41101309` y `65114631`, que fallan en los dos sistemas.
4. Actualizar la memoria técnica (alcance pediátrico, `avances/23` §8.4, y
   la comparación de metodologías de §5).
5. Si se retoma el pipeline separado: estabilizar el gate (learning rate
   menor o scheduler, weight decay, elegir la época por errores por
   grabación en val) y adoptar `abl_card` como rama cardíaca.

## 10. Estado del repositorio

Sin commitear al cerrar:

- Código: `run_kfold.py`, `evaluate_unified.py`,
  `evaluate_full_recordings.py` (modificados), `evaluate_circor_spr.py`,
  `plot_circor_spr.py` y `gate_model/run_kfold_circor_spr.py` (nuevos).
- Reportes: `reports/unified_model/runs/cs_base/`,
  `reports/unified_model/circor_spr/` (con `pipeline_kfold_cs/`),
  `reports/gate/runs/kfold_cs/` y las dos cachés de
  `reports/sprsound/full_recording/`.
- Este documento.
