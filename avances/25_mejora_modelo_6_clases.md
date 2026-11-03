# Mejora del modelo de 6 clases: red ancha y doble resolución

Continúa `avances/24` §9.1. Se buscó dónde pierde el modelo integrado, se
probaron dos variantes con un filtro por validación y la ganadora se entrenó
y evaluó con los 5 folds. Fecha: 2026-10-30.

Resumen en cuatro líneas:

- **`cs_wide` (95k parámetros) es la nueva referencia**: 0.776 de accuracy
  de 6 clases, contra 0.766 de `cs_base` (+0.96 puntos, IC95 [+0.13, +1.77]).
- **Mejora las dos tareas y en los 5 folds**: F2 de murmullo 0.789 (antes
  0.772), score pulmonar 0.594 (antes 0.573), Se 0.33 (antes 0.27).
- **Alcanza a las ramas dedicadas chicas** y queda por encima del pipeline
  separado (0.771), aunque esa diferencia no es significativa.
- **La doble resolución (`cs_res2`) y los batches balanceados
  (`cs_wide_bal`) no pasaron el filtro.**

## 1. Diagnóstico de `cs_base`

Sobre las 5920 grabaciones de SPRSound (predicciones de test de cada fold).
De las 1702 anormales:

| Resultado | Grabaciones |
|---|---|
| Clase exacta | 26.5% (la Se de 0.27) |
| Anomalía detectada, clase equivocada | 18% |
| Mandadas a normal | 55% |

- **Mover umbrales no alcanza.** Al bajarlos, la detección llega a 0.70,
  pero Se se estanca en 0.36, Sp cae a 0.63 y el score baja a 0.50. Se miró
  sobre test solo para ver la forma de la curva; no eligió nada.
- **La accuracy de 6 clases premia decir "normal"**: el 71% de SPRSound es
  normal, y en ese barrido, al subir Se, la accuracy de 4 clases cae de ~0.70
  a 0.55.
- **Crackle es la tarea más débil**, y la que más se va a normal.

## 2. Protocolo

Decidido con el usuario antes de entrenar:

- **Métrica de decisión**: media de F2 de murmullo y score pulmonar por
  grabación, coherente con el criterio B. La accuracy de 6 clases se reporta
  al lado y no debe empeorar de forma significativa.
- **Filtro en dos fases**: folds 0 y 3 solo con validación, y a 5 folds solo
  las finalistas.
- **Variantes elegidas**: `cs_wide` y `cs_res2`. Con `cs_wide` como
  referencia, después se probaron los batches balanceados (`cs_wide_bal`,
  §4.1). La normalización robusta queda sin probar.
- **CNN6**: después, en Colab, con la mejor receta.
- Se mantienen los folds, el criterio B fijado antes del test y A como
  sensibilidad.

## 3. Implementación (`src/unified_model/`)

- `model.py` y `run_kfold.py`: `--n-fft2` agrega un segundo canal de entrada
  con otro n_fft y el mismo hop, cada canal con su z-score (24.4k parámetros).
- `run_kfold.py --balance-classes`: en cada época, cada dominio es la mitad
  de lo que se ve y, dentro de él, cada clase pesa lo mismo (1/4 por clase
  cardíaca, 1/8 por clase pulmonar), con reposición. Va con
  `--class-weights none` (pérdida sin pesos), para no balancear dos veces.
- `evaluate_full_recordings.infer(..., folds)`: inferencia en algunos folds,
  con caché propia.
- `filter_variants.py` (nuevo): métrica de decisión en val, por grabación. El
  valor es optimista (calibra y mide sobre lo mismo), pero igual para todas
  las variantes.
- `evaluate_circor_spr.py --base-run`: compara `--run` con otra corrida de 6
  clases con la diferencia apareada, en
  `reports/unified_model/circor_spr/<run>_vs_<base>/`, sin pisar las tablas
  de `cs_base`.

`cs_res2` usa n_fft 512 + 256, hop 80 y el último bloque sin reducir tiempo:
la configuración con la que crackle mejoró en `avances/22` §3.

## 4. Filtro por validación (folds 0 y 3)

| Variante | Parámetros | F2 murmullo | Score pulmonar | Decisión | Por fold |
|---|---|---|---|---|---|
| `cs_base` | 24k | 0.828 | 0.586 | 0.707 | 0.707 / 0.706 |
| **`cs_wide`** | 95k | **0.845** | **0.606** | **0.726** | 0.728 / 0.723 |
| `cs_res2` | 24k | 0.826 | 0.592 | 0.709 | 0.716 / 0.703 |

- **`cs_wide` mejora las dos tareas en los dos folds.**
- **`cs_res2` no es consistente**: +0.009 en el fold 0 y −0.003 en el 3. El
  conflicto de resolución de `avances/22` no se resolvió con dos canales. Sus
  folds 1, 2 y 4 no se entrenaron.
- No se probó la combinación de las dos: `cs_res2` no aportó por sí sola.

### 4.1 Batches balanceados sobre `cs_wide` (2026-11-02/03)

Regla fijada antes de entrenar: pasa a 5 folds si supera a `cs_wide` en la
media de la métrica de decisión y no baja en ningún fold.

| Variante | F2 murmullo | Score pulmonar | Decisión | Por fold |
|---|---|---|---|---|
| `cs_wide` | 0.845 | 0.606 | 0.7255 | 0.728 / 0.723 |
| `cs_wide_bal` | 0.849 | 0.603 | 0.7261 | 0.723 / 0.729 |

- **No pasa**: la media es la misma (+0.0006), el fold 0 baja (−0.005) y el
  3 sube (+0.007). Es ruido entre folds, no una mejora.
- **No ataca la sensibilidad pulmonar**: el score pulmonar no sube. Por
  ventana, en val, crackle y wheeze bajan en el fold 0 (0.690 / 0.868 contra
  0.703 / 0.897) y suben apenas en el 3. La accuracy balanceada de 6 clases
  por ventana baja en los dos folds (0.531 y 0.568 contra 0.561 y 0.596).
- **Interpretación**: los pesos inv ya igualaban la contribución de cada
  clase a la pérdida. El muestreo cambia cómo se reparten las clases en cada
  batch, pero no agrega información: repite las ventanas raras (la clase
  "both" se ve ~25 veces más que en el índice) y la red corta antes (mejores
  épocas 8 y 12, contra 13 y 20).
- Tardó ~6.5 min por época con dos procesos en paralelo (~2 h por fold).

## 5. Resultados de `cs_wide` con 5 folds

Mejores épocas (B): 13, 18, 8, 20, 11. Selección fijada por validación antes
de evaluar por grabación en test.

### 5.1 Por ventana (AUC en test, media de 5 folds)

| Modelo | Murmullo | Crackle | Wheeze | Dominio |
|---|---|---|---|---|
| **`cs_wide` (B)** | **0.917** ± 0.021 | **0.758** ± 0.041 | 0.852 ± 0.038 | 0.999 |
| `cs_wide` (A) | 0.905 | 0.758 | 0.857 | — |
| `cs_base` (B) | 0.903 ± 0.025 | 0.726 ± 0.055 | 0.860 ± 0.020 | 0.999 |
| `abl_card` / `spr_inv` | 0.913 ± 0.021 | 0.741 ± 0.049 | 0.869 ± 0.024 | — |

### 5.2 Capa 2: por grabación

| Modelo | F2 murm. | AUC murm. | Precisión | Score pulm. (Se / Sp) | AUC crk / whz | AUC normal-anormal |
|---|---|---|---|---|---|---|
| **`cs_wide` (B)** | **0.789** ± 0.050 | **0.933** | 0.685 | **0.594** ± 0.012 (0.33 / 0.86) | 0.686 / 0.797 | **0.793** ± 0.035 |
| `cs_wide` (A) | 0.782 | 0.924 | 0.640 | 0.589 (0.31 / 0.87) | 0.685 / 0.791 | 0.795 |
| `cs_base` (B) | 0.772 ± 0.044 | 0.919 | 0.605 | 0.573 ± 0.027 (0.27 / 0.88) | 0.658 / 0.803 | 0.767 ± 0.037 |
| `abl_card` / `spr_inv` | 0.794 ± 0.034 | 0.927 | 0.675 | 0.588 ± 0.020 (0.31 / 0.87) | 0.686 / 0.783 | 0.767 ± 0.035 |

- **Métrica de decisión en test**: 0.691 contra 0.673 de `cs_base`, y mejora
  en los 5 folds (0.661 / 0.724 / 0.693 / 0.709 / 0.670 contra 0.659 / 0.703
  / 0.679 / 0.692 / 0.631).
- **Murmullo**: recupera la precisión perdida (0.685 contra 0.605) con el
  mismo recall (0.83). El desvío entre folds de la precisión sigue alto
  (±0.105).
- **Pulmonar**: Se sube 0.066 y Sp baja 0.025. Crackle por grabación mejora
  (+0.028) y wheeze queda parecido (−0.006).
- **Las ramas dedicadas son de 24k**: `cs_wide` tiene casi el doble de
  parámetros que las dos juntas. "Alcanza a las ramas" vale frente a esas
  ramas, no como red igual de grande.

### 5.3 Capa 1: sistema de punta a punta

Todos los pacientes (9083 grabaciones, 1796 pacientes):

| | `cs_wide` (B) | `cs_wide` (A) | `cs_base` (B) | Pipeline separado |
|---|---|---|---|---|
| Accuracy de dominio | 0.9992 (7 errores) | 0.9992 | 0.9992 | 0.9912 |
| **Accuracy de 6 clases** | **0.776** | 0.772 | 0.766 | 0.771 |
| Con dominio perfecto | 0.777 | 0.773 | 0.767 | 0.778 |
| F2 murmullo del sistema | 0.793 | 0.784 | 0.774 | 0.795 |
| Score pulmonar del sistema | 0.595 | 0.589 | 0.572 | 0.579 |

Diferencias apareadas (bootstrap por paciente, 2000 repeticiones):

| Comparación | Diferencia | IC95 | Solo `cs_wide` / solo el otro |
|---|---|---|---|
| `cs_wide` − `cs_base` | **+0.96 puntos** | **[+0.13, +1.77]** | 506 / 419 |
| `cs_wide` − pipeline separado | +0.52 puntos | [−0.25, +1.29] | 483 / 436 |

Aciertos por clase pulmonar anormal (y mandadas a normal), `cs_wide` contra
`cs_base`:

| Clase | `cs_wide` | `cs_base` |
|---|---|---|
| Crackle | 31% (56%) | 25% (63%) |
| Wheeze | 41% (41%) | 31% (43%) |
| Both | 28% (31%) | 24% (42%) |

- **Mejora significativa contra `cs_base`**, y la accuracy sube junto con
  Se, a pesar de que premia decir "normal".
- **Contra el pipeline separado es empate estadístico**, pero `cs_wide` es
  un solo modelo y ya no queda debajo en la accuracy del sistema.
- El IC95 no corrige por haber probado dos variantes, aunque solo una llegó
  a test.

### 5.4 Figuras

`plot_circor_spr.py --run cs_wide --base-run cs_base` →
`reports/unified_model/circor_spr/cs_wide_vs_cs_base/figures/`:

- `recording_metrics.png`: métricas por grabación de `cs_wide`, `cs_base` y
  las ablaciones, con la media y el valor de cada fold. El eje arranca en 0.4,
  por eso va con puntos y no con barras. Los rangos entre folds se superponen:
  la evidencia es la mejora apareada fold a fold, no la distancia entre medias.
- `class_accuracy.png` (nueva): aciertos del sistema por clase contra
  `cs_base` y el pipeline separado. `cs_wide` es igual o mejor que el pipeline
  en las clases anormales (crackle 0.31 contra 0.28, wheeze 0.41 contra 0.38,
  both 0.28 contra 0.23). Contra `cs_base` cede en pulmonar normal (0.85
  contra 0.88).
- `confusion_system.png`: capa 1 por grabación. La columna con más error
  sigue siendo "pulm: normal".
- `confusion_windows.png`: 6 clases por ventana.
- `roc_windows.png`: ROC por ventana de los tres modelos. En crackle
  `cs_wide` queda arriba; en wheeze, apenas debajo, y con la banda entre folds
  más ancha.
- `learning_curves.png`: la loss de train sigue bajando mientras la de val se
  mantiene ruidosa alrededor de 1.1, y los AUC se estabilizan hacia la época
  10. Se entrena de más sin ganar en val: lo marca el early stopping.
- `domain_separation.png`: los dominios quedan separados; los 7 errores
  caen cerca de 1.

## 6. Conclusiones

1. **Al modelo integrado le faltaba capacidad**: con cuatro veces más
   parámetros mejoran las dos tareas sin perder separación de dominio.
2. **`cs_wide` (B) reemplaza a `cs_base` como referencia.**
3. **El cuello de botella pulmonar sigue**: 56% de los crackle y 41% de los
   wheeze todavía se van a normal.
4. **El modelo sin gate ya no pierde contra el pipeline separado** en capa 1,
   y los dos están en empate estadístico.
5. **El costo es tamaño y tiempo**: 95k contra 24k parámetros y un
   entrenamiento varias veces más lento. Esto le da sentido a la etapa de
   reducción de parámetros.

## 7. Comandos

Desde `desarrollo/src`:

    python unified_model/run_kfold.py --data circor_spr --width wide --feature logmel --pooling avg_max --class-weights inv --select select_domain --run-name cs_wide
    python unified_model/run_kfold.py --data circor_spr --feature logmel --pooling avg_max --class-weights inv --select select_domain --n-fft 512 --n-fft2 256 --hop 80 --time-keep-blocks 1 --run-name cs_res2 --folds 0 3
    python unified_model/run_kfold.py --data circor_spr --width wide --feature logmel --pooling avg_max --class-weights none --balance-classes --select select_domain --run-name cs_wide_bal --folds 0 3
    python unified_model/filter_variants.py --runs cs_base cs_wide cs_res2 cs_wide_bal
    python unified_model/evaluate_circor_spr.py --run cs_wide --base-run cs_base
    python unified_model/plot_circor_spr.py --run cs_wide --base-run cs_base

Con cuatro procesos en paralelo (3 hilos cada uno), cada época tardó ~11 min
en las dos variantes. Checkpoints en `src/unified_model/checkpoints_cs_wide/`
y `checkpoints_cs_res2/` (no versionados). Logs en
`reports/unified_model/logs/`. La comparación contra el pipeline separado
está en `reports/unified_model/circor_spr/cs_wide_vs_cs_base/layer1_paired_difference_pipeline.csv`.

## 8. Pendiente y traspaso (en orden)

1. ~~Batches balanceados~~: probado, no pasa (§4.1). Con tres variantes
   descartadas, el ajuste de receta sobre la CNN chica da poco. La
   normalización robusta queda opcional.
2. **CNN6 en Colab** con la mejor receta, como techo de desempeño.
3. **Reducción de parámetros** de la variante final (curva parámetros contra
   desempeño).
4. Validación externa: ICBHI completo y HLS, con el modelo final
   (`avances/23` §8.3).
5. Lo que seguía pendiente de `avances/24` §9: escuchar los pacientes con
   error de dominio y actualizar la memoria técnica.

## 9. Estado del repositorio

Sin commitear al cerrar:

- Código: `model.py`, `run_kfold.py`, `evaluate_full_recordings.py`,
  `evaluate_circor_spr.py`, `plot_circor_spr.py` (modificados),
  `filter_variants.py` (nuevo).
- Reportes: `reports/unified_model/runs/{cs_wide,cs_res2,cs_wide_bal}/`,
  `reports/unified_model/circor_spr/cs_wide_vs_cs_base/`,
  `filter_folds03.csv` y las cachés nuevas de
  `reports/sprsound/full_recording/`.
- Este documento.
