# CNN6 de 6 clases en Colab: techo de desempeño del modelo sin gate

Continúa `avances/25` §8 paso 2. Se entrenó la CNN6 de PANNs como modelo de
6 clases CirCor + SPRSound, con la receta de `cs_wide`, en Google Colab. Se
probaron dos frontends con el filtro por validación y el ganador se entrenó
y evaluó con los 5 folds. Fecha: 2026-11-04.

Resumen en cuatro líneas:

- **`cnn6_ours64` es el mejor modelo hasta ahora**: 0.800 de accuracy de 6
  clases, contra 0.776 de `cs_wide` (+2.4 puntos, IC95 [+1.4, +3.4]) y 0.771
  del pipeline separado (+3.0 puntos, IC95 [+1.9, +4.1]).
- **Toda la mejora es pulmonar**: score 0.654 (antes 0.594), Se 0.47 (antes
  0.33), y los crackle acertados pasan del 31% al 52%. Murmullo queda igual
  (F2 0.782 contra 0.789), con más precisión y menos recall.
- **El frontend propio a 4 kHz (`ours64`) le gana al de PANNs
  (`panns32k`)** en el filtro, en los dos folds.
- **El costo es tamaño**: 4.57 M de parámetros, unas 48 veces `cs_wide`.

## 1. Protocolo

Decidido con el usuario antes de entrenar:

- **Receta de `cs_wide`**: softmax de 6 clases con pesos `inv`, criterio B
  (`select_domain`) como principal y A (`select`) como sensibilidad,
  augmentation de onda (ganancia ±3 dB y ruido), 30 épocas con paciencia 8.
- **Fine-tuning variante B** de `avances/17`: se entrenan los bloques 3 y 4 y
  `fc1`, con lr 1e-4 en lo preentrenado, 1e-3 en la cabeza (Linear 512 → 6) y
  weight decay 1e-4. SpecAugment de esa variante (máscara de tiempo ≤ 40
  frames y de frecuencia ≤ 6 bandas, solo dentro de las bandas con
  contenido).
- **Dos frontends**, filtro en los folds 0 y 3 solo con validación (la
  métrica de decisión de `avances/25` §2), y a 5 folds solo el ganador.
- Mismos folds, ventanas e índice que `cs_wide`; en SPRSound, las
  grabaciones completas pasan por el checkpoint de cada fold.

## 2. Implementación

### 2.1 Paquete para Colab (`src/unified_model/pack_cnn6_colab.py`)

Un solo `data/colab_cnn6_circor_spr.zip` (1154 MB, sin comprimir), que Colab
copia una vez al disco local:

- `audio_4k_int16.npy`: el audio a 4 kHz de las cachés de `run_kfold`
  (cardíaco de CirCor y todo SPRSound). Se empaqueta audio y no log-Mel, así
  los dos frontends salen del mismo paquete y la augmentation sigue sobre la
  onda.
- `windows.csv` (las 60591 ventanas del índice), `full_windows_spr.csv` (las
  30366 ventanas de todas las grabaciones de SPRSound), `recordings.csv`,
  `folds/fold{0..4}.csv` y los pesos de PANNs.
- Controles: 200 ventanas por dominio, idénticas byte a byte a las de las
  cachés; folds sin pacientes repetidos y que cubren todas las grabaciones.

### 2.2 Entrenamiento (`src/unified_model/run_cnn6_gpu.py`)

- Todo el audio vive en la GPU y cada batch se arma con (inicio, largo); el
  log-Mel se calcula en la GPU. Precisión mixta y estado completo guardado
  por época para retomar si Colab corta.
- **Frontends**:
  - `panns32k`: reproduce el log-Mel de PANNs sin remuestrear a 32 kHz. STFT
    a 4 kHz con n_fft 128 y hop 40, potencia ×64, y las primeras 64 filas
    del banco Mel de 32 kHz / 1024. Contra librosa remuestreando, la
    diferencia es < 0.1 dB en las bandas 0 a 27; solo ~30 de las 64 bandas
    tienen contenido (el resto queda en −100 dB).
  - `ours64`: el de nuestras CNN, a 4 kHz, n_fft 256, hop 40 y 64 bandas Mel
    entre 50 y 2000 Hz. Usa las 64 bandas.
  - Sin filtro pasa altos ni z-score: normaliza la `bn0` de PANNs.
- Salidas con los nombres y columnas de `run_kfold`: `history_fold*.csv`,
  `window_probs_fold*[_select].csv`, `per_fold.csv`, y las predicciones de
  las grabaciones completas de SPRSound.

### 2.3 Vuelta al repositorio (`src/unified_model/import_colab_run.py`)

Copia la corrida bajada de Drive a `reports/unified_model/runs/<run>/` y los
checkpoints a `src/unified_model/checkpoints_<run>/`, y arma las cachés de
`evaluate_full_recordings.infer`. Así `filter_variants.py`,
`evaluate_circor_spr.py` y `plot_circor_spr.py` funcionan sin cambios y sin
reconstruir la CNN6 localmente.

Notebook: `colab_cnn6_circor_spr.ipynb`. Cada época tardó ~55 s en la GPU
de Colab (~28 min por fold de 30 épocas), contra ~11 min por época de
`cs_wide` en CPU.

## 3. Filtro por validación (folds 0 y 3)

| Variante | Parámetros | F2 murmullo | Score pulmonar | Decisión | Por fold |
|---|---|---|---|---|---|
| `cs_wide` | 95k | **0.845** | 0.606 | 0.726 | 0.728 / 0.723 |
| `cnn6_panns32k` | 4.57M | 0.814 | 0.645 | 0.729 | 0.723 / 0.736 |
| **`cnn6_ours64`** | 4.57M | 0.832 | **0.652** | **0.742** | 0.742 / 0.742 |

- **`ours64` gana en los dos folds**, a `cs_wide` y a `panns32k`.
- **`panns32k` empata con `cs_wide`**: mejora en pulmonar, pero pierde 0.03
  de F2 de murmullo, y baja en el fold 0.
- **El murmullo explica la diferencia entre frontends**: por ventana, los
  dos tienen casi el mismo AUC de murmullo (0.925 y 0.923), pero `ours64`
  calibra mejor por grabación. Con más bandas útiles (64 contra ~30) también
  tiene mejor accuracy balanceada de 6 clases por ventana (0.64 contra 0.61).
- **Interpretación**: el banco Mel de PANNs fue diseñado para 32 kHz; a
  4 kHz deja la mitad de las bandas vacías. Lo preentrenado se adapta igual
  a un banco distinto, porque se reentrenan los bloques 3 y 4.

## 4. Resultados de `cnn6_ours64` con 5 folds

Mejores épocas (B): 29, 30, 30, 29, 26; (A): 23, 22, 30, 26, 23. Ningún fold
cortó por early stopping. Selección fijada por validación antes de evaluar
por grabación en test.

### 4.1 Por ventana (AUC en test, media de 5 folds)

| Modelo | Murmullo | Crackle | Wheeze | Dominio | Acc. bal. 6 clases |
|---|---|---|---|---|---|
| **`cnn6_ours64` (B)** | **0.925** ± 0.017 | **0.827** ± 0.036 | **0.908** ± 0.011 | 0.999 | **0.641** |
| `cnn6_ours64` (A) | 0.922 | 0.827 | 0.910 | 0.999 | 0.649 |
| `cs_wide` (B) | 0.917 ± 0.021 | 0.758 ± 0.041 | 0.852 ± 0.038 | 0.999 | 0.576 |

Crackle sube 0.07 y wheeze 0.06 por ventana, con menos desvío entre folds.

### 4.2 Capa 2: por grabación

| Modelo | F2 murm. | AUC murm. | Precisión / Recall | Score pulm. (Se / Sp) | AUC crk / whz | AUC normal-anormal |
|---|---|---|---|---|---|---|
| **`cnn6_ours64` (B)** | 0.782 ± 0.052 | 0.927 | **0.760** / 0.795 | **0.654** ± 0.018 (0.47 / 0.84) | **0.778** / **0.828** | **0.844** ± 0.031 |
| `cnn6_ours64` (A) | 0.790 | 0.926 | 0.740 / 0.809 | 0.647 (0.45 / 0.85) | 0.773 / 0.816 | 0.847 |
| `cs_wide` (B) | 0.789 ± 0.050 | 0.933 | 0.685 / 0.827 | 0.594 ± 0.012 (0.33 / 0.86) | 0.686 / 0.797 | 0.793 ± 0.035 |
| `abl_card` / `spr_inv` | 0.794 ± 0.034 | 0.927 | 0.675 / 0.833 | 0.588 ± 0.020 (0.31 / 0.87) | 0.686 / 0.783 | 0.767 ± 0.035 |

- **Métrica de decisión en test**: 0.718 contra 0.691 de `cs_wide`. Mejora
  en 4 de los 5 folds (0.726 / 0.742 / 0.728 / 0.698 / 0.695 contra 0.661 /
  0.724 / 0.693 / 0.709 / 0.670). En el fold 3 baja por murmullo (F2 0.725
  contra 0.805).
- **Pulmonar**: el score sube +0.060 y mejora en los 5 folds (0.668 / 0.627
  / 0.655 / 0.671 / 0.647 contra 0.596 / 0.595 / 0.580 / 0.613 / 0.587). Se
  sube 0.13 y Sp baja apenas 0.013. Por primera vez el modelo integrado
  supera con claridad a la rama dedicada (`spr_inv`, 0.588).
- **Murmullo**: empate en F2, con otro equilibrio: más precisión (0.760
  contra 0.685) y menos recall (0.795 contra 0.827). El desvío de la
  precisión entre folds sigue alto (±0.124).

### 4.3 Capa 1: sistema de punta a punta

Todos los pacientes (9083 grabaciones, 1796 pacientes):

| | `cnn6_ours64` (B) | `cnn6_ours64` (A) | `cs_wide` (B) | Pipeline separado |
|---|---|---|---|---|
| Accuracy de dominio | 0.9997 (3 errores) | 0.9998 (2) | 0.9992 (7) | 0.9912 (80) |
| **Accuracy de 6 clases** | **0.800** | 0.796 | 0.776 | 0.771 |
| Con dominio perfecto | 0.801 | 0.797 | 0.777 | 0.778 |
| F2 murmullo del sistema | 0.785 | 0.792 | 0.793 | 0.795 |
| Score pulmonar del sistema | **0.655** | 0.645 | 0.595 | 0.579 |

Diferencias apareadas (bootstrap por paciente, 2000 repeticiones):

| Comparación | Diferencia | IC95 | Solo CNN6 / solo el otro |
|---|---|---|---|
| `cnn6_ours64` − `cs_wide` | **+2.43 puntos** | **[+1.36, +3.43]** | 778 / 557 |
| `cnn6_ours64` − pipeline separado | **+2.95 puntos** | **[+1.86, +4.07]** | 815 / 547 |

Aciertos por clase (y, en las pulmonares anormales, mandadas a normal):

| Clase | `cnn6_ours64` | `cs_wide` |
|---|---|---|
| Cardíaco sin murmullo | 95% | 92% |
| Cardíaco con murmullo | 80% | 83% |
| Pulmonar normal | 84% | 86% |
| Crackle | **52%** (38%) | 31% (56%) |
| Wheeze | 44% (34%) | 41% (41%) |
| Both | 28% (17%) | 28% (31%) |

- **Primera mejora significativa contra el pipeline separado**: en
  `avances/25` era empate estadístico.
- **El cuello de botella pulmonar se achica**: las anormales mandadas a
  normal bajan en las tres clases; en crackle, de 56% a 38%.
- **Errores de dominio distintos**: `cs_wide` mandaba 7 grabaciones
  pulmonares de 3 pacientes a cardíaco con P(cardíaco) > 0.95; la CNN6 las
  acierta todas. Sus 3 errores son cardíacos (`CIR_00309`, `CIR_00311`,
  `CIR_00508`), con P(cardíaco) entre 0.25 y 0.50: dudosos, no seguros.
- Las ganancias de Se pulmonar se pagan con 2 puntos de pulmonar normal; la
  accuracy sube igual, a pesar de que premia decir "normal".

### 4.4 Figuras

`plot_circor_spr.py --run cnn6_ours64 --base-run cs_wide` →
`reports/unified_model/circor_spr/cnn6_ours64_vs_cs_wide/figures/`:

- `class_accuracy.png`: aciertos del sistema por clase contra `cs_wide` y
  el pipeline separado. El salto está en crackle (0.52 contra 0.31 y 0.28).
- `learning_curves.png`: la loss de train baja sin parar (0.62 a 0.67 en la
  época 30), mientras la de val queda ruidosa entre 0.8 y 1.1 desde la
  época ~10. Los AUC de murmullo siguen subiendo lento hasta el final; los
  de crackle y wheeze se estabilizan antes. Más épocas darían poco.
- `recording_metrics.png`, `roc_windows.png`, `confusion_system.png`,
  `confusion_windows.png`, `domain_separation.png`: mismas figuras que en
  `avances/25`, con la CNN6.

## 5. Conclusiones

1. **La capacidad sigue siendo lo que más pesa**: igual que de `cs_base` a
   `cs_wide`, la red más grande mejora sin perder separación de dominio.
   Esta vez el salto es casi todo pulmonar.
2. **`cnn6_ours64` (B) es el techo de desempeño** del modelo sin gate con
   esta receta, y el primero que supera al pipeline separado de forma
   significativa.
3. **El preentrenamiento sirve con nuestro frontend**: no hace falta
   reproducir el log-Mel de PANNs a 32 kHz.
4. **Murmullo no mejora con la CNN6**: parece estar en un techo cerca de
   F2 0.79 con los datos de CirCor, con cualquiera de los modelos.
5. **48 veces más parámetros que `cs_wide`**: la etapa de reducción tiene
   ahora un objetivo concreto, que es cuánto de los +2.4 puntos se puede
   conservar con una red chica.

## 6. Comandos

Desde `desarrollo/src`:

    python unified_model/pack_cnn6_colab.py
    # en Colab (colab_cnn6_circor_spr.ipynb):
    #   run_cnn6_gpu.py --frontend panns32k --folds 0 3
    #   run_cnn6_gpu.py --frontend ours64 --folds 0 3
    #   run_cnn6_gpu.py --frontend ours64 --folds 0 1 2 3 4   (saltea 0 y 3)
    python unified_model/import_colab_run.py --src ../reports/cnn6_panns32k/run_v1 --run-name cnn6_panns32k
    python unified_model/import_colab_run.py --src ../reports/cnn6_ours64/run_v1 --run-name cnn6_ours64
    python unified_model/filter_variants.py --runs cs_wide cnn6_panns32k cnn6_ours64
    python unified_model/import_colab_run.py --src ../reports/cnn6_ours64/5_folds --run-name cnn6_ours64
    python unified_model/evaluate_circor_spr.py --run cnn6_ours64 --base-run cs_wide --gate-run kfold_cs
    python unified_model/plot_circor_spr.py --run cnn6_ours64 --base-run cs_wide

En Windows, `evaluate_circor_spr.py` necesita `PYTHONIOENCODING=utf-8` para
imprimir el último mensaje. Checkpoints en
`src/unified_model/checkpoints_cnn6_{panns32k,ours64}/` (no versionados).

## 7. Pendiente y traspaso

Plan acordado con el usuario el 2026-11-09, para llegar al ~90% del
desarrollo hacia el 2026-12-21. Tres etapas, en orden.

### 7.1 Etapa 1: reducción de parámetros (~2–3 semanas)

**Pregunta**: cuánto de los +2.4 puntos de capa 1 de `cnn6_ours64` (4.57M)
sobre `cs_wide` (95k) se conserva con una red chica. **Resultado esperado**:
una curva parámetros contra desempeño (accuracy de 6 clases del sistema y
métrica de decisión), con la CNN6 como techo y `cs_wide` / `cs_base` como
referencias sin destilar.

Pasos:

1. **Predicciones de la CNN6 (maestro) en TODAS las ventanas del índice**,
   incluidas las de train de cada fold: hoy solo hay val y test
   (`window_probs_fold*.csv`). Para destilar en el fold *i* se usa el
   checkpoint del fold *i* (`src/unified_model/checkpoints_cnn6_ours64/fold{i}_best.pt`,
   criterio B), que no vio val ni test de ese fold.
   - Se puede hacer local, en CPU, sin Colab: armar `Cnn6Classifier(pesos,
     "ours64")` de `run_cnn6_gpu.py`, cargar el `state_dict` del checkpoint y
     pasar las ventanas en modo eval, sin augmentation. El audio sale de las
     mismas cachés que usa `run_kfold --data circor_spr` (son las que
     empaqueta `pack_cnn6_colab.py`).
   - Guardar los logits (no solo las probabilidades), para poder usar
     temperatura. Control: en val y test tienen que coincidir con los
     `window_probs_fold*.csv` de Colab (salvo diferencias de AMP).
2. **Destilación en `run_kfold.py`** (opción nueva, p. ej. `--distill
   cnn6_ours64 --alpha 0.5 --temperature 2`): pérdida = α · CE con las
   etiquetas (pesos `inv`, como hoy) + (1 − α) · T² · KL entre las
   distribuciones del maestro y del alumno a temperatura T. El resto, receta
   de `cs_wide` (log-Mel, avg_max, `select_domain`).
   - Decidir antes de entrenar si la parte KL lleva pesos por clase (el
     maestro ya aprendió con pesos `inv`; probablemente no).
3. **Tamaños del alumno**: `--width` hoy tiene `base` (16 canales, 24k) y
   `wide` (32, 95k) en `model.py`. Sumar uno intermedio (p. ej. 48 o 64
   canales, ~200–380k) para tener al menos tres puntos en la curva.
4. **Filtro** con `filter_variants.py` en los folds 0 y 3 (solo val), con
   regla fijada antes: pasa a 5 folds si supera al mismo tamaño sin destilar
   en la media y no baja en ningún fold. Primero α y T con `wide` (dos o tres
   combinaciones como mucho); después los tamaños con la mejor.
5. **5 folds** de la ganadora y evaluación igual que en §4:
   `evaluate_circor_spr.py --run <run> --base-run cs_wide --gate-run kfold_cs`
   y `plot_circor_spr.py`.
6. **Opcional, sobre la variante final**: convoluciones separables en
   profundidad, poda y cuantización int8 (tamaño en disco y latencia en CPU;
   barato y útil para la memoria).

Ojo: los tiempos locales de `cs_wide` fueron ~11 min por época con cuatro
procesos en paralelo. Si la destilación pesa mucho en CPU, se puede llevar
a Colab con `run_cnn6_gpu.py` como modelo (mismo paquete).

### 7.2 Etapa 2: validación externa (~1 semana)

ICBHI completo y HLS-CMDS (`avances/23` §8.3), con los modelos finales:
`cnn6_ours64`, `cs_wide` y el reducido. Va después de la etapa 1 para
correrla una sola vez. HLS sigue siendo un conjunto limpio: no se elige
nada con él.

### 7.3 Etapa 3: cierre y memoria (en paralelo o al final)

- Escuchar las grabaciones con error de dominio: las 7 pulmonares de
  `cs_wide` (pacientes 40801342, 41101309 y 65114631) y las 3 cardíacas de la
  CNN6 (`CIR_00309`, `CIR_00311`, `CIR_00508`).
- Cap. 3 de la memoria: las 6 clases y la adaptación de la entrada de la
  CNN6, que ya se puede escribir con §2.2 y §3 de este documento (gana
  `ours64` contra `panns32k`).

### 7.4 Lo que queda afuera

- **Ramas dedicadas más chicas**: perdió sentido, porque el modelo
  integrado ya supera al pipeline separado de forma significativa.
- **Más ajuste de la CNN6** (variante C, más épocas): las curvas muestran
  poco margen, y como techo alcanza.

## 8. Estado del repositorio

El código (`pack_cnn6_colab.py`, `run_cnn6_gpu.py`, `import_colab_run.py`
y `colab_cnn6_circor_spr.ipynb`) ya está en el commit `99e4de9`. Sin
commitear al cerrar:

- Reportes: `reports/unified_model/runs/{cnn6_panns32k,cnn6_ours64}/`,
  `reports/unified_model/circor_spr/cnn6_ours64_vs_cs_wide/`,
  `filter_folds03.csv` y las cachés nuevas de
  `reports/sprsound/full_recording/`.
- Las carpetas crudas bajadas de Drive (`reports/cnn6_panns32k/`,
  `reports/cnn6_ours64/`) duplican lo importado y no conviene versionarlas
  (una tiene un zip de 74 MB).
- Este documento.
