# Destilación desde la CNN6: reducción de parámetros del modelo sin gate

Continúa `avances/26` §7.1 (etapa 1 del plan hacia el 2026-12-21). Se
destiló la CNN6 `cnn6_ours64` (4.57 M de parámetros) a la `UnifiedCNN` de
`cs_wide` en tres tamaños, con un control sin destilar para cada uno, en
Google Colab. Fechas: 2026-11-09 a 2026-11-15.

Resumen en cuatro líneas:

- **La destilación no mejora el sistema**: a 5 folds, el alumno `base`
  destilado (24k) empata con su control sin destilar (0.769 contra 0.769
  de accuracy de 6 clases, +0.04 puntos, IC95 [−0.64, +0.78]).
- **La ganancia del filtro no se confirmó**: en validación de los folds 0
  y 3, `base` era el único tamaño que pasaba la regla (+0.009 de métrica
  de decisión); en `wide` y `large` la destilación empató.
- **Lo pulmonar de la CNN6 no se transfiere**: el score pulmonar del
  alumno queda en 0.593, contra 0.655 del maestro. Sin destilar, la
  capacidad se satura en `wide` (de 95k a 374k no se gana nada).
- **Curva de parámetros contra desempeño** (accuracy del sistema): 24k →
  0.769, 95k (`cs_wide`) → 0.776, 4.57 M (CNN6) → 0.800. Achicar la red
  cuesta entre 2.4 y 3.1 puntos, y destilar no recupera nada.

## 1. Protocolo

Decidido con el usuario antes de entrenar:

- **Destilación por respuestas, offline** (Hinton et al., 2015): el maestro
  está congelado y el alumno imita sus probabilidades. Pérdida:

      α · CE(alumno, etiqueta; pesos inv) + (1 − α) · T² · KL(maestro_T ‖ alumno_T)

  con las dos distribuciones a temperatura T. La KL va **sin pesos por
  clase**: el maestro ya aprendió con pesos `inv`.
- **Maestro**: la CNN6 `ours64` del mismo fold (`fold{i}_best.pt`,
  criterio B), que no vio val ni test de ese fold.
- **Maestro online**: predice en la GPU, batch por batch, sobre la misma
  onda aumentada (ganancia y ruido) que ve el alumno, en modo eval y sin
  enmascarado. Es la "consistencia" de Beyer et al. (2022); la
  alternativa, logits precalculados sin augmentation, le daba al alumno
  objetivos de la ventana limpia.
- **Alumno**: la `UnifiedCNN` con la receta de `cs_wide` (log-Mel con
  z-score, n_fft 512 / hop 160, `avg_max`, sin SpecAugment, Adam lr 1e-3,
  30 épocas con paciencia 8, criterio B principal), en tres anchos: `base`
  (16 canales, 24 294 parámetros), `wide` (32, 94 662) y `large` (64,
  373 638, nuevo).
- **Control sin destilar para cada tamaño** (α = 1, `gpu_<ancho>`),
  entrenado con el mismo código. La regla del filtro compara contra él y
  no contra `cs_base` / `cs_wide`, que se entrenaron en CPU con otro
  código.
- **Filtro** en los folds 0 y 3, solo con validación (métrica de decisión
  de `avances/25` §2). Regla fijada antes: pasa a 5 folds si supera a su
  control en la media y no baja en ningún fold. Primero α y T con `wide`:
  (0.5, 2), (0.5, 4) y (0.1, 4); después los tamaños con la mejor.

## 2. Implementación

Todo en `src/unified_model/run_cnn6_gpu.py`, que ya tenía el banco de audio
en la GPU, los folds, la selección por criterio y las salidas de
`run_kfold`:

- `--model unified --width {base,wide,large} --alpha --temperature
  --teacher-dir`. Con `--alpha 1` no se carga el maestro (control).
- El alumno corre en fp32 (el log-Mel con z-score no conviene en fp16 y
  la red chica no necesita precisión mixta); el maestro, con precisión
  mixta y sin gradientes.
- **Control del maestro**: antes de entrenar cada fold, el maestro predice
  la validación y tiene que reproducir el criterio B de su corrida
  (`result_fold{i}.json`, tolerancia 0.005); si no, corta. Localmente, en
  CPU, el fold 0 dio 0.8740 contra 0.8739 esperado.
- La historia guarda la CE y la KD por separado (`train_ce`, `train_kd`).
- Los checkpoints son `state_dict` de `UnifiedCNN` y `config.json` lleva
  los argumentos de `run_kfold.build_model`: se cargan localmente igual
  que los de `run_kfold` (sirven para la validación externa).
- `model.py`: ancho nuevo `large` (64 canales).
- Notebook `colab_cnn6_circor_spr.ipynb`: configuración con `MODEL`,
  `WIDTH`, `ALPHA` y `TEMPERATURE`; el maestro se sube a Drive
  (`maestro_cnn6_ours64/`, los 5 `fold{i}_best.pt` y `result_fold{i}.json`).

Tiempos en la GPU de Colab, por época: control `base` 7 s, `wide` 14 s,
`large` 28 s; con el maestro, 34, 40 a 43 y 55 s.

## 3. Filtro por validación (folds 0 y 3)

### 3.1 α y T con el alumno `wide`

| Variante | F2 murmullo | Score pulmonar | Decisión | Por fold |
|---|---|---|---|---|
| `cs_wide` (CPU, referencia) | 0.845 | 0.606 | 0.7255 | 0.728 / 0.723 |
| `gpu_wide` (control) | 0.849 | 0.611 | 0.7298 | 0.735 / 0.725 |
| **`kd_wide_a50_t2`** | **0.851** | **0.618** | **0.7345** | 0.734 / 0.735 |
| `kd_wide_a50_t4` | 0.828 | 0.604 | 0.7164 | 0.709 / 0.724 |
| `kd_wide_a10_t4` | 0.823 | 0.607 | 0.7150 | 0.707 / 0.723 |
| `cnn6_ours64` (maestro) | 0.832 | 0.652 | 0.7416 | 0.742 / 0.742 |

- **El código nuevo reproduce a `cs_wide`**: el control da 0.7298 contra
  0.7255.
- **(0.5, 2) es la mejor combinación**, pero no pasa la regla: +0.005 en
  la media y +0.010 en el fold 3, y −0.0008 en el fold 0.
- **Con T = 4 empeora**, sobre todo murmullo en el fold 0 (F2 0.80 a 0.82
  contra 0.86).
- **Por ventana, todas las destiladas mejoran** el criterio B de
  validación (0.870 a 0.876 contra 0.863 a 0.866 del control), sobre todo
  en crackle. Esa mejora casi no llega a la métrica por grabación.
- La KD de train baja de ~0.67 a ~0.17 en 30 épocas: el alumno imita al
  maestro en train, pero eso no se traduce en validación.

### 3.2 Tamaños con α 0.5 y T 2

| Tamaño | Parámetros | Control | Destilado | Diferencia | Fold 0 / Fold 3 | ¿Pasa? |
|---|---|---|---|---|---|---|
| `base` | 24k | 0.7129 | **0.7214** | **+0.009** | +0.013 / +0.004 | **Sí** |
| `wide` | 95k | 0.7298 | 0.7345 | +0.005 | −0.001 / +0.010 | No |
| `large` | 374k | 0.7309 | 0.7301 | −0.001 | −0.007 / +0.006 | No |

- **Sin destilar, la capacidad se satura en `wide`**: de `base` a `wide`
  +0.017, de `wide` a `large` +0.001. Cuadruplicar la red chica no acerca
  a la CNN6, así que su ventaja parece venir del preentrenamiento en
  AudioSet más que del tamaño.
- **La destilación ayuda más donde falta capacidad** (`base`), y en
  `large` mejora el criterio B por ventana (0.880 / 0.877 contra 0.868 /
  0.873) sin mejorar por grabación.
- Las diferencias entre folds (±0.007) son del mismo tamaño que los
  efectos: solo la confirmación a 5 folds podía decidir.

## 4. `kd_base_a50_t2` con 5 folds

Pasa por la regla, junto con su control `gpu_base` para la comparación
apareada. Los folds 0 y 3 son los del filtro (idénticos byte a byte).
Mejores épocas (B): 23, 23, 9, 21, 19. En los dos modelos y en todos los
folds, los criterios A y B mejoraron en la misma época, así que dan los
mismos resultados.

### 4.1 Por ventana (AUC en test, media de 5 folds)

| Modelo | Murmullo | Crackle | Wheeze | Acc. bal. 6 clases |
|---|---|---|---|---|
| `kd_base_a50_t2` (B) | 0.916 ± 0.023 | 0.770 ± 0.028 | 0.876 ± 0.023 | 0.573 |
| `gpu_base` (B) | 0.907 ± 0.025 | 0.766 ± 0.028 | 0.881 ± 0.011 | 0.576 |
| `cs_base` (B, CPU) | 0.903 ± 0.025 | 0.726 ± 0.055 | 0.860 ± 0.020 | 0.550 |
| `cs_wide` (B) | 0.917 ± 0.021 | 0.758 ± 0.041 | 0.852 ± 0.038 | 0.576 |
| `cnn6_ours64` (B) | 0.925 ± 0.017 | 0.827 ± 0.036 | 0.908 ± 0.011 | 0.641 |

El control `gpu_base` ya está por encima de `cs_base` por ventana (crackle
+0.04): el código de GPU entrena algo mejor que el de CPU con la misma
receta. Por eso la comparación válida es contra el control.

### 4.2 Capa 2: por grabación

| Modelo | F2 murm. | AUC murm. | Precisión / Recall | Score pulm. (Se / Sp) | AUC crk / whz | AUC normal-anormal |
|---|---|---|---|---|---|---|
| `kd_base_a50_t2` (B) | **0.793** ± 0.051 | 0.927 | 0.701 / 0.824 | 0.594 ± 0.013 (0.36 / 0.83) | 0.701 / 0.793 | 0.804 ± 0.031 |
| `gpu_base` (B) | 0.770 ± 0.046 | 0.918 | 0.675 / 0.804 | 0.597 ± 0.010 (0.36 / 0.83) | 0.709 / 0.791 | 0.799 ± 0.023 |
| `cs_wide` (B) | 0.789 ± 0.050 | 0.933 | 0.685 / 0.827 | 0.594 ± 0.012 (0.33 / 0.86) | 0.686 / 0.797 | 0.793 ± 0.035 |
| `cnn6_ours64` (B) | 0.782 ± 0.052 | 0.927 | 0.760 / 0.795 | 0.654 ± 0.018 (0.47 / 0.84) | 0.778 / 0.828 | 0.844 ± 0.031 |

- **Murmullo**: la destilación sube el F2 0.023 sobre el control y lo deja
  al nivel de `cs_wide` y de la rama dedicada (`abl_card`, 0.794).
- **Pulmonar**: sin cambios (0.594 contra 0.597). Lo que la CNN6 gana en
  pulmonar (+0.06 de score) no llega al alumno.

### 4.3 Capa 1: sistema de punta a punta

Todos los pacientes (9083 grabaciones, 1796 pacientes):

| | `kd_base_a50_t2` (B) | `gpu_base` (B) | `cs_wide` (B) | `cnn6_ours64` (B) | Pipeline separado |
|---|---|---|---|---|---|
| Parámetros | 24k | 24k | 95k | 4.57 M | |
| Accuracy de dominio | **1.0000 (0 errores)** | 0.9991 (8) | 0.9992 (7) | 0.9997 (3) | 0.9912 (80) |
| **Accuracy de 6 clases** | 0.769 | 0.769 | 0.776 | **0.800** | 0.771 |
| F2 murmullo del sistema | 0.794 | 0.773 | 0.793 | 0.785 | 0.795 |
| Score pulmonar del sistema | 0.593 | 0.597 | 0.595 | 0.655 | 0.579 |

Diferencias apareadas de `kd_base_a50_t2` (bootstrap por paciente, 2000
repeticiones):

| Contra | Diferencia | IC95 | Solo destilado / solo el otro |
|---|---|---|---|
| `gpu_base` (control) | +0.04 puntos | [−0.64, +0.78] | 406 / 402 |
| pipeline separado | −0.12 puntos | [−0.88, +0.65] | 479 / 490 |
| `cs_wide` | −0.64 puntos | [−1.33, +0.04] | 374 / 432 |
| `cnn6_ours64` | **−3.07 puntos** | **[−4.04, −2.08]** | 500 / 779 |

Aciertos por clase:

| Clase | `kd_base_a50_t2` | `gpu_base` | Pipeline separado |
|---|---|---|---|
| Cardíaco sin murmullo | 93% | 92% | 92% |
| Cardíaco con murmullo | 83% | 81% | 83% |
| Pulmonar normal | 83% | 83% | 86% |
| Crackle | 33% | **39%** | 28% |
| Wheeze | **42%** | 33% | 38% |
| Both | **36%** | 30% | 23% |

- **Empate con el control**: la destilación cambia el reparto de los
  errores, no el total. Mejora murmullo, wheeze, both y el dominio (sin
  errores, contra 8 grabaciones pulmonares mandadas a cardíaco), y pierde
  crackle.
- **Empate con el pipeline separado** y casi con `cs_wide` (el IC95 toca
  el cero), con un cuarto de los parámetros de `cs_wide`.
- **3 puntos debajo de la CNN6**, y la diferencia es toda pulmonar.

### 4.4 Figuras

`plot_circor_spr.py --run kd_base_a50_t2 --base-run gpu_base` →
`reports/unified_model/circor_spr/kd_base_a50_t2_vs_gpu_base/figures/`:
`class_accuracy.png`, `recording_metrics.png`, `learning_curves.png`,
`roc_windows.png`, `confusion_system.png`, `confusion_windows.png` y
`domain_separation.png`. Las comparaciones contra `cs_wide` y la CNN6 (sin
figuras) quedaron en `kd_base_a50_t2_vs_cs_wide/` y
`kd_base_a50_t2_vs_cnn6_ours64/`.

## 5. Conclusiones

1. **Resultado negativo y limpio**: con control sin destilar, filtro con
   regla fijada antes y confirmación a 5 folds, la destilación por
   respuestas no mejora el modelo sin gate en ningún tamaño.
2. **La ventaja de la CNN6 no está en sus salidas**: el alumno las imita en
   train (la KD baja a ~0.17), mejora algo por ventana, pero no por
   grabación. Junto con la saturación de `wide` a `large`, apunta al
   preentrenamiento en AudioSet como fuente de la ganancia pulmonar.
3. **Curva de parámetros contra desempeño** (accuracy de 6 clases del
   sistema, 5 folds): 24k → 0.769 (con o sin destilar), 95k → 0.776,
   4.57 M → 0.800. El costo de achicar es de 2.4 a 3.1 puntos.
4. **El filtro con dos folds tiene un ruido del orden de los efectos**
   (±0.007 de métrica de decisión): sirvió para descartar (T = 4), pero no
   alcanzó para elegir entre variantes tan parecidas.
5. **Modelos para la etapa 2**: `cnn6_ours64` (techo) y `cs_wide` (la
   versión chica de referencia). El `base` destilado no aporta un punto
   distinto de la curva y queda afuera.

Limitación: maestro y alumno usan frontends distintos (log-Mel de 64
bandas a 4 kHz con n_fft 256, contra n_fft 512 con z-score). En la
destilación por respuestas solo se comparan las salidas, así que no es un
problema de método, pero la destilación por representaciones internas
(FitNets) no aplicaba directamente.

## 6. Referencias

A verificar antes de llevarlas a la memoria (año, conferencia y título
exacto):

- Buciluă, Caruana y Niculescu-Mizil (2006), *Model Compression*, KDD.
- Hinton, Vinyals y Dean (2015), *Distilling the Knowledge in a Neural
  Network*.
- Romero et al. (2015), *FitNets: Hints for Thin Deep Nets*, ICLR.
- Cho y Hariharan (2019), *On the Efficacy of Knowledge Distillation*,
  ICCV.
- Mirzadeh et al. (2020), *Improved Knowledge Distillation via Teacher
  Assistant*, AAAI.
- Stanton et al. (2021), *Does Knowledge Distillation Really Work?*,
  NeurIPS.
- Gou et al. (2021), *Knowledge Distillation: A Survey*, IJCV.
- Beyer et al. (2022), *Knowledge Distillation: A Good Teacher is Patient
  and Consistent*, CVPR.
- Schmid, Koutini y Widmer (2023), *Efficient Large-Scale Audio Tagging
  via Transformer-to-CNN Knowledge Distillation*, ICASSP.

## 7. Comandos

Desde `desarrollo/src`:

    # en Colab (colab_cnn6_circor_spr.ipynb), MODEL = "unified":
    #   WIDTH wide: ALPHA 1; (0.5, 2); (0.5, 4); (0.1, 4)     FOLDS "0 3"
    #   WIDTH base y large: ALPHA 1; (0.5, 2)                 FOLDS "0 3"
    #   WIDTH base: ALPHA 1; (0.5, 2)                         FOLDS "0 1 2 3 4"
    python unified_model/import_colab_run.py --src unified_model/destilacion_resultados/<run>
    python unified_model/filter_variants.py --runs cs_base cs_wide cnn6_ours64 gpu_base kd_base_a50_t2 \
        gpu_wide kd_wide_a50_t2 kd_wide_a50_t4 kd_wide_a10_t4 gpu_large kd_large_a50_t2
    python unified_model/evaluate_circor_spr.py --run kd_base_a50_t2 --base-run gpu_base --gate-run kfold_cs
    python unified_model/evaluate_circor_spr.py --run kd_base_a50_t2 --base-run cs_wide --gate-run kfold_cs
    python unified_model/evaluate_circor_spr.py --run kd_base_a50_t2 --base-run cnn6_ours64 --gate-run kfold_cs
    python unified_model/plot_circor_spr.py --run kd_base_a50_t2 --base-run gpu_base

En Windows, los scripts de evaluación necesitan `PYTHONIOENCODING=utf-8`.
Checkpoints en `src/unified_model/checkpoints_<run>/` (no versionados).

## 8. Pendiente y traspaso

### 8.1 Etapa 2: validación externa (~1 semana)

ICBHI completo y HLS-CMDS (`avances/23` §8.3), con `cnn6_ours64` y
`cs_wide`. HLS sigue siendo un conjunto limpio: no se elige nada con él.

- `cs_wide` se carga con `run_kfold.build_model`. La CNN6 no: hay que
  armar `Cnn6Classifier(pesos, "ours64")` de `run_cnn6_gpu.py` y cargar el
  `state_dict` (CPU alcanza para inferencia; el control del maestro tardó
  ~21 min por 7465 ventanas). Conviene una función de inferencia común
  que acepte los dos tipos de modelo.

### 8.2 Etapa 3: cierre y memoria

- Lo que ya estaba en `avances/26` §7.3 (escuchar los errores de dominio,
  Cap. 3 con las 6 clases y la adaptación de la entrada de la CNN6).
- Cap. 3 o 4: un párrafo de reducción de parámetros con la curva de §5 y
  el resultado negativo de la destilación, con las referencias de §6.

### 8.3 Lo que queda afuera

- **Cuantización, poda y convoluciones separables** (`avances/26` §7.1
  paso 6): opcional; se puede retomar al final si sobra tiempo, sobre
  `cs_wide` o la CNN6.
- **Más variantes de destilación** (por representaciones, maestro
  asistente, entrenamientos más largos): el resultado de §4 no da motivo
  para seguir.

## 9. Estado del repositorio

El código (`run_cnn6_gpu.py` con el alumno y la destilación, `model.py`
con el ancho `large` y `colab_cnn6_circor_spr.ipynb`) ya está commiteado:
se subió para correr en Colab. Sin commitear al cerrar:

- Reportes: `reports/unified_model/runs/{gpu,kd}_*`,
  `reports/unified_model/circor_spr/kd_base_a50_t2_vs_*`,
  `filter_folds03.csv` y las cachés nuevas de
  `reports/sprsound/full_recording/`.
- `src/unified_model/destilacion_resultados/` duplica lo importado y tiene
  los `.pt`: no conviene versionarla.
- Este documento.
