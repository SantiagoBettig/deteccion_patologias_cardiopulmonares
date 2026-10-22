# Recalibración en HLS-CMDS y traspaso al modelo sin gate

Cierra la etapa de integración (`avances/18`-`20`). Tiene dos partes: el
estudio de recalibración en HLS-CMDS (§1) y el documento de traspaso para
arrancar el **modelo único sin gate** en una conversación nueva (§2-§5). El
modelo sin gate es un **objetivo primario de la tesis** (plan: `avances/3`
§4; charter: HU6), así que va antes que cualquier mejora extensa de los
modelos. Fecha: 2026-10-21.

## 1. ¿Se recupera HLS recalibrando solo los umbrales?

Script: `src/integration/recalibrate_hls.py`. Lee
`reports/integration/external_hls/recordings_by_fold.csv` y no corre
modelos.

- **Partición sin fuga**: HLS tiene dos maniquíes, femenino (`F_`) y
  masculino (`M_`). Se calibra con uno y se evalúa con el otro, en las dos
  direcciones. Por maniquí hay ~25 grabaciones cardíacas y ~16 pulmonares
  con equivalente en ICBHI. El maniquí femenino tiene un solo wheezing.
- **Qué se recalibra**: solo los umbrales (gate, murmullo, crackle/wheeze).
  Los modelos y las reglas de agregación quedan como en `avances/20`.
- Resultados: media ± desvío sobre 2 direcciones × 5 modelos de fold.

| | Umbrales congelados (avances/20) | Recalibrados con el otro maniquí |
|---|---|---|
| **Pulmonar**: score ICBHI | 0.627 ± 0.104 | **0.734 ± 0.061** |
| Recall de crackle | 0.04 | **0.54** |
| Recall de wheeze / Sp | 1.00 / 0.93 | 1.00 / 0.83 |
| **Murmullo**: accuracy balanceada | 0.535 | 0.624 |
| Especificidad / recall | 0.08 / 0.99 | 0.33 / 0.92 |
| **Gate**: cardíacas bien ruteadas | 0.83 (M) / 0.77 (F) | 0.92 / 0.81, pero pierde pulmonares en una dirección (1.00 → 0.83) |
| **Sistema**: accuracy en 6 clases | 0.502 | 0.560 |

Lectura (confirma el diagnóstico de `avances/20` §3.4):

- **Pulmonar: el problema era de umbral.** Con ~16 grabaciones etiquetadas
  del equipo nuevo, el score sube +0.11 y el recall de crackle pasa de 0.04
  a 0.54, sin tocar el modelo.
- **Murmullo: el umbral no alcanza.** Aun recalibrado, la especificidad
  queda en 0.33: el modelo no separa bien en este dominio (AUC 0.67). Es un
  problema de representación: murmullos sintéticos de adulto en el
  maniquí, contra CirCor, que es pediátrico.
- **Gate: tampoco se arregla con el umbral.** Lo que gana en cardíacas lo
  pierde en pulmonares.

Para la tesis: el sistema es **trasladable a otro equipo con una
recalibración liviana en la rama pulmonar**. La rama cardíaca y el gate
necesitan trabajo de modelo para funcionar fuera de dominio.

Archivos: `reports/integration/external_hls/recalibration_{per_fold,summary,gate}.csv`
y `recalibration_run.txt`.

## 2. Estado al cerrar (lo que hay que superar)

**Configuración vigente del pipeline gate → ramas** (`avances/20` §2):
- gate v5 (MFCC, ventanas de 5 s; por grabación, probabilidad media > 0.5);
- murmullo v1 k-fold (MFCC, ventanas de 2.5 s, regla `frac`);
- pulmonar CNN6 variante B (log-Mel de PANNs, pasa-altos 100 Hz, ventanas
  de 4 s, regla `max`);
- calibración `pooled_fixed` (default de `evaluate_integration.py`).

**Números de referencia** (por grabación, todos con `pooled_fixed`):

| Evaluación | Murmullo | Pulmonar |
|---|---|---|
| Capa 2: ramas, todos los pacientes, 5 folds | F2 0.725 ± 0.013, AUC 0.898 | score ICBHI 0.526 ± 0.014, AUC crackle 0.661, AUC wheeze 0.821 |
| Capa 1: sistema, 161 pacientes de test del gate | F2 0.660 (0.671 con ruteo perfecto) | score 0.606 (0.655 con ruteo perfecto) |
| Capa 1: accuracy en 6 clases | 0.812 | |
| HLS externo, umbrales congelados | F2 0.828 (no informativo), AUC 0.667 | score 0.648, AUC crackle 0.865, AUC wheeze 1.00 |
| HLS externo: sistema, accuracy en 6 clases | 0.504 | |

Fuentes: `reports/integration/{murmur,pulmonary}_summary.csv`,
`layer1_*.csv`, `external_hls/*.csv`.

## 3. Modelo único sin gate: qué hay que decidir al arrancar

En el plan (`avances/3` §4) es un "modelo entrenado sobre el dataset
completo, sin pre-identificar cardíaco/pulmonar", evaluado con el **mismo
split** y las **mismas métricas** que el pipeline, por etapa y de punta a
punta. Preguntas para resolver con el usuario antes de escribir código.
Cada una trae una opción propuesta:

1. **Salida.**
   - (a) **6 clases softmax** (cardíaco sin/con murmullo; pulmonar
     normal/crackle/wheeze/both). Es la propuesta: es directamente
     comparable con la matriz de 6 clases del sistema (capa 1) y no tiene
     un gate escondido.
   - (b) Multi-etiqueta (murmullo, crackle, wheeze) más una cabeza de
     dominio. Más flexible, pero la cabeza de dominio es un gate dentro del
     modelo.
2. **Unidad de entrenamiento y etiquetas.**
   - Ventanas fijas, porque es lo que ve el sistema al predecir y ya se
     verificó que funcionan (`avances/19`-`20`).
   - Etiqueta cardíaca: la de la grabación (`pathology_label`).
   - Etiqueta pulmonar por ventana: derivada de los ciclos anotados que se
     superponen con ella. La alternativa es la etiqueta de la grabación,
     que es más débil.
3. **Duración de ventana y preprocesamiento.** Es el punto delicado: cada
   rama usa algo distinto.
   - Murmullo: 2.5 s, sin pasa-altos.
   - Pulmonar: 4 s, pasa-altos de 100 Hz, que **borraría S1/S2 en las
     cardíacas**.
   - Propuesta: 4 s (o 5 s, como el gate) y **sin pasa-altos**, porque el
     modelo tiene que ver los dos dominios. Se puede probar más de una.
4. **Arquitectura y dónde correrla.**
   - Primero una CNN chica con MFCC como base, igual a gate/murmullo, en CPU.
   - Después CNN6 de PANNs con fine-tuning, en Colab. Es la comparación
     justa contra la rama pulmonar, que ya usa CNN6. Para esto hay que
     reutilizar `pack_for_colab.py` / `run_finetune_gpu.py`.
5. **Folds.** Los pacientes de CirCor e ICBHI no se superponen, así que el
   fold *i* del modelo único es la unión del fold *i* de murmullo
   (`data/unified_v2/cardiac_murmur_kfold/`) y del fold *i* pulmonar
   (`data/unified_v2/pulmonary_kfold/`). Así, cada paciente queda en test
   en el **mismo fold** que en las ramas, y la comparación con la capa 2 es
   paciente a paciente.
6. **Desbalance.** Hay 3163 grabaciones cardíacas contra 920 pulmonares, y
   un 16% de murmullos. Hay que decidir entre pesos por clase,
   sobremuestreo o pérdida focal.
7. **Riesgo de atajo**, el mismo que el del gate (`avances/4`): dominio =
   dataset de origen (CIR/ICB). El modelo único puede distinguir cardíaco
   de pulmonar por la "firma" del equipo. HLS-CMDS sirve para medirlo: un
   solo equipo para los dos dominios.

## 4. Cómo evaluarlo (reutilizando lo que ya existe)

- **Inferencia**: agregar un modelo a `integration/window_inference.py`
  (`--branch unified`), con la misma lógica de pasar cada grabación por el
  modelo del fold donde es validación y por el del fold donde es test.
- **Agregación y calibración**: las mismas reglas y la estrategia
  `pooled_fixed` (`evaluate_integration.py`: `aggregate`,
  `calibrate_folds`).
  - Con 6 clases softmax, por grabación: P(murmullo) = probabilidad de
    "card: murmullo"; P(crackle) = P(crackle) + P(both); P(wheeze) =
    P(wheeze) + P(both); dominio = suma de las clases de cada dominio.
  - Así sale la misma tabla que la capa 2.
- **Comparaciones** que tiene que producir:
  1. Capa 2 contra las ramas, sobre todos los pacientes (F2 de murmullo,
     score ICBHI, AUC).
  2. Capa 1 contra el sistema con gate, sobre los 161 pacientes de test del
     gate: matriz de 6 clases y accuracy.
  3. HLS externo contra el sistema (`external_hls.py`), incluida la
     accuracy de dominio, para medir el atajo del punto 3.7.
- Documentar en `avances/22`.

## 5. Después del modelo sin gate (pendiente, en orden)

1. **Números definitivos** de los dos sistemas: gate con 5 folds para que
   la capa 1 cubra a todos los pacientes (126 pulmonares en lugar de 19).
2. **Mejoras de modelo** (elegidas con validación propia; HLS se mira una
   sola vez al final de cada mejora para que siga siendo externo):
   - pulmonar en Colab: augmentation con sonido cardíaco de fondo
     (crackle: AUC 0.78 → 0.65 en las mezclas de HLS) y sensibilidad
     pediátrica (Se 0.125);
   - gate: robustez a equipo (20% de cardíacas mal ruteadas en HLS);
   - murmullo fuera de dominio (AUC 0.67 en HLS).

## 6. Estado del repositorio

Sin commitear al cerrar: `avances/21` y `src/integration/recalibrate_hls.py`,
más sus reportes en `reports/integration/external_hls/recalibration_*`. El
resto está commiteado y pusheado (último commit: avances 20). El usuario
commitea por su cuenta.
