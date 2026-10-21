# Etapa 3 — Plan de integración del pipeline (gate → cardíaco / pulmonar)

Documento de traspaso: resume el estado de los tres modelos al cerrar la
etapa 2 y deja acordado cómo se hace la evaluación integrada, para retomar en
una conversación nueva sin releer todo `avances/`. Las decisiones se tomaron
con el usuario el 2026-10-20.

## 1. Decisión de fondo

Se pasa a la **integración** y se deja para una etapa posterior el ajuste fino
de cada modelo. Motivo: los tres modelos están en un nivel razonable, y la
evaluación integrada es el resultado central del trabajo (comparar
cardíaco/pulmonar por separado contra el sistema integrado, plan del
trabajo, HU6) que todavía no existe. Cualquier mejora posterior de un modelo
se enchufa en el pipeline sin rehacerlo.

## 2. Modelos vigentes (punto de partida de la integración)

| Etapa | Modelo vigente | Entrada | Resultado | Checkpoints | Detalle |
|---|---|---|---|---|---|
| Gate (cardíaco vs. pulmonar) | **v5**: CNN chica, MFCC, class weighting | Ventanas fijas de 5 s, paso 2.5 s (`preprocessing/make_gate_windows.py`) | Test propio: accuracy 0.975, F1 0.985 por ventana. HLS-CMDS (externo): F1 0.847, recall cardíaco 0.74 | `src/gate_model/checkpoints_v5_mfcc/best.pt` (split único, `data/unified_v2/splits.csv`); umbral 0.5 | `avances/4`, `avances/6` |
| Cardíaco — murmullo (presente/ausente) | **v1**: CNN 3 bloques, MFCC (20 coef., n_mels 64, n_fft 512, hop 160), `pos_weight`×1.4, checkpoint por F2 | Ventanas de **4 latidos** (S1→S1 de los `.tsv` de CirCor, mediana 2.3 s, rango intercuartil 2.1-2.6 s) | k-fold: F2 0.737 ± 0.022, recall 0.78 por ventana | **`src/cardiac_murmur_model/checkpoints_kfold/fold{0-4}_best.pt`**; umbrales por fold en `reports/cardiac_murmur/kfold/per_fold.csv` (`window_threshold`: 0.50 / 0.50 / 0.55 / 0.50 / 0.50); folds en `data/unified_v2/cardiac_murmur_kfold/` | `avances/10`, `avances/11` |
| Cardíaco — outcome clínico | Cerrado como limitación: no hay señal acústica suficiente (accuracy ~0.5) | — | — | — | `avances/10` §4-5 |
| Pulmonar — crackle / wheeze por ciclo | **Variante B**: CNN6 de PANNs, bloques 3-4 + fc1 ajustados en Colab, augmentation en GPU | **Un ciclo respiratorio** (límites de los `.txt` de ICBHI), recortado/rellenado a 4 s; pasa-altos 100 Hz → 4 kHz → RMS por grabación → remuestreo a 32 kHz → log-Mel de PANNs | Score ICBHI 0.572 ± 0.027, AUC crackle 0.708, AUC wheeze 0.836 | **`reports/pulmonary_adventitious/v8_cnn6_ft_B/fold{0-4}_best.pt`** (cargar con `run_finetune_gpu.FineTuneCnn6`); umbrales por fold en `.../v8_cnn6_ft_B/per_fold.csv` (crackle 0.65/0.50/0.65/0.50/0.60; wheeze 0.85/0.75/0.70/0.95/0.70); folds en `data/unified_v2/pulmonary_kfold/` | `avances/13`-`17` |

Limitaciones conocidas que se arrastran: sensibilidad pediátrica baja en
pulmonar (~0.16), dominio externo débil del gate (HLS-CMDS), umbrales
calibrados con solo 16 pacientes de validación por fold (`avances/17` §4).

## 3. Aclaraciones conceptuales (preguntas del usuario)

**¿Por qué con un gate de 5 folds se evaluaría sobre los 1068 pacientes y no
sobre 161?** Entre folds no cambia el umbral: cambia qué pacientes se usan
para entrenar y cuáles para evaluar. Con un split único, solo el 15% de test
(161 pacientes, 19 pulmonares) nunca fue visto por el modelo. Con 5 folds se
entrenan 5 gates y cada paciente queda en test exactamente una vez, evaluado
por un gate que nunca lo vio, así que los 1068 tienen una predicción limpia.
Es lo que ya hacen los modelos de murmullo y pulmonar.

**¿Por qué aparecía la "segmentación", si entrenamos con ventanas?** Solo el
gate usa ventanas fijas. Los modelos de rama se entrenaron sobre **ciclos
definidos por las anotaciones de los datasets**: murmullo con 4 latidos
según los `.tsv` de CirCor, y pulmonar con cada ciclo respiratorio según los
`.txt` de ICBHI. En una grabación nueva esas marcas no existen, así que el
pipeline necesitaría o usar las anotaciones ("segmentación oráculo", ideal y
poco realista) o detectar los ciclos automáticamente (costoso). **La opción
elegida, ventanas fijas, evita las dos.**

## 4. Decisiones acordadas

1. **Ramas con ventanas fijas de X segundos, solo al predecir, sin
   reentrenar** (opción "3a"). Cada grabación se corta en ventanas fijas y
   se pasa por los modelos de rama ya entrenados. Es lo realista: el sistema
   recibe solo audio. Duraciones propuestas, para parecerse a lo que cada
   modelo vio al entrenar:
   - murmullo: **~2.5 s** (sus ventanas de 4 latidos miden 2.1-2.6 s);
   - pulmonar: **4 s** (los ciclos se llevaron a 4 s; su pooling por máximo
     en el tiempo detecta el evento esté donde esté en la ventana).
   - Solapamiento: 50% (como el gate).

   **Si los resultados son malos**, el paso siguiente es reentrenar las
   ramas con ventanas fijas (opción "3b"): murmullo con la etiqueta por
   grabación; pulmonar con la etiqueta de cada ventana derivada de los
   ciclos anotados que se superponen con ella.

2. **Salida por grabación** (como un médico con una auscultación):
   - Gate: clasifica cada ventana de 5 s y decide cardíaco/pulmonar para la
     grabación por mayoría o promedio de probabilidades.
   - Cardíaco: murmullo / sin murmullo. Etiqueta real: `pathology_label ==
     "heart_murmur"` de esa grabación (es por ubicación de auscultación).
   - Pulmonar: normal / crackles / wheezes / ambos. Etiqueta real:
     `has_crackles` / `has_wheezes` del CSV unificado (por grabación).
   - **Agregación de ventanas a grabación: hay que elegirla con cuidado.**
     Tomar el máximo de las ventanas acumuló falsos positivos en murmullo
     bajo k-fold (`avances/10` §14). Candidatas: proporción de ventanas
     positivas ≥ k, o probabilidad media con umbral propio. La regla y su
     umbral se calibran **sobre los pacientes de validación de cada fold**,
     nunca sobre test.

3. **Evaluación en dos capas, sin reentrenar el gate:**
   - **Capa 1 — pipeline de punta a punta sobre los 161 pacientes de test
     del gate** (475 grabaciones cardíacas, 116 pulmonares). Gate → rama →
     resultado. Para cada paciente, la rama usa **el modelo del fold donde
     ese paciente estaba en test**: se verificó que 31 de los 142 pacientes
     cardíacos del test del gate estaban en el train del murmullo v1 de
     split único. Por eso se usan los checkpoints k-fold y no `best.pt` de
     v1. Mide cuánto se pierde por los errores del gate (se espera muy poco:
     acierta el 97.5% de las ventanas en el test propio).
   - **Capa 2 — ramas con ventanas fijas sobre todos los pacientes**, con
     sus 5 folds (942 cardíacos, 126 pulmonares), cada uno evaluado por un
     modelo que no lo vio, suponiendo que el gate rutea bien. Da la
     cobertura estadística que la capa 1 no tiene (solo 19 pacientes
     pulmonares).
   - **Gate con 5 folds: solo si hace falta** — si la capa 1 muestra una
     pérdida apreciable por el gate, o al final, para los números
     definitivos de la tesis. Ganaría cobertura (19 → 126 pacientes
     pulmonares en la evaluación integrada) y media ± desvío, pero hoy
     cambiaría poco, porque el gate casi no se equivoca en los datos
     propios, y su debilidad real (otro dominio) no se arregla con folds.

4. **Comparación con la segmentación oráculo**: evaluar las mismas métricas
   por grabación agregando las predicciones por ciclo anotado (ya
   disponibles como OOF: `v8_cnn6_ft_B/oof_predictions.csv`; las de murmullo
   k-fold habría que regenerarlas desde los checkpoints). La diferencia
   **ventanas fijas vs. ciclos anotados** cuantifica cuánto se pierde por no
   tener las anotaciones, que es un resultado para la tesis.

## 5. Pasos concretos para la próxima conversación

1. **Script de inferencia por ventanas fijas** para cada rama (murmullo y
   pulmonar), a partir del audio crudo del dataset unificado, con el mismo
   preprocesamiento que usó cada modelo al entrenar:
   - murmullo: `audio_ops.preprocess` (sin pasa-altos) → ventanas de ~2.5 s
     → MFCC;
   - pulmonar: `preprocess(..., highpass_cutoff=100)` → ventanas de 4 s →
     remuestreo a 32 kHz → log-Mel de PANNs (`extract_cnn6_embeddings.logmel`,
     frontend `panns32k`).
   - Guardar probabilidades por ventana (grabación, fold, paciente) para
     poder probar reglas de agregación sin volver a correr los modelos.
2. **Agregación a nivel grabación**, calibrada en validación de cada fold.
3. **Capa 2**: métricas por grabación de cada rama sobre todos los
   pacientes. Murmullo: recall, precisión, F1, F2. Pulmonar: score ICBHI por
   grabación (4 clases) y AUC por etiqueta. Más el desglose por equipo y
   edad de siempre.
4. **Capa 1**: gate v5 (votación por grabación) + ramas, sobre los 161
   pacientes de test del gate; matriz de confusión del sistema completo y
   pérdida por la cascada.
5. **Comparación con la segmentación oráculo** (punto 4 de §4).
6. Documentar en `avances/19`.

Líneas posteriores, ya identificadas: modelo único sin gate para comparar
contra el pipeline (`avances/3` §4); validación externa de punta a punta con
HLS-CMDS (tiene murmullos, crackles y wheezes del mismo equipo); gate con 5
folds; ajuste fino de los modelos (`avances/17` §5).

## 6. Estado del repositorio al cerrar esta conversación

Sin commitear al cerrar (el usuario commitea por su cuenta): `avances/17` y
`18`, `README.md`, `calibration_study.py`, `export_val_probs.py`, y los
reportes nuevos (`calibration_study/`, `v8_cnn6_ft_{A,B,C}/`,
`v4_hp100/val_probs_fold*.csv`). Los checkpoints `.pt` y `data/` están en
`.gitignore`. Los checkpoints de la variante B viven solo en la carpeta de
reportes local (y en Drive). Si se borra esa carpeta, hay que volver a
bajarlos de Drive.
