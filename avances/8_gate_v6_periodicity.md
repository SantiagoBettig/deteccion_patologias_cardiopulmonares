# Proyecto Final CEIA — Modelo gate v6: feature de periodicidad (resultado negativo)

## 1. Motivación

Ver `avances/7_investigacion_debilidad_cardiaca.md` sección 4: log-Mel y
MFCC son ambos features de timbre, exactamente la propiedad que está
invertida entre CirCor/ICBHI y HLS-CMDS (sección 7 de
`4_gate_model_baseline_y_sesgo.md`). Se propuso una feature de **ritmo**
(periodicidad de la envolvente de energía: ~0.33-1.5s para el ciclo
cardíaco, ~1.5-7.5s para el respiratorio) como señal complementaria menos
dependiente del timbre, con la hipótesis de que sobreviviría a la brecha de
dominio mejor que log-Mel/MFCC.

## 2. Qué se implementó

Gate v6, construido **sobre v5** (MFCC, n_mfcc=20): arquitectura de dos
ramas.

- `gate_model/periodicity.py`: `periodicity_features(y, sr)` — autocorrela-
  ción de la envolvente RMS de la ventana, resumida en 3 valores:
  `heart_score` (máximo de autocorrelación normalizada en el rango
  40-180 lpm), `breath_score` (ídem, 8-40 rpm) y su diferencia.
- `gate_model/dataset_periodicity.py` (`GatePeriodicityDataset`): igual que
  `GateDataset` pero calcula la periodicidad sobre la MISMA forma de onda
  (ya augmentada) usada para el MFCC, y devuelve `(x_spec, x_period, y)`.
- `gate_model/model_periodicity.py` (`GateCNNPeriodicity`): la misma CNN de
  `model.py` para la rama espectral + un MLP chico (`Linear(3,16)+ReLU`)
  para la rama de periodicidad; ambos embeddings se concatenan antes de la
  capa de clasificación final.
- `gate_model/run_v6_periodicity.py`: self-contenido (no reutiliza
  `train.py`/`evaluate.py` de v1-v5, porque el modelo toma dos inputs en
  vez de uno), guarda en `gate_model/checkpoints_v6_periodicity/` y
  `reports/gate/runs/v6_periodicity/`.

## 3. Chequeo previo — señal de alerta

Antes de entrenar, se calculó el promedio de la feature de periodicidad
sobre 50 ventanas cardíacas y 50 pulmonares (train, sin augmentation):

| | heart_score | breath_score | diff |
|---|---|---|---|
| Cardíaco | 0.402 | 0.206 | **0.196** |
| Pulmonar | 0.465 | 0.196 | **0.269** |

El `diff` (que debería ser más alto en cardíaco si la hipótesis fuera
correcta) salió **más alto en pulmonar** — el sentido contrario al
esperado. Se decidió entrenar igual (el promedio simple no captura
relaciones no lineales que el MLP podría aprender), documentando la
alerta.

## 4. Resultados

| Métrica | Test v5 (MFCC) | Test v6 (MFCC + periodicidad) | HLS-CMDS v5 | HLS-CMDS v6 |
|---|---|---|---|---|
| Accuracy | 0.9753 | 0.9753 | 0.867 | 0.748 |
| Precision | 0.9825 | 0.9913 | 0.991 | 1.000 |
| Recall | 0.9871 | 0.9781 | 0.740 | **0.497** |
| F1 | 0.9848 | 0.9847 | 0.847 | **0.664** |

En test, v6 es equivalente a v5. En HLS-CMDS, **v6 empeoró
sustancialmente**: F1 0.847 → 0.664, con precisión perfecta (1.000) pero
recall de cardíaco desplomado a 49.7% (la mitad de las ventanas cardíacas
de HLS-CMDS se clasifican como pulmonares). El modelo se volvió mucho más
conservador — mismo patrón cualitativo que v3 (SpecAugment) pero mucho más
marcado.

## 5. Análisis

La alerta de la sección 3 se confirmó: la feature de periodicidad, tal
como se diseñó, **no es más robusta al cambio de dominio que el timbre —
es peor**. Hipótesis de por qué:

- La ventana de 5s alcanza para varios ciclos cardíacos (0.33-1.5s) pero
  como mucho 1-2 ciclos respiratorios completos (1.5-7.5s, casi tan largo
  como la ventana misma) — la estimación de `breath_score` es ruidosa por
  diseño, no por el dominio.
- Es probable que la envolvente de energía en ventanas pulmonares (sonidos
  de crepitantes/sibilancias, más las propias características del
  micrófono/stetoscopio de ICBHI) genere autocorrelación alta también en
  el rango "cardíaco" por azar o por estructura de ruido de banda ancha,
  sin relación real con el ritmo cardíaco — de ahí que el `diff` saliera
  invertido incluso en los datos de entrenamiento.
- Al concatenarse con el embedding espectral, esta feature ruidosa (y con
  una correlación espuria con la clase en el propio train) parece haber
  introducido un atajo adicional que generaliza peor a HLS-CMDS que el
  atajo de timbre que ya tenía v5 — el modelo se volvió más "seguro" de
  pulmonar en casos ambiguos.

**Descartado.** v5 (MFCC) sigue siendo la versión vigente del gate. No se
promueve v6 — queda documentado como resultado negativo, igual que v3.

## 6. Conclusión y decisión

Tras dos intentos de mitigación adicional sobre v4 (v5: MFCC, mejora
parcial; v6: periodicidad, resultado negativo), y a pedido del usuario: se
deja de iterar sobre el gate por ahora. **v5 (MFCC, F1 en HLS 0.847) queda
como versión final** para esta etapa del proyecto. Se documenta como
limitación conocida (no resuelta) la debilidad en recall de cardíaco sobre
HLS-CMDS, con la hipótesis de dominio (maniquí vs. paciente real,
`4_gate_model_baseline_y_sesgo.md` sección 7, y el hallazgo de
`avances/7_...md` sobre los modos de filtro Bell/Diaphragm usados al grabar
HLS-CMDS) como explicación más probable y no descartada.

## 7. Próximos pasos

- Avanzar a la etapa 2 (modelos de patología cardíaca y pulmonar, con
  segmentación por ciclo) usando el gate v5 como está.
- Evaluar agregación por grabación completa (voto mayoritario por
  `file_id`) — pendiente desde `4_gate_model_baseline_y_sesgo.md` sección 9,
  sigue siendo una mejora barata de probar en paralelo si se retoma el gate
  más adelante.
