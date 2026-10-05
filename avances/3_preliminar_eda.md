# Proyecto Final CEIA — Clasificación de Patologías Cardiopulmonares
## Resumen de sesión — EDA y decisiones de diseño

---

## 1. Bugs encontrados y corregidos

| # | Script | Problema | Causa | Solución |
|---|---|---|---|---|
| 4 | HLS-CMDS (metadatos) | Identificadores de tipo de sonido en el CSV no coincidían con los nombres reales de los `.wav` (ej. G=Coarse_Crackles, C=Fine_Crackles) | Discrepancia entre el CSV y el filesystem | Corregido manualmente, archivo por archivo |
| 5 | `parse_circor.py` | Conteo de WAVs en `unified/` (3209) no coincidía con `raw/` (3163) para CirCor | El campo `Recording locations:` a veces repite un código de ubicación (ej. `AV+PV+AV+TV+MV`); como el glob por ubicación ya trae todos los WAVs de esa ubicación (incluidos sufijos `_1`/`_2`/`_3`), la repetición reprocesaba y duplicaba el mismo conjunto de archivos | Deduplicar la lista `locations` preservando el orden (`dict.fromkeys(...)`) antes de iterar |

*(Bugs 1-3, de sesiones anteriores: construcción de filename en HLS, columnas distintas entre HS/LS, separador de `demographic_info.csv` en ICBHI.)*

---

## 2. EDA del `master.csv` — integridad estructural

- **4183 registros totales**, sin `file_id` duplicados.
- **0 archivos huérfanos** en ambas direcciones (CSV ↔ filesystem).
- Todo el NaN se explica por la fuente de origen (`source_db`), no hay datos faltantes anómalos:
  - `age_years`, `has_crackles`, `has_wheezes`: exclusivos de ICBHI.
  - `height_cm`, `weight_kg`: solo CirCor e ICBHI parcialmente.
  - `duration_sec`: recalculable desde el WAV, no es un problema real.

---

## 3. Distribución de clases — hallazgo clave y decisión

Se encontró que casi todas las clases de **HLS-CMDS** (`atrial_fibrillation`, `av_block`, `tachycardia`, `extra_heart_sound`, `crackles`, `pleural_rub`, `rhonchi`, `wheezing`) estaban 100% asociadas a esa única fuente, grabada en **maniquí**, con muy pocos casos por clase (3-14).

**Decisión: se descartó HLS-CMDS.** Dataset final de trabajo: **CirCor + ICBHI** (sujetos reales).

### Dataset final por rama

**Cardíaco (100% CirCor):**
| Clase | N |
|---|---|
| `normal_heart` | 1534 |
| `abnormal_heart_unspecified` | 1013 |
| `heart_murmur` | 616 |

**Pulmonar (100% ICBHI):**
| Clase | N |
|---|---|
| `copd` | 793 |
| `pneumonia` | 37 |
| `normal_lung` | 35 |
| `respiratory_infection` | 25 |
| `bronchiectasis` | 16 |
| `bronchiolitis_asthma` | 14 |

### Limitación importante documentada
Al descartar HLS, **`sound_type` quedó perfectamente correlacionado con `source_db`** (cardíaco = 100% CirCor, pulmonar = 100% ICBHI). No hay forma, con este dataset, de verificar que un futuro modelo "gate" generalice a un pulmonar grabado con el equipo de CirCor o un cardíaco grabado con el de ICBHI. Se considera de riesgo moderado porque cardíaco/pulmonar son señales físicamente muy distintas, pero queda como limitación a mencionar en el informe.

### Patrón demográfico
- Cardíaco: bien distribuido entre categorías de edad y sexo, sin sesgo fuerte.
- Pulmonar: `copd` es casi exclusivamente adulto (787/793) — consecuencia directa de la composición de ICBHI, no un artefacto a corregir.

---

## 4. Decisión de arquitectura de modelado

Se evaluaron dos diseños a comparar en el trabajo final:

1. **Pipeline de 3 modelos:**
   - Modelo 1 (gate): clasificador binario cardíaco vs. pulmonar.
   - Modelo 2: clasificador de patología cardíaca (activado si el gate predice "cardíaco").
   - Modelo 3: clasificador de patología pulmonar (activado si el gate predice "pulmonar").
2. **Modelo único unificado:** entrenado sobre el dataset completo (sin HLS), sin pre-identificar cardíaco/pulmonar.

**Puntos a tener en cuenta:**
- Evaluar **por etapa** (cada modelo aislado) y **end-to-end** (pipeline completo), ya que el error del gate se propaga en cascada en el diseño de 3 modelos.
- Usar el **mismo split** train/val/test para ambos diseños, y las mismas métricas (accuracy + F1 macro/weighted, dado el desbalance).
- El modelo único no depende de que el gate acierte primero — punto de comparación interesante incluso si rinde peor en accuracy pura.

---

## 5. EDA de audio (waveforms + espectrogramas Mel)

Análisis sobre muestra estratificada (3 ejemplos por clase, separado por `sound_type`), usando librosa.

**Hallazgos:**
- **Rango de frecuencias:** en espectrogramas pulmonares (sr nativo hasta 44100 Hz), toda la energía relevante está por debajo de ~2048 Hz. Se observaron algunos picos que exceden los 4 kHz.
- **Periodicidad cardíaca:** patrón vertical muy marcado correspondiente a los ciclos S1-S2 — posible insumo para segmentación por ciclo cardíaco en la etapa de features (pendiente, no resuelto aún).
- **Diferencias de amplitud entre clases pulmonares:** `bronchiolitis_asthma`, `copd` y `pneumonia` muestran amplitudes bruscamente mayores (~±1.0) que `bronchiectasis`/`normal_lung` (~±0.05–0.2). Pendiente verificar si correlaciona con la clase (posible atajo espurio) o es incidental a la grabación.
- **Sample rate mixto confirmado visualmente en ICBHI:** un archivo de `copd` nativo a 4000 Hz mientras el resto de la clase está a 44100 Hz.

---

## 6. Decisiones de preprocesamiento finalizadas

| Paso | Decisión | Justificación |
|---|---|---|
| **Filtrado** | Pasabanda, cortando por encima de ~2048 Hz. Aplicar **antes** del resample (evita aliasing) | Confirmado visualmente: sin contenido clínico relevante por encima de ese umbral |
| **Resample** | Frecuencia objetivo: **4000 Hz** | Nyquist para 2048 Hz; coincide con el sample rate nativo de CirCor (no requiere resample para media base) |
| **Normalización de amplitud** | Por **RMS** | Más robusto que min-max/pico ante ruido puntual; equipara energía entre equipos de grabación distintos |
| **Duración / ventaneo** | **Ventaneo con solapamiento (sliding window)** para clases minoritarias, en vez de solo truncar | Genera más ejemplos de entrenamiento donde más se necesita (dataset chico y desbalanceado) |

### ⚠️ Advertencia de diseño crítica
El split train/val/test debe hacerse **por `file_id` (o por sujeto) antes de aplicar el ventaneo**, nunca después. Ventanear primero y splitear después generaría **data leakage** (fragmentos del mismo audio/paciente en train y test simultáneamente, inflando métricas).

---

## 7. Próximos pasos

- [ ] Definir criterio de split train/val/test (por `file_id` o por sujeto, estratificado por `pathology_label`).
- [ ] Verificar si la diferencia de amplitud entre clases pulmonares correlaciona con la clase o es incidental (ej. gráfico de RMS medio por `pathology_label`).
- [ ] Construir pipeline concreto de preprocesamiento: filtrado → resample → normalización RMS → ventaneo.
- [ ] Evaluar segmentación por ciclo cardíaco (S1/S2) como alternativa a ventanas de tiempo fijas, para la rama cardíaca.
- [ ] Definir estrategia de extracción de features (MFCCs vs. espectrogramas Mel como input al modelo).
- [ ] Implementar y entrenar el pipeline de 3 modelos (gate + cardíaco + pulmonar).
- [ ] Implementar y entrenar el modelo único unificado, para comparación.
- [ ] Definir estrategia de balanceo de clases para la rama pulmonar (class weights, oversampling, focal loss, o agrupar clases ultra-minoritarias).
