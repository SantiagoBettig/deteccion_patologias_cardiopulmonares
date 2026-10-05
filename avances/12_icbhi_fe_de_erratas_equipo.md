# ICBHI — fe de erratas del equipo de grabación (91 archivos mal nombrados)

## 1. Motivación

Al arrancar el modelo pulmonar (etapa 2), el usuario señaló que la página
oficial del challenge (https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
avisa, al final, de un error en el nombrado de los archivos de la base.
Se revisó si esto se había tenido en cuenta en la unificación
(`avances/2`, `avances/3`).

No se pudo abrir la página (certificado SSL vencido al momento de la
consulta), pero la versión de Kaggle que usamos trae la misma fe de erratas
en `data/raw/ICBHI/filename_differences.txt`: una lista de **91 nombres
correctos**. Esas grabaciones se publicaron con `Meditron` como equipo en
el nombre del archivo, cuando en realidad se grabaron con `AKGC417L` o
`LittC2SE`.

## 2. Qué se encontró

- **La corrección no estaba aplicada.** Los 91 archivos siguen en disco
  como `..._Meditron.wav` / `.txt` (ninguno existe con el nombre
  corregido), y `parse_icbhi.py` tomaba el equipo directamente del nombre.
  Los archivos que se corrigieron a mano durante la unificación eran de
  **HLS-CMDS** (discrepancia CSV ↔ nombre de WAV, `avances/3` bug #4), no
  estos.
- **El sample rate real del WAV confirma la fe de erratas de forma
  independiente**: los 91 están a 44.1 kHz (como AKG/LittC2SE); los
  Meditron genuinos (36) están a 4 kHz (30) o 10 kHz (6).
- **`SAMPLE_RATE_MAP` del parser también estaba mal** en otros equipos:
  asignaba LittC2SE → 4000 Hz (real: 44100) y Litt3200 → 10000 Hz (real:
  4000).

Sample rate real por equipo (header de los WAV, con el equipo ya corregido):

| Equipo | 4000 Hz | 10000 Hz | 44100 Hz |
|---|---|---|---|
| AKG C417L | 0 | 0 | 683 |
| Littmann Classic II SE | 0 | 0 | 141 |
| Littmann 3200 | 60 | 0 | 0 |
| Meditron | 30 | 6 | 0 |

## 3. Impacto en lo ya hecho

**Ninguno sobre el audio ni sobre los modelos entrenados.** El pipeline de
audio siempre leyó el sample rate del WAV (`preprocessing/audio_ops.py`,
`librosa` con `sr=None`), nunca la columna del CSV. Ningún script fuera de
`unified_dataset/` usa `equipment` ni `sample_rate_hz`. El gate y el modelo
de murmullo no se ven afectados.

**Sí afectaba cualquier análisis de sesgo por equipo sobre ICBHI**, que es
justo lo que hace falta para la etapa pulmonar. Con los nombres sin
corregir, parecía que el equipo determinaba casi la clase: sanos, URTI,
bronquiectasia y bronquiolitis aparecían "100% Meditron". Tabla corregida,
en **pacientes** por diagnóstico nativo y equipo:

| Diagnóstico | AKG | Litt3200 | LittC2SE | Meditron |
|---|---|---|---|---|
| COPD | 32 | 11 | 16 | 8 |
| Healthy | 15 | 0 | 11 | 0 |
| URTI | 6 | 0 | 8 | 0 |
| Bronchiectasis | 1 | 0 | 6 | 0 |
| Bronchiolitis | 2 | 0 | 4 | 0 |
| Pneumonia | 0 | 0 | 6 | 0 |
| LRTI | 0 | 0 | 2 | 0 |
| Asthma | 0 | 0 | 1 | 0 |

(Un mismo paciente de COPD puede figurar en más de un equipo.)

Lectura: el sesgo por equipo es **bastante más leve** de lo que parecía.
Los sanos se reparten entre AKG y LittC2SE. Quedan sesgos parciales:
Meditron y Litt3200 son 100% COPD, y neumonía es 100% LittC2SE. El sesgo
que **sí se mantiene** es por **edad**: Healthy (mediana 5 años), URTI (3)
y Bronchiolitis (1) son pediátricos; COPD (69), Pneumonia (70) y
Bronchiectasis (60) son adultos.

## 4. Corrección

`unified_dataset/parsers/parse_icbhi.py`:

- Nueva función `_read_equipment_errata`. Lee `filename_differences.txt` y
  arma `{stem en disco (..._Meditron): stem correcto}`. Si el archivo no
  está, avisa y sigue sin corregir.
- El equipo (y el resto de los campos del nombre) se parsean del **stem
  corregido**. `original_filename` sigue siendo el nombre real del archivo
  en disco: se usa para copiar el WAV y para trazabilidad. **No se renombra
  ningún archivo de `data/raw/`.**
- `sample_rate_hz` se lee del header del WAV (`soundfile.info`). Se eliminó
  `SAMPLE_RATE_MAP`.
- El parser imprime cuántas correcciones aplicó (`91/91`) y avisa si alguna
  entrada de la fe de erratas no coincide con ningún WAV.

Se re-corrió `unified_dataset/parsers/build_dataset.py` y se comparó el CSV
nuevo contra una copia del anterior:

- Mismos `file_id`, mismas columnas, mismo orden.
- Cambian solo **91 celdas de `equipment`** y **244 de `sample_rate_hz`**,
  todas en filas ICBHI. Las 244 se descomponen así: los 91 de la fe de
  erratas, 87 LittC2SE (4000→44100), 60 Litt3200 (10000→4000) y 6 Meditron
  a 10 kHz (4000→10000).
- `pathology_label` y `subject_id` idénticos. **No hace falta regenerar
  splits, ventanas ni checkpoints**: todos se indexan por `file_id`, que no
  cambió.

## 5. Pendiente

- La revisión de clases pulmonares (continuación de `avances/9`) debe usar
  la tabla corregida de la sección 3, no la versión anterior.
- Cualquier diagnóstico de atajo por equipo en el modelo pulmonar (análogo
  a `gate_model/diagnostics_source_db.py`) puede usar ahora la columna
  `equipment` del CSV unificado con confianza.
