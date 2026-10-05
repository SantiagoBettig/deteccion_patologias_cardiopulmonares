# Proyecto Final CEIA — Intento de migrar entrenamiento/evaluación a Google Colab

## 1. Motivación

Entrenar localmente el modelo gate es lento por falta de GPU en la máquina
del usuario. Toda la carpeta `desarrollo/` (incluida `data/`, no versionada
en git) ya está sincronizada en Google Drive vía el cliente de escritorio,
así que en principio Google Colab podría acceder a los mismos datos y
checkpoints sin subir nada manualmente, aprovechando su GPU gratuita.

## 2. Qué se implementó

- **`desarrollo/src/gate_model/train.py`**: `NUM_WORKERS` pasó de una
  constante fija (`0`, pensada para Windows) a una detección simple por
  plataforma (`0` en Windows, `2` en el resto), sin tocar el resto de la
  lógica de entrenamiento.
- **`desarrollo/google_collab_runner.ipynb`** (nuevo): notebook que monta
  Google Drive, fija `PROJECT_PATH` (ruta editable a la copia de `desarrollo`
  en Drive), instala `requirements.txt`, verifica que Colab detecte la GPU,
  y corre en orden `gate_model/train.py`, `evaluate.py` y
  `evaluate_external_hls.py`, imprimiendo cada etapa para poder seguir el
  progreso. No se migró el resto del pipeline (`build_dataset.py`,
  `make_splits.py`, `make_gate_windows.py`, `make_hls_windows.py`): se
  asumió que `data/unified_v2/` y `data/gate/windows/` ya existen en Drive
  por haberse generado localmente.
- Ningún cambio de rutas fue necesario para que los scripts corrieran en
  Colab: `common/paths.py` ya calcula todo relativo a `Path(__file__)`
  ([ver `common/paths.py`](../desarrollo/src/common/paths.py)), así que
  montar Drive en la misma estructura relativa alcanzó.

## 3. Problema encontrado — cuello de botella de I/O con Drive montado

Primera corrida en Colab: la GPU (Tesla T4) fue detectada correctamente,
pero después de imprimir el balance de clases del split de train, el script
quedó sin imprimir nada durante 15+ minutos (más lento que entrenar en CPU
local).

**Diagnóstico**: `train.py` solo imprime el log de una época cuando
terminan **todos** los batches de train (764) y val (175) de esa época — no
hay progreso intermedio (`_run_epoch`, ver
[`gate_model/train.py`](../desarrollo/src/gate_model/train.py)). Que no
apareciera ni la época 1 en 15 minutos, con un modelo de solo 23.585
parámetros (cómputo de GPU trivial), apuntó a un cuello de botella de I/O,
no de cómputo.

**Causa**: `GateDataset.__getitem__` ([`gate_model/dataset.py`](../desarrollo/src/gate_model/dataset.py))
abre y decodifica con `librosa` un archivo `.wav` individual por cada
ventana, en cada época. El dataset tiene **35.820 archivos** de ventanas
(train+val+test+HLS externo). Leerlos desde Google Drive montado vía FUSE
implica una latencia de red por cada apertura de archivo — muchísimo más
lenta que leer del disco local — y ese costo se multiplica por las 15
épocas. La GPU queda ociosa esperando datos.

## 4. Mitigación parcial — copia local + `GATE_DATA_DIR`

- **`desarrollo/src/common/paths.py`**: `GATE_DIR` ahora puede
  sobreescribirse con la variable de entorno `GATE_DATA_DIR`, sin cambiar el
  comportamiento por defecto (si la variable no está definida, se comporta
  exactamente igual que antes).
- **`google_collab_runner.ipynb`**: nueva celda que copia `data/gate/` a
  disco local de la VM (`/content/gate_data`) usando 32 hilos en paralelo
  (`ThreadPoolExecutor`) antes de entrenar, y setea `GATE_DATA_DIR` para que
  los scripts lean de ahí en vez de Drive montado.

**Resultado**: la copia paralela reportó los 35.820 archivos a copiar, con
un tiempo de copia considerable (dominado por la misma latencia de Drive por
archivo, ahora concentrada en un solo paso en vez de repetirse en cada
época). El usuario evaluó que, en la práctica, el tiempo total no muestra
una mejora notoria frente a entrenar en su PC — la copia inicial sigue
pagando el costo de 35.820 aperturas de archivo, solo que una vez en lugar
de quince.

## 5. Discusión — ¿ayudaría cambiar a MFCC?

Se planteó si extraer MFCC en vez de espectrograma log-Mel reduciría el
tiempo de cómputo. Conclusión: **no**. `librosa.feature.mfcc` calcula
internamente el mismo espectrograma log-Mel (STFT + banco de filtros Mel)
que ya se usa, y le agrega una DCT (transformada discreta coseno) para
obtener los coeficientes — es más cómputo, no menos. Además, el cómputo de
features en CPU no es el cuello de botella real (sección 3): el costo
dominante es la cantidad de archivos leídos desde Drive, no lo que se
calcula a partir de ellos.

## 6. Discusión — separar extracción de features de la lectura de archivos

Se planteó precalcular y empaquetar los datos en pocos archivos grandes (un
tensor único por split, `.npy`/`.pt`) en vez de miles de `.wav` sueltos,
para que tanto la copia a Colab como la lectura en cada época sean pocas
operaciones de I/O en lugar de 35.820.

Distinción importante encontrada en la discusión: **qué se empaqueta
importa para la augmentation**, no solo la velocidad.

- El split de train aplica augmentation estocástica en cada época — ganancia
  aleatoria + ruido sobre la forma de onda, y SpecAugment sobre el
  espectrograma (`GateDataset._augment_waveform`/`_spec_augment`,
  mitigación de sesgo documentada en `4_gate_model_baseline_y_sesgo.md`).
  Si se precalcula y guarda el **espectrograma/MFCC final**, esa
  augmentation queda fija — todas las épocas verían la misma versión
  "aumentada" de cada ventana, perdiendo la variabilidad que la hace útil.
- Empaquetar en cambio la **forma de onda preprocesada** (post filtro/resample,
  tal como la genera `make_gate_windows.py`, antes de cualquier feature)
  preserva la augmentation y el cálculo de espectrograma al vuelo, pero
  reemplaza miles de lecturas de archivo por una lectura de tensor en RAM
  por split.
- Para val/test/HLS externo (sin augmentation, deterministas) sí sería
  seguro precalcular directamente el espectrograma final.

**Sin implementar todavía** — el usuario quiere evaluar primero si migrar a
Colab tiene sentido en términos de costo/beneficio antes de invertir en este
cambio de formato de datos.

## 7. Estado y próximos pasos

- Pendiente de decisión del usuario: si seguir invirtiendo en la migración a
  Colab (empaquetar los datos en tensores por split, con o sin extracción de
  features offline) o descartar Colab y seguir entrenando localmente.
- Si se retoma: la opción evaluada como más prometedora es empaquetar forma
  de onda preprocesada (no features finales) para el split de train, y
  opcionalmente features ya calculadas para val/test/HLS externo, ya que ahí
  no hay augmentation que preservar.
