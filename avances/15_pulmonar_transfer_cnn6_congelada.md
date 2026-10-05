# Modelo pulmonar — transfer learning, etapa 1: CNN6 (PANNs) congelada

Continuación de `avances/14`. Parte de v4 (CNN de 3 bloques entrenada desde
cero, pasa-altos 100 Hz): score ICBHI **0.559 ± 0.007**, AUC crackle 0.676,
AUC wheeze 0.793.

## 1. Motivación y por qué por etapas

Todas las mejoras de v1 → v4 vinieron de cómo se presenta la señal y de cómo
se elige el modelo, no de su capacidad. El límite de fondo es la **escasez de
datos**: con ~85 pacientes de train por fold, una red que aprende todo desde
cero tiene un techo. El transfer learning ataca eso: parte de filtros ya
aprendidos sobre ~2M clips de AudioSet.

Modelo elegido: **PANNs CNN6** (Kong et al., 2020). Pesos oficiales en
Zenodo, doi:10.5281/zenodo.3987831, licencia CC-BY-4.0, archivo
`Cnn6_mAP=0.343.pth` (23.7 MB, MD5 verificado), en
`data/pretrained/` (no versionado). Es la variante chica de PANNs (4.6M
parámetros); CNN14 (80M) y AST quedan descartados sin GPU.

**Benchmark de tiempo en CPU** (cómputo de la red, 10 hilos, batch 64,
extrapolado a una corrida completa de 40 épocas × 5 folds; hay que sumarle
el cálculo de espectrogramas):

| Configuración | Minutos por época | Corrida completa |
|---|---|---|
| v4 (CNN 3 bloques, 24k parámetros) | 0.8 | 2.7 h |
| CNN6, fine-tuning completo, entrada 64×401 | 16.3 | 54 h |
| CNN6, congelando los bloques 1-2 | 11.7 | 39 h |
| CNN6, fine-tuning completo, entrada 40×251 | 7.1 | 24 h |
| **CNN6 congelada (embeddings una vez + clasificador)** | — | **~15 min en total** |

Reentrenar la CNN6 entera como v4 es inviable en CPU. Por eso se va por
etapas:

- **Etapa 1**: red congelada y un clasificador chico encima. Responde en
  minutos si lo que aprendió de AudioSet sirve para nuestros ciclos.
- **Etapa 2**: fine-tuning, solo si la etapa 1 promete.

## 2. Método

### Embeddings (`extract_cnn6_embeddings.py`, `panns_cnn6.py`)

- Arquitectura CNN6 reimplementada con los mismos nombres de capa que el
  checkpoint. Los pesos cargan en modo estricto.
- Se usa la salida de **fc1** (512 valores, después de ReLU): la capa
  anterior a la cabeza de 527 clases de AudioSet, que se descarta.
- Cada uno de los 6898 ciclos (variante `hp100`, la entrada de v4) se pasa
  **una sola vez** por la red congelada, con su duración real (sin recorte a
  4 s: el pooling final es global).
- El pooling temporal de CNN6 es **máximo + promedio**, así que ya incorpora
  la lección de v1 → v2 (`avances/13`).

**Dos formas de armar la entrada**, porque PANNs se entrenó a 32 kHz y
nuestro audio es de 4 kHz (banda de 2 kHz a propósito, por el riesgo de
atajo por equipo, `avances/14` §2):

| Entrada | Construcción | Ventaja | Desventaja |
|---|---|---|---|
| `panns32k` | Remuestreo a 32 kHz + frontend original de PANNs (STFT 1024 / hop 320, 64 bandas Mel 50-14000 Hz) | Cada banda significa la frecuencia que la red conoce | 34 de 64 bandas (las de más de 2 kHz) quedan vacías |
| `ours64` | Espectrograma propio a 4 kHz: 64 bandas Mel 50-2000 Hz, STFT de 64 ms, paso de 10 ms | Usa toda la resolución | La banda k no corresponde a la frecuencia con la que la red aprendió |

En ambas, la escala es 10·log10 de la potencia Mel, absoluta (sin normalizar
por el máximo), como PANNs, porque la primera capa (bn0) usa las
estadísticas de AudioSet.

**Detalle técnico**: las bandas vacías (−100 dB) generan números
*denormales* que la CPU procesa muy lento.
`torch.set_flush_denormal(True)` da el mismo resultado (verificado,
atol=1e-5) y es ~1.6× más rápido. Tiempos de extracción: 7.4 min
(`ours64`) y 11.3 min (`panns32k`).

### Clasificador (`run_frozen_probe.py`)

**Mismo protocolo que v4**: mismos 5 folds por paciente, BCE con
`pos_weight` natural, mejor época por AUC medio de val, umbrales calibrados
sobre val, predicciones OOF y desglose por equipo y edad. Cada dimensión del
embedding se estandariza con estadísticas **solo de train** de cada fold.
Adam 1e-3, weight decay 1e-4, 100 épocas, batch 256.

- **linear**: 512 → 2 (dos regresiones logísticas; el *linear probe* de la
  literatura).
- **mlp**: 512 → 128 (ReLU, dropout 0.3) → 2.

Sin data augmentation: los embeddings se calculan una vez.

## 3. Resultados (media ± std, 5 folds, test)

| Modelo | Score ICBHI | Se | Sp | AUC crackle | AUC wheeze | F1 crackle | F1 wheeze |
|---|---|---|---|---|---|---|---|
| **v4** — CNN desde cero | **0.559 ± 0.007** | 0.368 | 0.750 | **0.676** | **0.793** | 0.464 | 0.475 |
| CNN6 congelada `ours64` + linear | 0.520 ± 0.032 | 0.285 | 0.754 | 0.649 | 0.733 | 0.373 | 0.306 |
| CNN6 congelada `ours64` + MLP | 0.536 ± 0.023 | 0.284 | 0.789 | 0.648 | 0.758 | 0.384 | 0.384 |
| CNN6 congelada `panns32k` + linear | 0.542 ± 0.033 | 0.374 | 0.710 | 0.657 | 0.787 | 0.472 | 0.467 |
| **CNN6 congelada `panns32k` + MLP** | **0.559 ± 0.024** | 0.357 | 0.761 | **0.675** | 0.773 | 0.449 | 0.471 |

Matriz de confusión OOF de `panns32k` + MLP (filas = real, columnas =
predicho):

```
         normal  crackle  wheeze  both
normal     2803      629     157    53
crackle    1008      792      43    21
wheeze      422      113     301    50
both        223       86     149    48
```

Por grupo de edad (`panns32k` + MLP): adultos 0.550 (Se 0.360 / Sp 0.741);
pediátricos 0.510 (Se **0.110** / Sp 0.911).

## 4. Lectura

- **La CNN6 congelada, sin entrenar ni un peso de la red con nuestro audio,
  iguala a v4**: mismo score (0.559) y mismo AUC de crackle (0.675 contra
  0.676). En wheeze queda algo por debajo (0.773 contra 0.793). Es decir,
  lo que la red aprendió de AudioSet ya contiene tanta información útil
  sobre crackles y wheezes como la que nuestra CNN logra aprender de cero
  con 85 pacientes.
- **La entrada original de PANNs (`panns32k`) le gana a la nuestra
  (`ours64`)** en los dos clasificadores (+0.02 de score, +0.03 de AUC en
  wheeze). Pesa más que cada banda conserve la frecuencia con la que la red
  aprendió que aprovechar toda la resolución: los filtros de la CNN6 son
  específicos de la frecuencia.
- **El MLP le gana al lineal** (+0.017 de score). Hay algo de estructura no
  lineal en el embedding, pero la mayor parte de la información ya es
  linealmente accesible.
- **Menos estable que v4**: desvío entre folds de 0.024 contra 0.007. En
  varios folds el MLP alcanza su mejor época muy temprano (épocas 1-2): con
  512 dimensiones y sin augmentation, el clasificador sobreajusta rápido.
- **La brecha pediátrica persiste, incluso peor** (Se 0.110 contra 0.171
  en v4). No es un problema de la arquitectura: viene de los datos (pocos
  ciclos anormales pediátricos, train dominado por COPD adulto,
  `avances/14` §4).
- **Equipo**: sin atajo evidente. Littmann 3200 sigue siendo el peor grupo
  (0.507), igual que en v4.

## 5. Decisión

El criterio fijado antes de correr era que, si la CNN6 congelada igualaba o
superaba a v4, valía la pena invertir en la etapa 2. **Se cumple**: iguala
el score sin adaptar ningún peso. Si con features genéricas de AudioSet ya
llega al nivel de v4, adaptarlas a nuestros ciclos (fine-tuning) es la vía
con más chances de superarlo.

Para la etapa 2:

- **Entrada `panns32k`** (la que ganó acá). Implica que el fine-tuning
  trabaje con la resolución original (64×401 por 4 s): la opción cara del
  benchmark (~16 min/época en CPU).
- **Hay que decidir dónde entrenar**:
  - **CPU local**: congelar los primeros bloques y pocas épocas (un modelo
    preentrenado suele converger en 10-15). Del orden de 1-2 días por
    corrida de 5 folds.
  - **Google Colab con GPU**: empaquetar los ciclos en un solo archivo
    (~150-300 MB) evita el cuello de botella de I/O que hizo fallar el
    intento anterior (`avances/5`). Con una T4, el fine-tuning completo
    debería tardar minutos por época.
- Idea barata complementaria: **ensamble v4 + CNN6 congelada** (promediar
  probabilidades). Son modelos de naturaleza muy distinta y podrían
  equivocarse en ciclos distintos. Requiere guardar las probabilidades de
  validación de v4 para recalibrar umbrales.

Reportes: `reports/pulmonary_adventitious/v5_cnn6_frozen_{ours64,panns32k}_{linear,mlp}/`.
Embeddings: `data/pulmonary_hp100/cnn6_embeddings_{ours64,panns32k}.npy`.
