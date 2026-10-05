# Proyecto Final CEIA — Pendiente: revisar `pathology_label` antes de la etapa 2

## 1. Motivación

Antes de empezar a entrenar los modelos de patología (etapa 2), el usuario
planteó revisar si conviene seguir usando `pathology_label` (la clase
"unificada" del CSV maestro) tal como está, o volver a las clases
**originales** de cada dataset y manejar las clases minoritarias de cada
uno por separado.

Motivo de fondo: `pathology_label` se diseñó para el **gate**, que necesita
un único split train/val/test estratificado que cubra cardíaco y pulmonar
a la vez (`splits/make_splits.py`). Para lograr eso con clases razonables
de ambos lados, el mapeo ya colapsa algo la granularidad original — ver
`unified_dataset/config.py` y los parsers. La etapa 2 entrena **dos
modelos independientes** (cardíaco y pulmonar), cada uno con su propio
split — ya no hay necesidad de mantener un vocabulario conjunto, así que
vale la pena revisar si ese colapso sigue siendo necesario o conviene
deshacerlo por separado en cada dominio.

## 2. Qué se colapsó exactamente

**CirCor (cardíaco)** — `parsers/parse_circor.py`, `_derive_pathology`:
mapea `Murmur`/`Outcome` (columnas nativas del dataset) a solo 3 clases:

```
Murmur=Present                    → heart_murmur
Murmur=Absent  + Outcome=Normal   → normal_heart
Outcome=Abnormal (murmur Absent o Unknown) → abnormal_heart_unspecified
Murmur=Unknown + Outcome=Normal   → normal_heart   (conservador)
```

Acá casi no hay colapso de *categorías* (CirCor no da un diagnóstico más
fino que Murmur+Outcome a nivel de dataset) — pero sí se **descarta
información real ya presente en el CSV crudo** que no se usa en absoluto
hoy: columnas de detalle del murmullo (timing sistólico/diastólico, grado,
calidad/pitch, ubicación específica del murmullo). Si se quiere una
`pathology_label` cardíaca más rica, está ahí sin necesidad de volver a
descargar nada — solo falta parsearla.

**ICBHI (pulmonar)** — `config.py`, `ICBHI_DIAGNOSIS_MAP`: agrupa
diagnósticos nativos que sí son clínicamente distintos:

```
URTI + LRTI               → respiratory_infection
Bronchiolitis + Asthma    → bronchiolitis_asthma
```

(el resto — Healthy, COPD, Pneumonia, Bronchiectasis — se mapean 1:1 sin
agrupar).

## 3. Números reales (dataset unificado actual, por sujeto)

| Dominio | Clase (`pathology_label` actual) | Sujetos |
|---|---|---|
| Cardíaco | normal_heart | 457 |
| Cardíaco | abnormal_heart_unspecified | 306 |
| Cardíaco | heart_murmur | 179 |
| Pulmonar | copd | 64 |
| Pulmonar | normal_lung | 26 |
| Pulmonar | respiratory_infection (URTI+LRTI) | 16 |
| Pulmonar | bronchiolitis_asthma (Bronchiolitis+Asthma) | 7 |
| Pulmonar | bronchiectasis | 7 |
| Pulmonar | pneumonia | 6 |

Si se separan las dos clases agrupadas de ICBHI en sus componentes nativos,
los grupos ya chicos (7 y 16 sujetos) se dividirían en subgrupos de
probablemente 2-10 sujetos cada uno — de un vistazo a la proporción típica
URTI/LRTI y Bronchiolitis/Asthma en ICBHI, varias de esas subclases
quedarían con **menos sujetos que splits** (train/val/test), haciendo
inviable estratificar de forma significativa (ya es un problema hoy con
`bronchiectasis`/`pneumonia`, documentado en el README como limitación
conocida).

## 4. Opinión / recomendación

Tiene sentido revisar esto, pero con una respuesta distinta para cada
dominio:

- **Cardíaco (CirCor)**: sí vale la pena recuperar el detalle del murmullo
  (timing/grado/calidad) del CSV crudo en vez de la etiqueta 3-clases
  actual — hay volumen razonable (179 sujetos con murmullo) y es
  información real que hoy se tira. Esto es indepen­diente del split
  conjunto del gate: se puede agregar como columna nueva sin tocar
  `pathology_label` ni romper nada de lo ya hecho.
- **Pulmonar (ICBHI)**: separar `respiratory_infection` y
  `bronchiolitis_asthma` en sus componentes nativos **probablemente no
  conviene** — los números de la sección 3 sugieren que quedarían
  subclases de un puñado de sujetos, peor que lo que ya tenemos. Antes de
  decidir, conviene mirar la proporción real URTI/LRTI y
  Bronchiolitis/Asthma en `data/raw/ICBHI/patient_diagnosis.csv` (no
  revisado todavía) para confirmar esto con números exactos en vez de una
  estimación.
- En cualquier caso, el punto de fondo del usuario es correcto: como ahora
  cada modelo tiene su propio split, el manejo de clases minoritarias
  (agrupar, oversamplear, class weighting, o simplemente aceptar la
  limitación) se puede decidir **por separado para cardíaco y pulmonar**,
  en vez de estar atado a lo que convenía para el split conjunto del gate.

## 5. Pendiente

- Antes de arrancar el entrenamiento de los modelos de patología: decidir
  si se agrega el detalle de murmullo de CirCor como columna nueva, y si
  se confirma (con los números reales de ICBHI) que separar
  `respiratory_infection`/`bronchiolitis_asthma` no conviene o si alguna sí
  tiene volumen suficiente.
- Si se cambia `pathology_label` o se agregan columnas nuevas, hay que
  re-correr `unified_dataset/parsers/build_dataset.py` y
  `splits/make_splits.py` (los splits de la etapa 2 van a ser nuevos de
  todos modos, por sujeto y por dominio, no reutilizan el split del gate).
