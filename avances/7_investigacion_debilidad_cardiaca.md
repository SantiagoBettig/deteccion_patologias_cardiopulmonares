# Proyecto Final CEIA — Investigación sobre la debilidad persistente en cardíaco

## 1. Motivación

Tras el resultado de gate v5 (`6_gate_v5_mfcc.md`): MFCC mejoró el F1 en
HLS-CMDS (0.816 → 0.847) pero el recall de cardíaco no mejoró (76.0% →
74.0%, incluso levemente peor) — la ganancia es por mayor precisión, no por
resolver la debilidad real. El usuario planteó dos ideas antes de seguir
iterando con más features/arquitectura:

1. Buscar otro dataset público que permita validar los resultados (además
   de HLS-CMDS).
2. Analizar si los resultados tienen sentido con el contenido real de
   HLS-CMDS — capaz algún parámetro de cómo se generó ese dataset explica
   por qué nunca podría aprenderse bien solo con CirCor + ICBHI.

## 2. Investigación — protocolo de grabación de HLS-CMDS

Búsqueda web sobre el dataset y su paper descriptor (Torabi, Shirani,
Reilly, *IEEE Data Descriptions*, doi: 10.1109/IEEEDATA.2025.3566012).
Hallazgos:

- Es un **maniquí/simulador de paciente** ("patient simulators"), grabado
  con asistencia del *Mohawk Institute for Applied Health Sciences* — un
  instituto de formación clínica. Este tipo de simuladores de alta
  fidelidad (usados para entrenar personal de salud) típicamente
  **reproducen grabaciones reales pregrabadas a través de un parlante
  interno bajo una "piel" sintética** en el torso, en vez de generar el
  sonido puramente de forma mecánica. Si es el caso acá (no se pudo
  confirmar el modelo exacto de maniquí ni de estetoscopio digital desde
  las páginas públicas del dataset — Kaggle/Zenodo/GitHub no detallan
  marca/modelo ni mecanismo de generación; solo el README completo del
  dataset o el paper completo lo tendrían), la cadena
  parlante→piel sintética→estetoscopio actúa como un **filtro acústico
  adicional** que no está presente en CirCor/ICBHI (grabados directo sobre
  el paciente real). Esto es consistente con la hipótesis de la sección 7
  de `4_gate_model_baseline_y_sesgo.md`: una función de transferencia
  distinta podría alterar sistemáticamente el balance espectral
  cardíaco/pulmonar sin que el contenido "semántico" del sonido cambie.
- **No se pudo confirmar con certeza** el mecanismo exacto (mecánico puro
  vs. playback bajo piel sintética) ni el modelo de manikin/estetoscopio
  con las fuentes públicas disponibles (Kaggle, GitHub, Zenodo, IEEE
  DataPort no publican esos detalles en sus páginas de listado). El
  `HLS_CMDS_README.txt` incluido en la descarga del dataset (Zenodo/Mendeley)
  y el paper completo (ResearchGate devolvió 403 al intentar leerlo) son
  los lugares donde sí debería estar esa información — **pendiente**: si se
  quiere seguir esta línea, hay que descargar el dataset y leer ese README
  directamente, o conseguir el PDF del paper (via institución/biblioteca).

## 3. Investigación — otro dataset real con cardíaco + pulmonar

Búsqueda de un dataset público con sonidos cardíacos **y** pulmonares
grabados sobre pacientes reales (no maniquí) con el mismo equipo — el
mismo rol que cumple HLS-CMDS pero sin el posible artefacto del maniquí.

**No se encontró un dataset así.** Lo más cercano identificado es un paper
(Training one model to detect heart and lung sound events from single point
auscultations, arXiv:2301.06078) que combina tres datasets *distintos* (2016
PhysioNet/CinC Challenge para cardíaco, 2022 George Moody Challenge, y
HF_Lung_V1 para pulmonar) en un esquema multi-tarea — es decir, ni siquiera
ese trabajo tenía un dataset combinado real; enfrentó el mismo problema que
nosotros (fuentes separadas) y no lo resolvió con un dataset unificado.
Esto sugiere que **HLS-CMDS podría ser, hoy, el único dataset público con
ambos tipos de sonido en pacientes/fuente comparable** — aunque sea un
maniquí. Si se quiere insistir en esta idea, lo más prometedor sería:
- Revisar el listado de 7 datasets del benchmark **CaReCoS**
  (arXiv:2607.03356, cardíaco/respiratorio/tos) — no confirmado si alguno
  tiene ambos tipos del mismo sujeto, pero es el punto de partida más
  organizado para no tener que buscar dataset por dataset.
- Buscar puntualmente si el dataset **HF_Lung_V1/V2** (usado para pulmonar)
  o el **PASCAL/PhysioNet 2016** (cardíaco) documentan si alguno de sus
  sujetos tiene también grabaciones del otro tipo de sonido (poco probable,
  pero no descartado sin revisar el detalle de cada uno).

## 4. Idea adicional (propuesta durante esta conversación) — features de periodicidad/ritmo

Tanto log-Mel como MFCC son features de **timbre** (qué tan brillante/opaco
suena, banda por banda) — exactamente la propiedad que está invertida entre
dominios (sección 7 de `4_...md`). Una fuente de información distinta, que
la CNN actual no explota de forma explícita, es el **patrón temporal
rítmico**:

- Un ciclo cardíaco (S1-S2) dura típicamente ~0.6-1s (60-100 lpm), con dos
  eventos cortos y bien definidos por ciclo.
- Un ciclo respiratorio dura ~2-5s (12-20 rpm), con una envolvente más
  continua (inspiración/espiración).

Esta diferencia de periodicidad es una propiedad **funcional** del sonido
(cuántas veces late/respira por segundo), no del timbre — en principio
debería sobrevivir aunque el maniquí altere el brillo espectral, porque el
timing de los eventos que reproduce sigue siendo el de un corazón/pulmón
real (o al menos plausible). Una forma simple de explotarla sin rehacer la
arquitectura: agregar como canal adicional (o feature separada) la
autocorrelación de la envolvente de energía de la ventana, o el espaciado
entre picos de energía (onset detection) — el modelo podría aprender a
distinguir por la cadencia además de por el timbre.

**No implementado** — se documenta como opción a evaluar, no se decidió
todavía si es la próxima prioridad.

## 5. Discusión y prioridad sugerida

En orden de costo/beneficio, para decidir con el usuario:

1. **Más barato y ya factible ahora**: leer el `HLS_CMDS_README.txt` /
   paper completo (descargando el dataset o consiguiendo el PDF) para
   confirmar o descartar el mecanismo de generación de sonido del maniquí.
   Si se confirma que es playback bajo piel sintética, refuerza fuerte la
   hipótesis de brecha de dominio y baja la prioridad de seguir iterando
   sobre features — el techo de HLS-CMDS podría ser artificialmente bajo
   por el propio dataset, no por el modelo.
2. **Costo medio, alto valor si funciona**: probar features de
   periodicidad/ritmo (sección 4) — ataca la causa raíz (dependencia del
   timbre) en vez de seguir variando cómo se comprime el timbre (que es lo
   que MFCC ya probó y no alcanzó).
3. **Costo alto, beneficio incierto**: seguir buscando un dataset real
   combinado (sección 3) — la búsqueda inicial no encontró uno; insistir
   acá tiene retorno incierto comparado con 1 y 2.
4. Independiente de estas tres: la agregación por grabación completa (voto
   mayoritario por `file_id`, ya prevista en `avances/4_...md` sección 9)
   sigue pendiente y es barata — vale la pena probarla en paralelo,
   cualquiera sea el camino elegido.

## 6. Próximos pasos

Pendiente de decisión del usuario sobre cuál de las líneas de la sección 5
seguir primero.
