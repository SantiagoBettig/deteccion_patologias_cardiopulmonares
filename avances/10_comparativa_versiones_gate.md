# Proyecto Final CEIA — Comparativa de versiones del modelo gate

Cuadro resumen de todas las corridas del gate (cardíaco vs. pulmonar), a
pedido del usuario. Cada fila referencia el documento de `avances/` con el
detalle completo y la carpeta de `reports/gate/runs/` con los artefactos
(checkpoint, curvas de entrenamiento, matrices de confusión).

| # | Referencia | Modificación introducida (sobre la fila anterior) | Test Acc | Test Prec | Test Rec | Test F1 | HLS Acc | HLS Prec | HLS Rec | HLS F1 | Resultado |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | [`avances/4`](4_gate_model_baseline_y_sesgo.md) §2 · `runs/v1_baseline_sin_mitigacion/` | Baseline: CNN 2D sobre espectrograma log-Mel (64 bandas), sin normalización especial ni augmentation. | 0.9998 | 0.9998 | 1.0000 | 0.9999 | 0.545 | 0.708 | 0.153 | 0.252 | Descartado — *shortcut learning* confirmado (el modelo aprendía `source_db`, no el sonido) |
| v2 | [`avances/4`](4_gate_model_baseline_y_sesgo.md) §5 · `runs/v2_normalizacion_augmentation/` | + Normalización z-score por instancia del espectrograma + augmentation de forma de onda (ganancia ±3 dB, ruido gaussiano std=0.005) | 0.9961 | 0.9953 | 1.0000 | 0.9976 | 0.818 | 0.852 | 0.770 | 0.809 | Mejora grande — mitigación funcionó |
| v3 | [`avances/4`](4_gate_model_baseline_y_sesgo.md) §6 · `runs/v3_specaugment/` | + SpecAugment (2 máscaras de frecuencia + 2 de tiempo) y augmentation de forma de onda más fuerte (±6 dB, ruido std=0.01) | 0.9857 | 0.9947 | 0.9876 | 0.9912 | 0.793 | 0.994 | 0.590 | 0.741 | **Descartado** — empeoró (modelo demasiado conservador, recall cardíaco cae a 59%) |
| v4 | [`avances/4`](4_gate_model_baseline_y_sesgo.md) §8 · `runs/v4_class_weighting/` | Sobre v2 (revierte SpecAugment): + `pos_weight` en `BCEWithLogitsLoss` por el desbalance cardíaco:pulmonar ≈3.5:1 | 0.9931 | 0.9950 | 0.9964 | 0.9957 | 0.828 | 0.880 | 0.760 | 0.816 | Prácticamente igual a v2 — confirma que el desbalance no era la causa principal |
| v5 | [`avances/6`](6_gate_v5_mfcc.md) · `runs/v5_mfcc/` | Sobre v4: feature de entrada cambiado de espectrograma log-Mel a **MFCC** (20 coeficientes), mismo banco de filtros Mel de base | 0.9753 | 0.9825 | 0.9871 | 0.9848 | 0.867 | 0.991 | 0.740 | 0.847 | **Versión final** — mejor F1 en HLS, aunque por mayor precisión, no por resolver el recall de cardíaco |
| v6 | [`avances/8`](8_gate_v6_periodicity.md) · `runs/v6_periodicity/` | Sobre v5: + rama adicional (MLP) con feature de periodicidad rítmica de la envolvente de energía (autocorrelación, rango cardíaco vs. respiratorio) | 0.9753 | 0.9913 | 0.9781 | 0.9847 | 0.748 | 1.000 | 0.497 | 0.664 | **Descartado** — empeoró fuerte (recall cardíaco cae a 49.7%) |

**Notas de lectura:**
- "Test" = split de test propio (CirCor+ICBHI, mismo origen que train/val).
  "HLS" = validación externa con HLS-CMDS, dataset nunca visto en
  entrenamiento y el único con ambos tipos de sonido grabados con el mismo
  equipo — el número que realmente importa para saber si el modelo
  generaliza (ver `preprocessing/make_hls_windows.py`).
- Todas las corridas comparten: split train/val/test por `subject_id`
  (`splits/make_splits.py`), ventaneo fijo 5s/50% overlap, arquitectura
  base `GateCNN` (3 bloques conv+pool + global average pooling, 23.585
  parámetros — sin cambios en ninguna versión, salvo la rama extra de v6),
  15 épocas, Adam lr=1e-3, batch_size=32.
- No incluye `gate_model/diagnostics_source_db.py` (no es una versión del
  gate, es un diagnóstico auxiliar que confirmó la causa de v1 — ver
  `avances/4` §4).
- Decisión final documentada en `avances/8` §6: se deja de iterar sobre el
  gate; **v5 queda como versión de cierre** para esta etapa, con la
  debilidad en recall de cardíaco sobre HLS-CMDS como limitación conocida
  (ver `avances/7_investigacion_debilidad_cardiaca.md`).
