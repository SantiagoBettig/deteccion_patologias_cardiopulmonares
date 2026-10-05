"""
gate_model/periodicity.py
============================
Feature de periodicidad rítmica de la envolvente de energía — hipótesis
alternativa a los features de timbre (log-Mel/MFCC) para el gate, propuesta
en avances/7_investigacion_debilidad_cardiaca.md tras ver que MFCC (v5) no
resolvió la debilidad en cardíaco de la validación externa (HLS-CMDS). Ver
avances/8_gate_v6_periodicity.md para el detalle y los resultados.

Un ciclo cardíaco (S1-S2) dura ~0.33-1.5s (40-180 lpm); un ciclo
respiratorio ~1.5-7.5s (8-40 rpm). Esta periodicidad es una propiedad
funcional del sonido (cuántas veces late/respira por segundo), no de su
timbre — en principio debería sobrevivir a diferencias de brillo espectral
entre dominios de grabación (ver avances/4_gate_model_baseline_y_sesgo.md,
sección 7).
"""

import numpy as np
import librosa

HEART_PERIOD_RANGE_S = (60.0 / 180.0, 60.0 / 40.0)   # 0.33s - 1.50s (40-180 lpm)
BREATH_PERIOD_RANGE_S = (60.0 / 40.0, 60.0 / 8.0)     # 1.50s - 7.50s (8-40 rpm)
N_PERIODICITY_FEATURES = 3


def periodicity_features(y: np.ndarray, sr: int, hop_length: int = 160) -> np.ndarray:
    """
    Autocorrelación de la envolvente de energía (RMS por frame), resumida en
    3 valores: fuerza de periodicidad en el rango cardíaco, en el rango
    respiratorio, y su diferencia (positivo => la ventana "suena" más
    cardíaca en términos de ritmo).
    """
    envelope = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    envelope = envelope - envelope.mean()

    if np.allclose(envelope, 0.0):
        return np.zeros(N_PERIODICITY_FEATURES, dtype=np.float32)

    autocorr = librosa.autocorrelate(envelope)
    autocorr = autocorr / (autocorr[0] + 1e-8)  # normalizada: autocorr[0] == 1

    frame_rate = sr / hop_length  # frames por segundo de la envolvente
    max_lag = len(autocorr) - 1

    def _max_in_range(period_range_s: tuple) -> float:
        lag_min = max(1, int(round(period_range_s[0] * frame_rate)))
        lag_max = min(max_lag, int(round(period_range_s[1] * frame_rate)))
        if lag_min > lag_max:
            return 0.0
        return float(autocorr[lag_min:lag_max + 1].max())

    heart_score = _max_in_range(HEART_PERIOD_RANGE_S)
    breath_score = _max_in_range(BREATH_PERIOD_RANGE_S)
    return np.array([heart_score, breath_score, heart_score - breath_score], dtype=np.float32)
