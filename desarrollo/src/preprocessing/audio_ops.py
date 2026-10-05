"""
preprocessing/audio_ops.py
===========================
Operaciones de preprocesamiento de audio, compartidas por el pipeline del
modelo gate y (más adelante) por el recorte por ciclo de los modelos de
patología.

Orden de aplicación (ya decidido, ver avances/3_preliminar_eda.md):
    1. Filtrado pasa-bajos (~2048 Hz) — antes del resample, evita aliasing.
    2. Resample a la frecuencia objetivo (4000 Hz).
    3. Normalización de amplitud por RMS.

Importante: el sample rate real de cada archivo se lee del propio WAV
(librosa con sr=None), nunca de la columna `sample_rate_hz` del CSV maestro
— esa columna es una inferencia por equipo de grabación y se detectó al
menos un archivo (ICBHI, clase copd) donde no coincide con el sample rate
real (ver avances/3_preliminar_eda.md).
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt


def load_audio(path) -> tuple[np.ndarray, int]:
    """Carga un WAV respetando su sample rate real (no confía en metadatos)."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    return y, sr


def lowpass_filter(y: np.ndarray, sr: int, cutoff: float = 2048.0, order: int = 4) -> np.ndarray:
    """
    Filtro pasa-bajos Butterworth. El cutoff se recorta a un margen seguro
    por debajo de Nyquist (sr/2) para no romper el diseño del filtro en
    archivos que ya nacen a un sample rate bajo (ej. CirCor, 4000 Hz nativo,
    Nyquist=2000 Hz < 2048 Hz).
    """
    nyquist = sr / 2
    effective_cutoff = min(cutoff, nyquist * 0.98)
    sos = butter(order, effective_cutoff, btype="low", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def highpass_filter(y: np.ndarray, sr: int, cutoff: float, order: int = 4) -> np.ndarray:
    """
    Filtro pasa-altos Butterworth (fase cero). Opcional — solo lo usa la
    rama pulmonar para atenuar los sonidos cardíacos (S1/S2, energía
    concentrada por debajo de ~150 Hz) que se cuelan en las grabaciones de
    pulmón (ver avances/14).
    """
    sos = butter(order, cutoff, btype="high", fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def resample_audio(y: np.ndarray, sr: int, target_sr: int = 4000) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return y, sr
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_resampled, target_sr


def rms_normalize(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """Normaliza por energía RMS. Deja señales de silencio (RMS~0) sin tocar."""
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms < 1e-8:
        return y
    y_norm = y * (target_rms / current_rms)
    return np.clip(y_norm, -1.0, 1.0)


def preprocess(path, target_sr: int = 4000, cutoff: float = 2048.0,
               highpass_cutoff: float | None = None) -> tuple[np.ndarray, int]:
    """Encadena filtrado -> resample -> normalización RMS. highpass_cutoff
    (Hz) agrega un pasa-altos antes del resample; None = comportamiento de
    siempre."""
    y, sr = load_audio(path)
    y = lowpass_filter(y, sr, cutoff=cutoff)
    if highpass_cutoff is not None:
        y = highpass_filter(y, sr, cutoff=highpass_cutoff)
    y, sr = resample_audio(y, sr, target_sr=target_sr)
    y = rms_normalize(y)
    return y, sr
