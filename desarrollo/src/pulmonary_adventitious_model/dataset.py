"""
pulmonary_adventitious_model/dataset.py
==========================================
Dataset de PyTorch para el modelo de sonidos adventicios por ciclo
respiratorio (multi-etiqueta: crackle sí/no, wheeze sí/no). Carga cada
ciclo generado por preprocessing/make_pulmonary_cycles.py y calcula su
feature al vuelo.

Diferencias deliberadas respecto a cardiac_murmur_model/dataset.py:

- **Resolución temporal más fina** (n_fft=256 / hop=64 a 4 kHz: ventana de
  64 ms, paso de 16 ms; el de murmullo usa 128 ms / 40 ms). Un crackle es un
  transitorio de menos de ~20 ms: con un paso de 40 ms quedaría diluido en
  uno o dos frames.
- **log-Mel por defecto** en vez de MFCC. Un wheeze es una línea tonal
  angosta (y sus armónicos); los 20 primeros coeficientes MFCC resumen la
  envolvente espectral y tienden a suavizar justamente ese detalle fino. En
  murmullo empataron (avances/10 §6); acá la hipótesis física es más
  fuerte, pero MFCC queda disponible (feature_type="mfcc") para comparar.
- **Duración fija por padding en el dominio de la feature**: se normaliza
  (z-score) el ciclo real y después se rellena con 0 (= la media) hasta
  TARGET_SEC. No se repite la señal ("repeat padding", común en la
  literatura de ICBHI) porque cada unión introduce un salto brusco — un
  transitorio de banda ancha que se parece a un crackle. Los ciclos más
  largos que TARGET_SEC se recortan (al azar en train, centrado en eval).
- Los WAV se cachean en memoria la primera vez que se leen (~7000 ciclos
  cortos, unos cientos de MB): evita releer del disco en cada época.
"""

import numpy as np
import librosa
import pandas as pd
import soundfile as sf
import torch
from pathlib import Path
from torch.utils.data import Dataset

LABEL_COLUMNS = ["crackle", "wheeze"]


class CycleDataset(Dataset):
    def __init__(self, df: pd.DataFrame, cycles_dir, target_sec: float = 4.0,
                 feature_type: str = "logmel", n_mels: int = 40, n_mfcc: int = 20,
                 n_fft: int = 256, hop_length: int = 64,
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0), wav_cache: dict = None):
        if feature_type not in ("logmel", "mfcc"):
            raise ValueError(f"feature_type debe ser 'logmel' o 'mfcc', recibido: {feature_type!r}")
        self.df = df.reset_index(drop=True)
        self.cycles_dir = Path(cycles_dir)
        self.target_sec = target_sec
        self.feature_type = feature_type
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.augment = augment
        self.noise_std = noise_std
        self.gain_db_range = gain_db_range
        # Cache compartible entre datasets (train/val/test de un mismo fold, o
        # entre folds): el audio de un ciclo no cambia, solo su split.
        self.wav_cache = wav_cache if wav_cache is not None else {}
        self.labels = self.df[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def _load(self, cycle_id: str) -> tuple[np.ndarray, int]:
        if cycle_id not in self.wav_cache:
            y, sr = sf.read(self.cycles_dir / f"{cycle_id}.wav", dtype="float32")
            self.wav_cache[cycle_id] = (y, sr)
        return self.wav_cache[cycle_id]

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y, sr = self._load(row["cycle_id"])

        target_len = int(self.target_sec * sr)
        if len(y) > target_len:
            start = (np.random.randint(0, len(y) - target_len + 1) if self.augment
                     else (len(y) - target_len) // 2)
            y = y[start:start + target_len]

        if self.augment:
            y = self._augment_waveform(y)

        feat = self._normalize(self._feature(y, sr))

        target_frames = 1 + target_len // self.hop_length
        if feat.shape[1] < target_frames:
            feat = np.pad(feat, ((0, 0), (0, target_frames - feat.shape[1])))
        feat = feat[:, :target_frames]

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, n_rows, n_frames)
        return x, torch.from_numpy(self.labels[idx])

    def _feature(self, y: np.ndarray, sr: int) -> np.ndarray:
        # n_fft no puede superar la señal (ciclos muy cortos): se rellena la
        # señal con ceros hasta n_fft solo para este cálculo.
        if len(y) < self.n_fft:
            y = np.pad(y, (0, self.n_fft - len(y)))
        if self.feature_type == "mfcc":
            return librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length,
            )
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length,
        )
        return librosa.power_to_db(mel, ref=np.max)

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(*self.gain_db_range)
        y = y * (10.0 ** (gain_db / 20.0))
        noise = np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
        return np.clip(y + noise, -1.0, 1.0)

    @staticmethod
    def _normalize(feat: np.ndarray) -> np.ndarray:
        """Z-score por instancia: quita nivel/offset propios de la grabación."""
        mean, std = feat.mean(), feat.std()
        if std < 1e-6:
            return feat - mean
        return (feat - mean) / std
