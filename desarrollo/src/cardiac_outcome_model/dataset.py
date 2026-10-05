"""
cardiac_outcome_model/dataset.py
===================================
Dataset de PyTorch para el modelo de outcome clínico cardíaco (binario).
Carga cada ventana de audio recortada por ciclo (generada por
preprocessing/make_cardiac_cycle_windows.py) y calcula su feature MFCC al
vuelo. label = 1 si clinical_outcome=="abnormal", 0 si "normal" — a
diferencia de pathology_label, clinical_outcome es el Outcome crudo de
CirCor, independiente de si el murmullo es audible en esa ubicación (ver
unified_dataset/parsers/parse_circor.py y
avances/10_cardiaco_split_dos_tareas.md).

Usa TODAS las ventanas cardíacas (con o sin murmullo) — el outcome es
ortogonal a la tarea de murmullo (cardiac_murmur_model/).

Independiente de cardiac_murmur_model/dataset.py (código duplicado a
propósito) para poder ajustar cada tarea por separado. Las ventanas no
tienen duración fija — collate_pad rellena con 0 hasta el máximo del batch.
"""

import numpy as np
import librosa
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class OutcomeDataset(Dataset):
    def __init__(self, index_csv, windows_dir, split: str,
                 n_mels: int = 64, n_fft: int = 512, hop_length: int = 160,
                 feature_type: str = "mfcc", n_mfcc: int = 20,
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0),
                 freq_mask_param: int = 0, time_mask_param: int = 0,
                 n_freq_masks: int = 0, n_time_masks: int = 0):
        if feature_type not in ("logmel", "mfcc"):
            raise ValueError(f"feature_type debe ser 'logmel' o 'mfcc', recibido: {feature_type!r}")
        df = pd.read_csv(index_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.windows_dir = Path(windows_dir)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.augment = augment
        self.noise_std = noise_std
        self.gain_db_range = gain_db_range
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav_path = self.windows_dir / row["split"] / f"{row['window_id']}.wav"
        y, sr = librosa.load(wav_path, sr=None, mono=True)

        if self.augment:
            y = self._augment_waveform(y)

        if self.feature_type == "mfcc":
            feat = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length,
            )
        else:
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
            )
            feat = librosa.power_to_db(mel, ref=np.max)
        feat = self._normalize(feat)

        if self.augment:
            feat = self._spec_augment(feat)

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, n_feat_rows, n_frames)
        label = 1.0 if row["clinical_outcome"] == "abnormal" else 0.0
        y_t = torch.tensor(label, dtype=torch.float32)
        return x, y_t

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

    def _spec_augment(self, feat: np.ndarray) -> np.ndarray:
        feat = feat.copy()
        n_rows, n_frames = feat.shape

        for _ in range(self.n_freq_masks):
            width = np.random.randint(0, self.freq_mask_param + 1)
            if width == 0 or width >= n_rows:
                continue
            f0 = np.random.randint(0, n_rows - width)
            feat[f0:f0 + width, :] = 0.0

        for _ in range(self.n_time_masks):
            width = np.random.randint(0, self.time_mask_param + 1)
            if width == 0 or width >= n_frames:
                continue
            t0 = np.random.randint(0, n_frames - width)
            feat[:, t0:t0 + width] = 0.0

        return feat


def collate_pad(batch):
    """Rellena con 0 (la media, tras la normalización z-score) hasta el
    máximo de frames del batch — las ventanas no tienen duración fija."""
    xs, ys = zip(*batch)
    max_t = max(x.shape[-1] for x in xs)
    padded = [
        torch.nn.functional.pad(x, (0, max_t - x.shape[-1])) if x.shape[-1] < max_t else x
        for x in xs
    ]
    return torch.stack(padded), torch.stack(ys)
