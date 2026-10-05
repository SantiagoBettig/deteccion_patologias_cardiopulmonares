"""
gate_model/dataset_periodicity.py
=====================================
Dataset del gate v6: igual que GateDataset (gate_model/dataset.py) pero
además devuelve un vector de periodicidad rítmica (gate_model/periodicity.py)
calculado sobre la MISMA forma de onda (ya augmentada, si corresponde) que
se usa para el feature de timbre — para que ambas ramas del modelo vean
exactamente la misma instancia.

Ver avances/8_gate_v6_periodicity.md.
"""

import numpy as np
import librosa
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

from gate_model.periodicity import periodicity_features, N_PERIODICITY_FEATURES


class GatePeriodicityDataset(Dataset):
    def __init__(self, index_csv, windows_dir, split: str,
                 label_column: str = "sound_type", positive_value: str = "cardiac",
                 n_mels: int = 64, n_fft: int = 512, hop_length: int = 160,
                 feature_type: str = "mfcc", n_mfcc: int = 20,
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0)):
        if feature_type not in ("logmel", "mfcc"):
            raise ValueError(f"feature_type debe ser 'logmel' o 'mfcc', recibido: {feature_type!r}")
        df = pd.read_csv(index_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.windows_dir = Path(windows_dir)
        self.label_column = label_column
        self.positive_value = positive_value
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.feature_type = feature_type
        self.n_mfcc = n_mfcc
        self.augment = augment
        self.noise_std = noise_std
        self.gain_db_range = gain_db_range

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

        periodicity = periodicity_features(y, sr, hop_length=self.hop_length)

        x_spec = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)  # (1, n_feat_rows, n_frames)
        x_period = torch.tensor(periodicity, dtype=torch.float32)      # (N_PERIODICITY_FEATURES,)
        label = 1.0 if row[self.label_column] == self.positive_value else 0.0
        y_t = torch.tensor(label, dtype=torch.float32)
        return x_spec, x_period, y_t

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(*self.gain_db_range)
        y = y * (10.0 ** (gain_db / 20.0))
        noise = np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
        return np.clip(y + noise, -1.0, 1.0)

    @staticmethod
    def _normalize(feat: np.ndarray) -> np.ndarray:
        """Z-score por instancia: quita nivel/offset propios del dataset de origen."""
        mean, std = feat.mean(), feat.std()
        if std < 1e-6:
            return feat - mean
        return (feat - mean) / std
