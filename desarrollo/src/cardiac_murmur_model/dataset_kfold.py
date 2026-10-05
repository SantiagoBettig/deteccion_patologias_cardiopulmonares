"""
cardiac_murmur_model/dataset_kfold.py
=========================================
Dataset para la validación k-fold del modelo de murmullo (ver
cardiac_murmur_model/run_kfold.py). Los WAV ya están guardados físicamente
en carpetas train/val/test del split ÚNICO original (ver
preprocessing/make_cardiac_cycle_windows.py) — un sujeto que en el split
original estaba en "train" puede ser "test" en un fold distinto, así que
NO se puede resolver la ruta del audio a partir de la columna "split" del
fold. En vez de eso, se arma una sola vez un índice window_id -> ruta
física (escaneando las 3 carpetas), independiente del fold.

Misma extracción de features que cardiac_murmur_model/dataset.py
(MFCC/log-Mel, normalización z-score, augmentation) — código duplicado a
propósito, para no acoplar el pipeline de k-fold (más que nada un chequeo
de robustez del resultado) con el dataset "de producción".
"""

import numpy as np
import librosa
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


def build_window_path_index(windows_dir: Path) -> dict[str, Path]:
    """Escanea windows_dir/{train,val,test}/*.wav una sola vez."""
    index = {}
    for split in ("train", "val", "test"):
        split_dir = windows_dir / split
        if not split_dir.exists():
            continue
        for wav_path in split_dir.glob("*.wav"):
            index[wav_path.stem] = wav_path
    return index


class KFoldMurmurDataset(Dataset):
    def __init__(self, fold_df: pd.DataFrame, path_index: dict[str, Path], split: str,
                 n_mels: int = 64, n_fft: int = 512, hop_length: int = 160,
                 feature_type: str = "mfcc", n_mfcc: int = 20,
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0)):
        self.df = fold_df[fold_df["split"] == split].reset_index(drop=True)
        self.path_index = path_index
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
        wav_path = self.path_index[row["window_id"]]
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

        x = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        label = 1.0 if row["pathology_label"] == "heart_murmur" else 0.0
        y_t = torch.tensor(label, dtype=torch.float32)
        return x, y_t

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(*self.gain_db_range)
        y = y * (10.0 ** (gain_db / 20.0))
        noise = np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
        return np.clip(y + noise, -1.0, 1.0)

    @staticmethod
    def _normalize(feat: np.ndarray) -> np.ndarray:
        mean, std = feat.mean(), feat.std()
        if std < 1e-6:
            return feat - mean
        return (feat - mean) / std


def collate_pad(batch):
    xs, ys = zip(*batch)
    max_t = max(x.shape[-1] for x in xs)
    padded = [
        torch.nn.functional.pad(x, (0, max_t - x.shape[-1])) if x.shape[-1] < max_t else x
        for x in xs
    ]
    return torch.stack(padded), torch.stack(ys)
