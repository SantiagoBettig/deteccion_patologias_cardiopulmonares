"""
cardiac_murmur_model/dataset.py
==================================
Dataset de PyTorch para el modelo de murmullo cardíaco (binario). Carga
cada ventana de audio recortada por ciclo (generada por
preprocessing/make_cardiac_cycle_windows.py) y calcula su feature MFCC al
vuelo. label = 1 si pathology_label=="heart_murmur", 0 en caso contrario
(normal_heart o abnormal_heart_unspecified) — pathology_label ya está
calculado por ubicación (ver unified_dataset/parsers/parse_circor.py), así
que alcanza sin tocar clinical_outcome.

Independiente de cardiac_outcome_model/dataset.py (código duplicado a
propósito, ver avances/10_cardiaco_split_dos_tareas.md) para poder ajustar
cada tarea por separado.

Las ventanas no tienen duración fija (N_CYCLES ciclos, no N segundos) —
collate_pad rellena con 0 hasta el máximo del batch para poder apilarlas.
"""

import numpy as np
import librosa
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset

# Orden fijo para el one-hot de edad — el mismo vocabulario que
# unified_dataset/parsers/parse_circor.py:_normalize_age_category.
AGE_CATEGORIES = ["neonate", "infant", "child", "adolescent", "adult"]
# sexo(1) + flag(1) + edad one-hot(5) + flag(1) + altura(1) + flag(1) + peso(1) + flag(1)
META_DIM = 1 + 1 + len(AGE_CATEGORIES) + 1 + 1 + 1 + 1 + 1

# Orden fijo para el one-hot de ubicación de auscultación — el vocabulario
# unificado de LOCATION_MAP (unified_dataset/config.py) que aparece en
# CirCor: aortic/pulmonic/tricuspid/mitral + "other" (Phc, muy poco usado).
LOCATIONS = ["apex", "right_upper_sternal_border", "left_upper_sternal_border",
             "left_lower_sternal_border", "other"]
LOCATION_DIM = len(LOCATIONS)


def encode_location(row) -> np.ndarray:
    """Codifica auscultation_location como one-hot — a diferencia de la
    metadata demográfica, la ubicación está mecánicamente ligada a si un
    murmullo se escucha o no (ver avances/10, sección 11)."""
    onehot = np.zeros(LOCATION_DIM, dtype=np.float32)
    loc = row.get("auscultation_location")
    if not pd.isna(loc) and loc in LOCATIONS:
        onehot[LOCATIONS.index(loc)] = 1.0
    return onehot


def compute_meta_stats(df: pd.DataFrame) -> dict:
    """Media/std de altura y peso — calcular UNA VEZ sobre train y reusar
    en val/test (ver cardiac_murmur_model/train_multimodal.py), para no
    filtrar estadísticas de val/test al entrenamiento."""
    return {
        "height_mean": df["height_cm"].mean(),
        "height_std": df["height_cm"].std() or 1.0,
        "weight_mean": df["weight_kg"].mean(),
        "weight_std": df["weight_kg"].std() or 1.0,
    }


def encode_metadata(row, meta_stats: dict) -> np.ndarray:
    """Codifica sexo/edad/altura/peso de una fila en un vector fijo de
    META_DIM valores, con un flag de "faltante" por cada campo que puede
    ser NaN (ver avances/10_cardiaco_split_dos_tareas.md, sección 8)."""
    sex = row.get("sex")
    sex_missing = 1.0 if pd.isna(sex) else 0.0
    sex_val = 0.5 if pd.isna(sex) else (1.0 if sex == "M" else 0.0)

    age_onehot = np.zeros(len(AGE_CATEGORIES), dtype=np.float32)
    age = row.get("age_category")
    age_missing = 1.0 if pd.isna(age) else 0.0
    if not pd.isna(age) and age in AGE_CATEGORIES:
        age_onehot[AGE_CATEGORIES.index(age)] = 1.0

    height = row.get("height_cm")
    height_missing = 1.0 if pd.isna(height) else 0.0
    height_z = 0.0 if pd.isna(height) else (height - meta_stats["height_mean"]) / meta_stats["height_std"]

    weight = row.get("weight_kg")
    weight_missing = 1.0 if pd.isna(weight) else 0.0
    weight_z = 0.0 if pd.isna(weight) else (weight - meta_stats["weight_mean"]) / meta_stats["weight_std"]

    return np.concatenate([
        [sex_val, sex_missing],
        age_onehot, [age_missing],
        [height_z, height_missing],
        [weight_z, weight_missing],
    ]).astype(np.float32)


class MurmurDataset(Dataset):
    def __init__(self, index_csv, windows_dir, split: str,
                 n_mels: int = 64, n_fft: int = 512, hop_length: int = 160,
                 feature_type: str = "mfcc", n_mfcc: int = 20,
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0),
                 freq_mask_param: int = 0, time_mask_param: int = 0,
                 n_freq_masks: int = 0, n_time_masks: int = 0,
                 include_metadata: bool = False, meta_stats: dict = None,
                 include_location: bool = False):
        if feature_type not in ("logmel", "mfcc"):
            raise ValueError(f"feature_type debe ser 'logmel' o 'mfcc', recibido: {feature_type!r}")
        if include_metadata and meta_stats is None:
            raise ValueError("include_metadata=True requiere meta_stats (ver compute_meta_stats)")
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
        self.include_metadata = include_metadata
        self.meta_stats = meta_stats
        self.include_location = include_location

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
        label = 1.0 if row["pathology_label"] == "heart_murmur" else 0.0
        y_t = torch.tensor(label, dtype=torch.float32)

        if self.include_metadata or self.include_location:
            parts = []
            if self.include_metadata:
                parts.append(encode_metadata(row, self.meta_stats))
            if self.include_location:
                parts.append(encode_location(row))
            extra_vec = torch.tensor(np.concatenate(parts), dtype=torch.float32)
            return x, extra_vec, y_t
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


def collate_pad_multimodal(batch):
    """Igual que collate_pad, pero para (x_audio, x_meta, y) — ver
    MurmurDataset(include_metadata=True)."""
    xs, metas, ys = zip(*batch)
    max_t = max(x.shape[-1] for x in xs)
    padded = [
        torch.nn.functional.pad(x, (0, max_t - x.shape[-1])) if x.shape[-1] < max_t else x
        for x in xs
    ]
    return torch.stack(padded), torch.stack(metas), torch.stack(ys)
