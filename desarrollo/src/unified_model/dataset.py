"""
unified_model/dataset.py
===========================
Dataset del modelo único sin gate: ventanas fijas del índice
(preprocessing/make_unified_windows.py) recortadas de la cache de audio
preprocesado (preprocessing/make_unified_audio_cache.py, abierta con memory
map). Features iguales a gate/murmullo: MFCC (20 coef., n_mels 64, n_fft
512, hop 160) con z-score por ventana; en train, la misma augmentation que
murmullo v1 (ganancia ±3 dB + ruido gaussiano, std 0.005) sobre la onda.

El MFCC con librosa tarda ~12 ms por ventana (~400 s por época en CPU), así
que el dataset devuelve la onda y el MFCC se calcula por batch en PyTorch
(SpectralFrontend, que también da el log-Mel), replicando librosa.feature.mfcc (STFT centrada con relleno
de ceros, ventana Hann, mel de librosa, power_to_db con top_db 80, DCT-II
ortonormal). El índice no tiene ventanas más cortas que 4 s (no hay
grabaciones tan cortas), así que todas las ondas de un batch miden lo mismo.

Probabilidades derivadas de las 6 clases (avances/21 §4), en
derived_probs: P(murmullo) = card_murmur; P(crackle) = crackle + both;
P(wheeze) = wheeze + both; P(cardíaco) = card_normal + card_murmur.
"""

import librosa
import numpy as np
import pandas as pd
import scipy.fft
import torch
from torch.utils.data import Dataset

from common.paths import UNIFIED_AUDIO_CACHE_NPY, UNIFIED_AUDIO_OFFSETS_CSV

SR = 4000
MFCC_KW = {"n_mfcc": 20, "n_mels": 64, "n_fft": 512, "hop_length": 160}


def open_audio_cache(cache_npy=UNIFIED_AUDIO_CACHE_NPY, offsets_csv=UNIFIED_AUDIO_OFFSETS_CSV
                     ) -> tuple[np.ndarray, dict[str, int]]:
    audio = np.load(cache_npy, mmap_mode="r")
    offsets = pd.read_csv(offsets_csv).set_index("file_id")["offset"].to_dict()
    return audio, offsets


FEATURES = ("mfcc", "logmel")


class SpectralFrontend(torch.nn.Module):
    """(B, n) onda -> (B, 1, bandas, frames) con z-score por ventana.

    - "mfcc": numéricamente igual a librosa.feature.mfcc(**MFCC_KW) (20 coef.).
    - "logmel": el log-Mel de 64 bandas del que sale ese MFCC (power_to_db
      con top_db 80), sin la DCT. Es lo que usó la rama pulmonar desde v1
      (avances/13: un wheeze es una línea tonal angosta y 20 MFCC resumen
      la envolvente espectral); el piloto 1 con MFCC no aprendió wheeze."""

    def __init__(self, feature: str = "mfcc", sr: int = SR, n_mfcc: int = 20, n_mels: int = 64,
                 n_fft: int = 512, hop_length: int = 160):
        super().__init__()
        if feature not in FEATURES:
            raise ValueError(f"feature debe ser uno de {FEATURES}, recibido: {feature!r}")
        self.feature = feature
        self.n_fft, self.hop = n_fft, hop_length
        self.register_buffer("window", torch.hann_window(n_fft, periodic=True))
        self.register_buffer("mel", torch.tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels),
                                                 dtype=torch.float32))
        dct = scipy.fft.dct(np.eye(n_mels), type=2, norm="ortho", axis=0)[:n_mfcc]   # (n_mfcc, n_mels)
        self.register_buffer("dct", torch.tensor(dct, dtype=torch.float32))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(y, self.n_fft, self.hop, window=self.window, center=True,
                          pad_mode="constant", return_complex=True).abs() ** 2       # (B, F, T)
        db = 10.0 * torch.log10(torch.clamp(self.mel @ spec, min=1e-10))
        db = torch.maximum(db, db.amax(dim=(1, 2), keepdim=True) - 80.0)
        feat = self.dct @ db if self.feature == "mfcc" else db                        # (B, bandas, T)
        mean = feat.mean(dim=(1, 2), keepdim=True)
        std = feat.std(dim=(1, 2), keepdim=True, unbiased=False)
        feat = torch.where(std < 1e-6, feat - mean, (feat - mean) / std.clamp_min(1e-6))
        return feat.unsqueeze(1)


class UnifiedWindowDataset(Dataset):
    """windows: filas del índice (ya filtradas por split). Devuelve la onda
    de la ventana (float32) y la clase; las features las calcula
    SpectralFrontend sobre el batch."""

    def __init__(self, windows: pd.DataFrame, audio: np.ndarray, offsets: dict[str, int],
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0)):
        self.df = windows.reset_index(drop=True)
        self.audio = audio
        self.starts = (self.df["file_id"].map(offsets) + self.df["start_sample"]).to_numpy()
        self.lengths = self.df["n_samples"].to_numpy()
        self.labels = self.df["label"].to_numpy()
        self.augment = augment
        self.noise_std = noise_std
        self.gain_db_range = gain_db_range

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        s = self.starts[idx]
        y = self.audio[s:s + self.lengths[idx]].astype(np.float32) / 32768.0
        if self.augment:
            y = y * (10.0 ** (np.random.uniform(*self.gain_db_range) / 20.0))
            y = np.clip(y + np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32), -1.0, 1.0)
        return torch.from_numpy(y), int(self.labels[idx])


def collate(batch):
    ys, labels = zip(*batch)
    return torch.stack(ys), torch.tensor(labels, dtype=torch.long)


def derived_probs(p: np.ndarray) -> dict[str, np.ndarray]:
    """p: (N, 6) softmax -> probabilidades por tarea."""
    return {
        "p_card_normal": p[:, 0], "p_card_murmur": p[:, 1], "p_pulm_normal": p[:, 2],
        "p_pulm_crackle": p[:, 3], "p_pulm_wheeze": p[:, 4], "p_pulm_both": p[:, 5],
        "p_cardiac": p[:, 0] + p[:, 1],
        "p_murmur": p[:, 1],
        "p_crackle": p[:, 3] + p[:, 5],
        "p_wheeze": p[:, 4] + p[:, 5],
    }
