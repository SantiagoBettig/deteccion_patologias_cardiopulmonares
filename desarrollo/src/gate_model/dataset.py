"""
gate_model/dataset.py
=======================
Dataset de PyTorch para el modelo gate. Carga cada ventana de audio ya
preprocesada (generada por preprocessing/make_gate_windows.py) y calcula su
feature de audio al vuelo — espectrograma log-Mel (default) o MFCC
(feature_type="mfcc", ver avances/6_gate_v5_mfcc.md).
"""

import numpy as np
import librosa
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset


class GateDataset(Dataset):
    """
    label_column / positive_value son parametrizables para poder reutilizar
    este mismo dataset en el diagnóstico de sesgo por source_db (ver
    gate_model/diagnostics_source_db.py): en vez de sound_type=="cardiac",
    se puede pedir source_db=="CIR", por ejemplo.

    Mitigaciones del sesgo detectado en el gate v1 (ver
    avances/4_gate_model_baseline_y_sesgo.md):
      - El espectrograma log-Mel siempre se normaliza por instancia
        (z-score), para reducir diferencias de nivel/offset atribuibles al
        dataset de origen más que al contenido acústico.
      - augment=True (solo para train) aplica ganancia aleatoria + ruido
        aditivo sobre la forma de onda antes de calcular el espectrograma,
        y SpecAugment (enmascarado de bandas de frecuencia y de tiempo)
        sobre el espectrograma ya normalizado — evita que el modelo se
        apoye en artefactos fijos y reproducibles de cada dataset de
        origen, ya sea en la forma de onda o en bandas espectrales fijas.

    feature_type controla qué se calcula a partir de la forma de onda:
      - "logmel" (default): espectrograma log-Mel, igual que en v1-v4.
      - "mfcc": coeficientes MFCC (n_mfcc filas), sobre el mismo banco de
        filtros Mel (n_mels/n_fft/hop_length) — hipótesis de
        avances/4_gate_model_baseline_y_sesgo.md sección 9: al comprimir la
        envolvente espectral en pocos coeficientes, podría ser más robusto
        a la inversión de brillo espectral encontrada en HLS-CMDS. La
        normalización z-score y el augmentation se aplican igual sobre el
        resultado, sea log-Mel o MFCC.
    """

    def __init__(self, index_csv, windows_dir, split: str,
                 label_column: str = "sound_type", positive_value: str = "cardiac",
                 n_mels: int = 64, n_fft: int = 512, hop_length: int = 160,
                 feature_type: str = "logmel", n_mfcc: int = 20,
                 augment: bool = False, noise_std: float = 0.005,
                 gain_db_range: tuple = (-3.0, 3.0),
                 freq_mask_param: int = 0, time_mask_param: int = 0,
                 n_freq_masks: int = 0, n_time_masks: int = 0):
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
        label = 1.0 if row[self.label_column] == self.positive_value else 0.0
        y_t = torch.tensor(label, dtype=torch.float32)
        return x, y_t

    def _augment_waveform(self, y: np.ndarray) -> np.ndarray:
        gain_db = np.random.uniform(*self.gain_db_range)
        y = y * (10.0 ** (gain_db / 20.0))
        noise = np.random.normal(0.0, self.noise_std, size=y.shape).astype(np.float32)
        return np.clip(y + noise, -1.0, 1.0)

    @staticmethod
    def _normalize(log_mel: np.ndarray) -> np.ndarray:
        """Z-score por instancia: quita nivel/offset propios del dataset de origen."""
        mean, std = log_mel.mean(), log_mel.std()
        if std < 1e-6:
            return log_mel - mean
        return (log_mel - mean) / std

    def _spec_augment(self, log_mel: np.ndarray) -> np.ndarray:
        """
        SpecAugment (Park et al., 2019): enmascara bandas de frecuencia y
        de tiempo con el valor medio (0, tras la normalización z-score).
        Evita que la red dependa de bandas espectrales o instantes fijos
        que podrían ser artefactos del dataset de origen en vez de
        contenido acústico real.
        """
        log_mel = log_mel.copy()
        n_mels, n_frames = log_mel.shape

        for _ in range(self.n_freq_masks):
            width = np.random.randint(0, self.freq_mask_param + 1)
            if width == 0 or width >= n_mels:
                continue
            f0 = np.random.randint(0, n_mels - width)
            log_mel[f0:f0 + width, :] = 0.0

        for _ in range(self.n_time_masks):
            width = np.random.randint(0, self.time_mask_param + 1)
            if width == 0 or width >= n_frames:
                continue
            t0 = np.random.randint(0, n_frames - width)
            log_mel[:, t0:t0 + width] = 0.0

        return log_mel
