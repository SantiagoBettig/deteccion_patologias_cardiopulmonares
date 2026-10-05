"""
pulmonary_adventitious_model/panns_cnn6.py
=============================================
Arquitectura CNN6 de PANNs (Kong et al., 2020, "PANNs: Large-Scale
Pretrained Audio Neural Networks for Audio Pattern Recognition"),
preentrenada en AudioSet (~2M clips, 527 clases). Pesos oficiales:
Zenodo doi:10.5281/zenodo.3987831 (CC-BY-4.0), en common/paths.py:
PANNS_CNN6_CKPT.

Se reimplementa acá (en vez de depender del paquete `panns_inference`)
para controlar la entrada: el frontend original (STFT + log-Mel a 32 kHz)
va aparte, en extract_cnn6_embeddings.py, porque se prueban dos
espectrogramas distintos como entrada (ver avances/15). Los nombres de las
capas coinciden con los del checkpoint original, así que los pesos cargan
directo con load_state_dict.

`embed()` devuelve la salida de fc1 (512 valores, después de ReLU) — el
"embedding" de PANNs, la capa anterior a la cabeza de 527 clases de
AudioSet, que se descarta.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.avg_pool2d(F.relu(self.bn1(self.conv1(x))), kernel_size=2)


class Cnn6(nn.Module):
    EMBED_DIM = 512
    N_MELS = 64

    def __init__(self, n_audioset_classes: int = 527):
        super().__init__()
        self.bn0 = nn.BatchNorm2d(self.N_MELS)
        self.conv_block1 = ConvBlock5x5(1, 64)
        self.conv_block2 = ConvBlock5x5(64, 128)
        self.conv_block3 = ConvBlock5x5(128, 256)
        self.conv_block4 = ConvBlock5x5(256, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc_audioset = nn.Linear(512, n_audioset_classes)  # se carga pero no se usa

    def frozen_trunk(self, logmel: torch.Tensor) -> torch.Tensor:
        """bn0 + bloques 1-3: la parte que queda congelada en el fine-tuning
        parcial (ver run_partial_finetune.py). Devuelve (batch, 256,
        frames/8, 8)."""
        x = logmel.transpose(1, 3)
        x = self.bn0(x).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3):
            x = F.dropout(block(x), p=0.2, training=self.training)
        return x

    def embed(self, logmel: torch.Tensor) -> torch.Tensor:
        """logmel: (batch, 1, frames, 64) en dB (10·log10 de la potencia Mel)."""
        x = logmel.transpose(1, 3)          # BN por banda Mel, como en PANNs
        x = self.bn0(x).transpose(1, 3)
        for block in (self.conv_block1, self.conv_block2, self.conv_block3, self.conv_block4):
            x = F.dropout(block(x), p=0.2, training=self.training)
        x = x.mean(dim=3)                   # promedio sobre frecuencia
        x = x.amax(dim=2) + x.mean(dim=2)   # máximo + promedio sobre tiempo
        x = F.dropout(x, p=0.5, training=self.training)
        return F.relu(self.fc1(x))


def load_pretrained_cnn6(checkpoint_path) -> Cnn6:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k: v for k, v in ckpt["model"].items()
             if not k.startswith(("spectrogram_extractor.", "logmel_extractor."))}
    model = Cnn6()
    model.load_state_dict(state)  # strict: falla si algún nombre/forma no coincide
    model.eval()
    return model
