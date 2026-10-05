"""
gate_model/plot_architecture.py
=================================
Genera un diagrama esquemático de GateCNN con las dimensiones reales del
tensor en cada etapa (calculadas con un forward pass de prueba, usando el
mismo shape de espectrograma que produce GateDataset). Pensado como recurso
visual para el informe final.

Guarda: reports/gate/architecture.png

Uso:
    python gate_model/plot_architecture.py
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import GATE_REPORTS_DIR
from gate_model.model import GateCNN

TARGET_SR = 4000
WINDOW_SEC = 5.0
N_MELS, N_FFT, HOP_LENGTH = 64, 512, 160


def _dummy_input() -> torch.Tensor:
    """Espectrograma de una ventana de silencio, solo para conocer el shape real."""
    n_samples = int(TARGET_SR * WINDOW_SEC)
    y = np.zeros(n_samples, dtype=np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=TARGET_SR, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    return torch.zeros(1, 1, *mel.shape)


def _collect_stages(model: GateCNN, x: torch.Tensor) -> list[tuple[str, tuple]]:
    stages = [("Input\nlog-Mel spectrogram", tuple(x.shape[1:]))]
    out = x
    for i, block in enumerate(model.features, start=1):
        out = block(out)
        stages.append((f"ConvBlock {i}\nConv3x3+BN+ReLU+MaxPool2", tuple(out.shape[1:])))
    pooled = model.pool(out).flatten(1)
    stages.append(("GlobalAvgPool", tuple(pooled.shape[1:])))
    logits = model.classifier(pooled)
    stages.append(("Linear -> 1 logit", tuple(logits.shape[1:])))
    return stages


def plot_architecture() -> None:
    model = GateCNN()
    model.eval()

    with torch.no_grad():
        stages = _collect_stages(model, _dummy_input())

    n_params = sum(p.numel() for p in model.parameters())

    box_w, box_h, gap = 2.2, 1.3, 0.9
    fig, ax = plt.subplots(figsize=(len(stages) * (box_w + gap) + 1, 3))

    for i, (name, shape) in enumerate(stages):
        x0 = i * (box_w + gap)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x0, 0), box_w, box_h,
            boxstyle="round,pad=0.05", linewidth=1.5,
            edgecolor="black", facecolor="#cfe8ff",
        ))
        shape_str = " × ".join(str(d) for d in shape)
        ax.text(x0 + box_w / 2, box_h * 0.62, name, ha="center", va="center", fontsize=8.5, weight="bold")
        ax.text(x0 + box_w / 2, box_h * 0.22, shape_str, ha="center", va="center", fontsize=8)

        if i > 0:
            prev_x_end = (i - 1) * (box_w + gap) + box_w
            ax.annotate("", xy=(x0, box_h / 2), xytext=(prev_x_end, box_h / 2),
                        arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.set_xlim(-0.4, len(stages) * (box_w + gap))
    ax.set_ylim(-0.3, box_h + 0.3)
    ax.axis("off")
    ax.set_title(f"GateCNN — {n_params:,} parámetros".replace(",", "."), fontsize=12)

    fig.tight_layout()
    GATE_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GATE_REPORTS_DIR / "architecture.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"✓ Diagrama de arquitectura guardado en: {out_path}")


if __name__ == "__main__":
    plot_architecture()
