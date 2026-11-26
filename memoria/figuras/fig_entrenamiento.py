"""
fig_entrenamiento.py
====================
Figura 3.4 de la memoria: diagrama de flujo del entrenamiento y la selección
del modelo en cada partición (época, evaluación en validación con el criterio
S, guardado del mejor modelo, corte temprano y predicción sobre prueba).
Mismo estilo que fig_arquitectura.py.

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_entrenamiento.py

Salida: memoria/desarrollo/Figures/entrenamiento.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = REPO_ROOT / "memoria" / "desarrollo" / "Figures" / "entrenamiento.pdf"

ANCHO = 6.3                     # pulgadas
X_MAX = 7.4                     # ancho del dibujo (unidades de datos)
BW, BH = 1.25, 0.62             # tamaño de cada bloque
DW, DH = 1.35, 0.95             # tamaño del rombo de decisión
C_DATO = "#f2f2f2"              # datos de entrada y salida
C_PROC = "#dbe8f4"              # pasos del procedimiento
C_DEC = "#f6dfc8"               # decisiones
BORDE = "#4d4d4d"


def bloque(ax, x, y, texto, color):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH,
                                boxstyle="round,pad=0.02,rounding_size=0.05", fc=color, ec=BORDE, lw=0.7))
    ax.text(x, y, texto, ha="center", va="center", fontsize=7, linespacing=1.15)


def rombo(ax, x, y, texto):
    ax.add_patch(Polygon([(x - DW / 2, y), (x, y + DH / 2), (x + DW / 2, y), (x, y - DH / 2)],
                         closed=True, fc=C_DEC, ec=BORDE, lw=0.7))
    ax.text(x, y, texto, ha="center", va="center", fontsize=6.5, linespacing=1.1)


def flecha(ax, puntos, rotulo=None, pos=None):
    """Flecha que recorre los puntos dados (segmentos rectos)."""
    for p0, p1 in zip(puntos[:-2], puntos[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=BORDE, lw=0.7)
    ax.add_patch(FancyArrowPatch(puntos[-2], puntos[-1], arrowstyle="-|>", mutation_scale=7, lw=0.7,
                                 color=BORDE, shrinkA=0, shrinkB=0))
    if rotulo:
        ax.text(*pos, rotulo, fontsize=7, ha="center", va="center")


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    })
    y0, y1 = -0.15, 2.75
    fig, ax = plt.subplots(figsize=(ANCHO, (y1 - y0) * ANCHO / X_MAX))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(y0, y1)
    ax.axis("off")

    ya, yb = 1.75, 0.3                         # filas
    xs = [0.68, 2.15, 3.62, 5.09, 6.66]
    bloque(ax, xs[0], ya, "Datos de la\npartición i", C_DATO)
    bloque(ax, xs[1], ya, "Época de\nentrenamiento", C_PROC)
    bloque(ax, xs[2], ya, "Evaluación en\nvalidación:\ncálculo de S", C_PROC)
    bloque(ax, xs[3], ya, "Se guarda el\nmodelo si S\nmejoró", C_PROC)
    rombo(ax, xs[4], ya, "¿8 épocas sin\nmejora o 30\népocas?")
    for a, b in zip(xs[:3], xs[1:4]):
        flecha(ax, [(a + BW / 2, ya), (b - BW / 2, ya)])
    flecha(ax, [(xs[3] + BW / 2, ya), (xs[4] - DW / 2, ya)])

    # No: vuelve a la época siguiente
    techo = 2.6
    flecha(ax, [(xs[4], ya + DH / 2), (xs[4], techo), (xs[1], techo), (xs[1], ya + BH / 2)],
           "No", (xs[4] + 0.15, ya + DH / 2 + 0.15))
    # Sí: predicción sobre prueba y reunión de las cinco particiones
    bloque(ax, xs[4], yb, "Predicción sobre\nprueba con el\nmejor modelo", C_PROC)
    bloque(ax, xs[3], yb, "Predicciones de\nlas cinco\nparticiones", C_DATO)
    flecha(ax, [(xs[4], ya - DH / 2), (xs[4], yb + BH / 2)], "Sí", (xs[4] + 0.15, (ya - DH / 2 + yb + BH / 2) / 2))
    flecha(ax, [(xs[4] - BW / 2, yb), (xs[3] + BW / 2, yb)])

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Figura guardada en {OUT_PDF}")


if __name__ == "__main__":
    main()
