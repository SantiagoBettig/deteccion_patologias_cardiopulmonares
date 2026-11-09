"""
fig_arquitectura.py
===================
Figura 3.1 de la memoria: diagrama de bloques de las dos configuraciones del
sistema. Arriba, la integrada (un modelo de 6 clases sobre ventanas de 4 s);
abajo, la separada (gate que decide el tipo de sonido de la grabación y la
deriva a la rama cardíaca o a la pulmonar).

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_arquitectura.py

Salida: memoria/desarrollo/Figures/arquitectura.pdf
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = REPO_ROOT / "memoria" / "desarrollo" / "Figures" / "arquitectura.pdf"

ANCHO, ALTO = 6.3, 3.3          # pulgadas
BW, BH = 0.82, 0.58             # tamaño de cada bloque (unidades de datos)
C_DATO = "#f2f2f2"              # entrada y salida
C_PROC = "#dbe8f4"              # procesamiento de la señal
C_MODELO = "#f6dfc8"            # modelos
BORDE = "#4d4d4d"


def bloque(ax, x, y, texto, color, w=BW, h=BH):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                fc=color, ec=BORDE, lw=0.7))
    ax.text(x, y, texto, ha="center", va="center", fontsize=7, linespacing=1.15)


def flecha(ax, p0, p1):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=7, lw=0.7, color=BORDE,
                                 shrinkA=0, shrinkB=0))


def fila(ax, y, xs, textos, colores):
    for x, t, c in zip(xs, textos, colores):
        bloque(ax, x, y, t, c)
    for a, b in zip(xs[:-1], xs[1:]):
        flecha(ax, (a + BW / 2, y), (b - BW / 2, y))


def main() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
    })
    fig, ax = plt.subplots(figsize=(ANCHO, ALTO))
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0.2, 3.9)
    ax.axis("off")
    xs = [0.45 + 1.05 * i for i in range(7)]

    # (a) Configuración integrada
    ya = 3.2
    ax.text(0.05, 3.75, "(a) Configuración integrada", fontsize=8, ha="left", va="center")
    fila(ax, ya, xs,
         ["Grabación", "Preproce-\nsamiento", "Ventanas\nde 4 s", "Espectro-\ngrama\nlog-Mel",
          "Modelo de\n6 clases", "Agregación\npor\ngrabación", "Tipo de\nsonido y\nalteración"],
         [C_DATO, C_PROC, C_PROC, C_PROC, C_MODELO, C_PROC, C_DATO])

    # (b) Configuración separada
    ax.text(0.05, 2.55, "(b) Configuración separada", fontsize=8, ha="left", va="center")
    yc, yp = 1.85, 0.6               # ramas cardíaca y pulmonar
    ym = (yc + yp) / 2
    fila(ax, ym, xs[:3], ["Grabación", "Preproce-\nsamiento", "Gate:\ncardíaco o\npulmonar"],
         [C_DATO, C_PROC, C_MODELO])
    for y in (yc, yp):
        flecha(ax, (xs[2] + BW / 2, ym), (xs[3] - BW / 2, y))
    fila(ax, yc, xs[3:6], ["Ventanas\ny log-Mel", "Rama\ncardíaca\n(2 clases)", "Agregación\npor\ngrabación"],
         [C_PROC, C_MODELO, C_PROC])
    fila(ax, yp, xs[3:6], ["Ventanas\ny log-Mel", "Rama\npulmonar\n(4 clases)", "Agregación\npor\ngrabación"],
         [C_PROC, C_MODELO, C_PROC])
    bloque(ax, xs[6], ym, "Tipo de\nsonido y\nalteración", C_DATO)
    for y in (yc, yp):
        flecha(ax, (xs[5] + BW / 2, y), (xs[6] - BW / 2, ym))

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Figura guardada en {OUT_PDF}")


if __name__ == "__main__":
    main()
