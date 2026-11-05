"""
fig_distribucion_bases.py
=========================
Figura de la sección 2.3 de la memoria: distribución de las etiquetas
originales de cada base de datos, en la unidad en que cada una las anota
(paciente en CirCor, grabación en SPRSound y HLS-CMDS, ciclo respiratorio en
ICBHI). Muestra las bases tal como se publican, antes de cualquier
procesamiento propio.

En SPRSound, las entregas 2024 y 2025 no traen la etiqueta por grabación;
se la deriva de los eventos anotados, que en las entregas que sí la traen
coinciden en el 100 % de los casos (avances/23 §1).

Uso (desde la raíz del repo, con el venv de desarrollo/):
    desarrollo/.venv/Scripts/python.exe memoria/figuras/fig_distribucion_bases.py

Salida: memoria/desarrollo/Figures/distribucion_bases.pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR   = REPO_ROOT / "desarrollo" / "data" / "raw"
SPR_REP   = REPO_ROOT / "desarrollo" / "reports" / "sprsound"
OUT_PDF   = REPO_ROOT / "memoria" / "desarrollo" / "Figures" / "distribucion_bases.pdf"

COLOR_NORMAL   = "#1f4e79"
COLOR_ANORMAL  = "#c0392b"
COLOR_OTRO     = "#9e9e9e"

CRACKLE = {"Fine Crackle", "Coarse Crackle", "Wheeze+Crackle"}
CONTINUO = {"Wheeze", "Rhonchi", "Stridor", "Wheeze+Crackle"}


def circor() -> list[tuple[str, int, str]]:
    m = pd.read_csv(RAW_DIR / "CirCor" / "training_data.csv", sep=";")["Murmur"].value_counts()
    return [("Sin soplo", m["Absent"], COLOR_NORMAL), ("Con soplo", m["Present"], COLOR_ANORMAL),
            ("Desconocido", m["Unknown"], COLOR_OTRO)]


def sprsound() -> list[tuple[str, int, str]]:
    rec = pd.read_csv(SPR_REP / "recordings.csv", dtype={"stem": str})
    ev = pd.read_csv(SPR_REP / "events.csv", dtype={"stem": str})
    tipos = ev.groupby("stem")["type"].agg(set)

    def etiqueta(fila) -> str:
        if fila.record_annotation == "Poor Quality":
            return "Mala calidad"
        t = tipos.get(fila.stem, set())
        crk, cont = bool(t & CRACKLE), bool(t & CONTINUO)
        if not t:
            return "Mala calidad"
        return {(False, False): "Normal", (True, False): "Discontinuos",
                (False, True): "Continuos", (True, True): "Ambos"}[(crk, cont)]

    n = rec.apply(etiqueta, axis=1).value_counts()
    return [("Normal", n.get("Normal", 0), COLOR_NORMAL),
            ("Discontinuos", n.get("Discontinuos", 0), COLOR_ANORMAL),
            ("Continuos", n.get("Continuos", 0), COLOR_ANORMAL),
            ("Ambos", n.get("Ambos", 0), COLOR_ANORMAL),
            ("Mala calidad", n.get("Mala calidad", 0), COLOR_OTRO)]


def icbhi() -> list[tuple[str, int, str]]:
    n = {"Normal": 0, "Crepitantes": 0, "Sibilancias": 0, "Ambos": 0}
    for t in (RAW_DIR / "ICBHI" / "audio_and_txt_files").glob("*.txt"):
        for linea in t.read_text().splitlines():
            p = linea.split()
            if len(p) == 4:
                n[{("0", "0"): "Normal", ("1", "0"): "Crepitantes",
                   ("0", "1"): "Sibilancias", ("1", "1"): "Ambos"}[(p[2], p[3])]] += 1
    return [(k, v, COLOR_NORMAL if k == "Normal" else COLOR_ANORMAL) for k, v in n.items()]


def hls() -> list[tuple[str, int, str]]:
    base = RAW_DIR / "HLS-CMDS" / "Dataset"
    hs = pd.read_csv(base / "HS" / "HS.csv")["Heart Sound Type"]
    ls = pd.read_csv(base / "LS" / "LS.csv")["Lung Sound Type"]
    soplo = hs.str.contains("Murmur")
    crep = ls.str.contains("Crackles")
    return [("Cardíaco normal", (hs == "Normal").sum(), COLOR_NORMAL),
            ("Soplo", soplo.sum(), COLOR_ANORMAL),
            ("Otra cardíaca", (~soplo & (hs != "Normal")).sum(), COLOR_OTRO),
            ("Pulmonar normal", (ls == "Normal").sum(), COLOR_NORMAL),
            ("Crepitantes", crep.sum(), COLOR_ANORMAL),
            ("Sibilancias", (ls == "Wheezing").sum(), COLOR_ANORMAL),
            ("Roncus", (ls == "Rhonchi").sum(), COLOR_ANORMAL),
            ("Frote pleural", (ls == "Pleural Rub").sum(), COLOR_OTRO)]


def main(salida=OUT_PDF):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Palatino Linotype", "Book Antiqua", "DejaVu Serif"],
        "font.size": 8,
    })
    paneles = [("CirCor (pacientes)", circor()), ("SPRSound (grabaciones)", sprsound()),
               ("ICBHI (ciclos respiratorios)", icbhi()), ("HLS-CMDS (grabaciones)", hls())]
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.6),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.55})
    for ax, (titulo, barras) in zip(axes.ravel(), paneles):
        nombres, valores, colores = zip(*barras)
        y = range(len(barras))[::-1]
        ax.barh(list(y), valores, color=colores, height=0.65)
        ax.set_yticks(list(y))
        ax.set_yticklabels(nombres)
        ax.set_title(titulo, fontsize=9, pad=4)
        ax.spines[["top", "right"]].set_visible(False)
        tope = max(valores)
        ax.set_xlim(0, tope * 1.22)
        for yi, v in zip(y, valores):
            ax.text(v + tope * 0.02, yi, f"{v:,}".replace(",", "."), va="center", fontsize=7)
        ax.tick_params(axis="x", labelsize=7)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
        print(titulo, dict(zip(nombres, valores)))

    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, bbox_inches="tight")
    print(f"Figura guardada en {salida}")


if __name__ == "__main__":
    if len(sys.argv) > 1:  # vista previa en otra ruta (png) sin pisar la figura
        main(salida=Path(sys.argv[1]))
    else:
        main()
