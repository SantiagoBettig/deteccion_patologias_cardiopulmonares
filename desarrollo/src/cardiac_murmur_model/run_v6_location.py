"""
cardiac_murmur_model/run_v6_location.py
===========================================
Murmullo v6: MFCC (igual que v1) + ubicación de auscultación como feature
— a diferencia de la metadata demográfica (v4, no ayudó), la ubicación
está mecánicamente ligada a si el murmullo se escucha (ver
avances/10_cardiaco_split_dos_tareas.md, sección 11).

Guarda checkpoint y reportes en carpetas separadas de v1-v5:
    cardiac_murmur_model/checkpoints_v6_location/best.pt
    reports/cardiac_murmur/runs/v6_location/

Uso:
    python cardiac_murmur_model/run_v6_location.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac_murmur_model.train_location import train
from cardiac_murmur_model.evaluate_location import evaluate


def main() -> None:
    print("=" * 55)
    print("  MURMULLO v6 — MFCC + ubicación de auscultación")
    print("=" * 55)
    train()
    evaluate()


if __name__ == "__main__":
    main()
