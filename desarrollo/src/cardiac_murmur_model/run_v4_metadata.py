"""
cardiac_murmur_model/run_v4_metadata.py
==========================================
Murmullo v4: agrega metadata del paciente (sexo, edad, altura, peso) al v1
(MFCC, sigue siendo la mejor versión de audio — ver
avances/10_cardiaco_split_dos_tareas.md, secciones 6-7).

Antes de confiar en la mejora (si la hay), corre también el diagnóstico
"solo metadata" (diagnostics_metadata_only.py) — mismo tipo de chequeo que
gate_model/diagnostics_source_db.py: si un modelo que NO ve audio ya
predice bien el murmullo a partir de la metadata sola, cualquier mejora
del modelo combinado podría ser un atajo demográfico y no una interacción
real con el audio.

Guarda checkpoint y reportes en carpetas separadas de v1/v2/v3:
    cardiac_murmur_model/checkpoints_v4_metadata/best.pt
    reports/cardiac_murmur/runs/v4_metadata/

Uso:
    python cardiac_murmur_model/run_v4_metadata.py
"""

from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac_murmur_model.train_multimodal import train
from cardiac_murmur_model.evaluate_multimodal import evaluate
from cardiac_murmur_model.diagnostics_metadata_only import run_diagnostic


def main() -> None:
    print("=" * 55)
    print("  MURMULLO v4 — MFCC + metadata del paciente")
    print("=" * 55)
    train()
    evaluate()

    print("\n" + "=" * 55)
    print("  DIAGNÓSTICO — ¿la metadata sola ya predice el murmullo?")
    print("=" * 55)
    run_diagnostic()


if __name__ == "__main__":
    main()
