"""
preprocessing/make_external_recordings.py
============================================
Tablas de grabaciones de las dos bases externas grabadas con el equipo de
CirCor (avances/28 §11). Lee los WAV extraídos en data/raw/<base>/audio/ y
escribe data/<base>/recordings.csv con file_id, audio_path (relativo a
data/raw/<base>/), subject_id y etiquetas. Después:
make_unified_audio_cache.py --data kauh|pascal.

KAUH (Fraiwan et al., 2021; Littmann 3200; 112 sujetos, pulmonar)
- Cada grabación viene tres veces, una por modo de filtro del 3200: B
  (campana, 20-1000 Hz, realza 20-200), D (diafragma, 20-2000 Hz, realza
  100-500) y E (extendido, 20-1000 Hz, realza 50-500). file_id = "<modo>/P<n>".
- Etiquetas del nombre del archivo: "<modo>P<n>_<diagnóstico>,<sonidos>,
  <ubicación>,<edad>,<sexo>". Sonidos: N normal, C o Crep crackle
  (crepitación), W wheeze, I/E fase (inspiratoria/espiratoria), B bronquial.
- Equivalente en ICBHI (lung_icbhi) para todo salvo "Bronchial" sola, que no
  es crackle ni wheeze ni normal (se analiza aparte, como rhonchi en HLS).
  "I C B" (crackle + bronquial) cuenta como crackle.

PASCAL Dataset B (challenge PASCAL 2011; DigiScope, Real Hospital Português,
Recife; cardíaco)
- Carpetas: normal, murmullo, extrasístole, y versiones "noisy" de normal y
  murmullo. murmur = 1 en murmullo y noisymurmur; extrasístole cuenta como
  "sin murmullo", igual que abnormal_heart_unspecified al entrenar.
- subject_id = el número antes del primer "_" (hay pacientes en más de una
  carpeta: las etiquetas son por grabación).
- Grabaciones a 4 kHz, de 0.8 a 28 s (mediana 5 s): las de menos de 4 s se
  rellenan con ceros en la inferencia, como en SPRSound.

Uso:
    python preprocessing/make_external_recordings.py
"""

import re

import pandas as pd
import soundfile as sf
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import KAUH_RAW_DIR, PASCAL_RAW_DIR, external_paths

KAUH_MODES = {"B": "campana", "D": "diafragma", "E": "extendido"}
PASCAL_CLASSES = {"Training B Normal": ("normal", 0, False), "Btraining_noisynormal": ("normal", 0, True),
                  "Btraining_murmur": ("murmullo", 1, False), "Btraining_noisymurmur": ("murmullo", 1, True),
                  "Btraining_extrastole": ("extrasistole", 0, False)}


def kauh() -> pd.DataFrame:
    rows = []
    for f in sorted((KAUH_RAW_DIR / "audio").glob("*.wav")):
        m = re.match(r"([BDE])P(\d+)_(.*)", f.stem)
        mode, pid, rest = m.groups()
        diag, sounds, loc, age, sex = [p.strip() for p in rest.split(",")]
        tokens = set(sounds.replace("Crep", "C").replace("Bronchial", "B").split())
        has_c, has_w = int("C" in tokens), int("W" in tokens)
        only_bronchial = tokens <= {"B", "I", "E"} and "B" in tokens and not (has_c or has_w)
        info = sf.info(f)
        rows.append({"file_id": f"{mode}/P{int(pid)}", "audio_path": f"audio/{f.name}",
                     "subject_id": f"KAU_P{int(pid)}", "modo": KAUH_MODES[mode], "sonidos": sounds,
                     "diagnostico": diag.lower(), "ubicacion": loc, "age_years": float(age), "sex": sex,
                     "has_crackles": has_c, "has_wheezes": has_w, "lung_icbhi": not only_bronchial,
                     "sample_rate": info.samplerate, "duration_sec": info.duration})
    return pd.DataFrame(rows)


def pascal() -> pd.DataFrame:
    rows = []
    for f in sorted((PASCAL_RAW_DIR / "audio").rglob("*.wav")):
        cls, murmur, noisy = PASCAL_CLASSES[f.parent.name]
        info = sf.info(f)
        rows.append({"file_id": f"{f.parent.name}/{f.stem}", "audio_path": f"audio/{f.parent.name}/{f.name}",
                     "subject_id": f"PAS_{f.stem.split('_')[0]}", "clase": cls, "murmur": murmur, "ruidosa": noisy,
                     "sample_rate": info.samplerate, "duration_sec": info.duration})
    return pd.DataFrame(rows)


def main() -> None:
    for name, build in (("kauh", kauh), ("pascal", pascal)):
        df = build()
        out = external_paths(name)[0]
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"✓ {out}: {len(df)} grabaciones, {df.subject_id.nunique()} sujetos")
        if name == "kauh":
            b = df[df.modo == "campana"]
            print(f"  por sujeto: normal {(b.has_crackles + b.has_wheezes == 0).sum() - (~b.lung_icbhi).sum()}, "
                  f"crackle {b.has_crackles.sum()}, wheeze {b.has_wheezes.sum()}, "
                  f"both {(b.has_crackles & b.has_wheezes).sum()}, sin equivalente {(~b.lung_icbhi).sum()}")
        else:
            print(f"  {df.groupby(['clase', 'ruidosa']).size().to_dict()}")


if __name__ == "__main__":
    main()
