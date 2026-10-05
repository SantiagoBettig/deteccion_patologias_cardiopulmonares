"""
pulmonary_adventitious_model/train.py
========================================
Entrenamiento de UN fold del modelo de sonidos adventicios. Lo orquesta
run_kfold.py (que corre los K folds y arma los reportes); este módulo solo
sabe entrenar sobre un train/val dados y devolver el mejor checkpoint.

Receta del baseline (punto de partida, adaptada de la v1 de murmullo):
- BCEWithLogitsLoss con pos_weight natural por etiqueta (n_neg/n_pos de
  train, calculado en cada fold). Sin el multiplicador x1.4 de murmullo:
  allá se buscaba recall a propósito (criterio clínico, F2); acá la métrica
  principal (score ICBHI) ya pesa igual Se y Sp.
- Dropout 0.3, Adam lr=1e-3, augmentation de ganancia + ruido en train.
- Criterio de mejor checkpoint (`select_metric`):
    "icbhi" (v1, v2): mayor score ICBHI en validación, con umbrales
        calibrados sobre validación en cada época. Alinea selección y
        métrica final, pero con 16 pacientes de validación es muy ruidoso:
        en fold 1 eligió la época 1 (ver avances/13 §5).
    "auc" (v3): mayor AUC medio (crackle, wheeze) en validación — no
        depende de ningún umbral, así que mide solo separabilidad y es más
        estable época a época. Los umbrales se calibran igual, una sola vez,
        sobre el checkpoint ya elegido (run_kfold.py).
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from pulmonary_adventitious_model.dataset import CycleDataset, LABEL_COLUMNS
from pulmonary_adventitious_model.model import AdventitiousCNN
from pulmonary_adventitious_model.metrics import calibrate_thresholds, full_metrics

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DROPOUT = 0.3


def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            all_probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
            all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def train_fold(train_df, val_df, cycles_dir, checkpoint_path, device,
               epochs: int = EPOCHS, wav_cache: dict = None, dataset_kwargs: dict = None,
               model_kwargs: dict = None, select_metric: str = "icbhi",
               log_prefix: str = "") -> list[dict]:
    if select_metric not in ("icbhi", "auc"):
        raise ValueError(f"select_metric debe ser 'icbhi' o 'auc', recibido: {select_metric!r}")
    dataset_kwargs = dataset_kwargs or {}
    model_kwargs = model_kwargs or {}
    train_ds = CycleDataset(train_df, cycles_dir, augment=True, wav_cache=wav_cache, **dataset_kwargs)
    val_ds = CycleDataset(val_df, cycles_dir, augment=False, wav_cache=wav_cache, **dataset_kwargs)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    n_pos = train_df[LABEL_COLUMNS].sum().to_numpy(dtype=np.float32)
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor(n_neg / n_pos, dtype=torch.float32, device=device)
    print(f"{log_prefix}Ciclos — train: {len(train_ds)}, val: {len(val_ds)} | "
          f"pos_weight crackle={pos_weight[0]:.2f} wheeze={pos_weight[1]:.2f}")

    model = AdventitiousCNN(dropout=DROPOUT, **model_kwargs).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_score = -1.0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(train_ds)

        val_labels, val_probs = predict_probs(model, val_loader, device)
        val_loss = float(criterion(torch.logit(torch.tensor(val_probs).clamp(1e-6, 1 - 1e-6)).to(device),
                                   torch.tensor(val_labels).to(device)).item())
        thresholds, _ = calibrate_thresholds(val_labels, val_probs)
        m = full_metrics(val_labels, val_probs, thresholds)

        print(f"{log_prefix}[{epoch:02d}/{epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
              f"val score={m['icbhi_score']:.4f} (Se={m['se']:.3f} Sp={m['sp']:.3f}) "
              f"AUC crackle={m['crackle_auc']:.3f} wheeze={m['wheeze_auc']:.3f}")

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        **{f"val_{k}": v for k, v in m.items()}})

        m["mean_auc"] = (m["crackle_auc"] + m["wheeze_auc"]) / 2
        history[-1]["val_mean_auc"] = m["mean_auc"]
        current = m["icbhi_score"] if select_metric == "icbhi" else m["mean_auc"]
        if current > best_score:
            best_score = current
            torch.save(model.state_dict(), checkpoint_path)
            print(f"{log_prefix}  ✓ Nuevo mejor checkpoint (val {select_metric}={best_score:.4f})")

    return history
