"""Fase 4 - Concept drift.

Simula un cambio en la relacion X -> y invirtiendo etiquetas en un subconjunto,
mide la degradacion de F1 sobre lotes sucesivos y define el criterio de
reentrenamiento.

Criterio de reentrenamiento (justificado en la seccion 6.2 del informe):
- Se dispara alarma cuando F1 cae >= 15 % respecto al F1 de referencia
  (F1 del test original) durante 3 lotes consecutivos.
- 15 % porque es una caida visible pero no catastrofica; por debajo de 10 %
  el ruido de muestreo puede confundir. 3 lotes consecutivos evita falsas
  alarmas por un lote anomalo.

Nota sobre el tamano de lote:
- Con lotes pequenos (500 filas) y clase positiva ~12 %, el F1 oscila mucho
  por azar (a veces solo hay ~30 positivos). Por eso se usan lotes de 2000
  filas: reducen la varianza y hacen que los lotes sin flip queden cerca del
  F1 de referencia.

Label delay:
- En produccion la etiqueta real llega con retraso (semanas o nunca).
- Mientras tanto se monitorea:
    * Data drift (6.1) sobre las features -> senal temprana sin etiquetas.
    * Distribucion de probabilidades predichas (PSI del score) -> si cambia,
      el modelo esta viendo entradas distintas.
    * Tasa de positivos predicha vs. esperada -> si se desvia, revisar.
- La metrica real (F1) se calcula solo cuando llegan etiquetas; el criterio
  de 3 lotes aplica a esa senal retrasada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.utils import RANDOM_STATE

F1_DROP_THRESHOLD = 0.15      # 15 % de caida
CONSECUTIVE_BATCHES = 3       # lotes consecutivos con caida para alertar


@dataclass
class BatchResult:
    batch_idx: int
    n_rows: int
    label_flip_fraction: float
    f1: float
    f1_reference: float
    drop_pct: float
    alert: bool


def _flip_labels(y: pd.Series, fraction: float, rng: np.random.Generator) -> pd.Series:
    """Invierte la etiqueta (0<->1) en una fraccion aleatoria de filas."""
    y = y.copy()
    n = len(y)
    n_flip = int(round(n * fraction))
    if n_flip == 0:
        return y
    idx = rng.choice(n, size=n_flip, replace=False)
    y.iloc[idx] = 1 - y.iloc[idx]
    return y


def simulate_concept_drift(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_batches: int = 8,
    batch_size: int = 2000,
    flip_schedule: List[float] | None = None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Genera lotes sucesivos con una fraccion creciente de etiquetas invertidas.

    flip_schedule: lista con la fraccion de etiquetas invertidas por lote.
    Por defecto: [0, 0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
    """
    if flip_schedule is None:
        flip_schedule = [0.0, 0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60]
    flip_schedule = flip_schedule[:n_batches]

    rng = np.random.default_rng(random_state)

    # F1 de referencia: modelo sobre el test original completo
    y_pred_ref = model.predict(X_test)
    f1_ref = float(f1_score(y_test, y_pred_ref, zero_division=0))

    # Muestreamos filas del test para cada lote (con reemplazo si hace falta)
    rows = []
    for i, frac in enumerate(flip_schedule):
        idx = rng.choice(len(X_test), size=min(batch_size, len(X_test)), replace=False)
        X_batch = X_test.iloc[idx].reset_index(drop=True)
        y_batch = y_test.iloc[idx].reset_index(drop=True)
        y_batch_drifted = _flip_labels(y_batch, frac, rng)

        y_pred = model.predict(X_batch)
        f1 = float(f1_score(y_batch_drifted, y_pred, zero_division=0))
        drop = (f1_ref - f1) / f1_ref if f1_ref > 0 else 0.0
        rows.append({
            "batch_idx": i + 1,
            "n_rows": len(X_batch),
            "label_flip_fraction": frac,
            "f1": f1,
            "f1_reference": f1_ref,
            "drop_pct": drop,
            "alert": bool(drop >= F1_DROP_THRESHOLD),
        })
    return pd.DataFrame(rows)


def check_retraining_criterion(results: pd.DataFrame) -> dict:
    """Aplica el criterio: caida >= 15 % durante 3 lotes consecutivos."""
    consec = 0
    triggered_at = None
    for _, row in results.iterrows():
        if row["alert"]:
            consec += 1
            if consec >= CONSECUTIVE_BATCHES and triggered_at is None:
                triggered_at = int(row["batch_idx"])
        else:
            consec = 0
    return {
        "threshold_drop_pct": F1_DROP_THRESHOLD,
        "consecutive_batches_required": CONSECUTIVE_BATCHES,
        "retraining_triggered": triggered_at is not None,
        "triggered_at_batch": triggered_at,
    }


def print_concept_report(results: pd.DataFrame, criterion: dict) -> None:
    print("=" * 90)
    print(" CONCEPT DRIFT  (inversion de etiquetas por lote)")
    print("=" * 90)
    print(f"{'lote':>5}{'n':>7}{'flip_frac':>12}{'f1':>10}{'f1_ref':>10}{'drop_%':>10}{'alerta':>9}")
    print("-" * 90)
    for _, r in results.iterrows():
        print(f"{int(r.batch_idx):>5}{int(r.n_rows):>7}{r.label_flip_fraction:>12.2f}"
              f"{r.f1:>10.4f}{r.f1_reference:>10.4f}{r.drop_pct*100:>9.2f}%"
              f"{'SI' if r.alert else 'no':>9}")
    print("-" * 90)
    if criterion["retraining_triggered"]:
        print(f"REENTRENAMIENTO DISPARADO en el lote {criterion['triggered_at_batch']} "
              f"(caida >= {criterion['threshold_drop_pct']*100:.0f}% "
              f"durante {criterion['consecutive_batches_required']} lotes consecutivos)")
    else:
        print("Reentrenamiento NO disparado: no se cumple el criterio.")
    print("=" * 90)