"""Fase 4 - Ejecuta data drift + concept drift y genera evidencia.

Uso (desde bank-mlops/):
    $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
    python scripts\\run_fase4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.drift.concept_drift import (
    check_retraining_criterion,
    print_concept_report,
    simulate_concept_drift,
)
from src.drift.data_drift import detect_data_drift, print_report
from src.inference.model_loader import load_model
from src.utils import binarize_target, load_baseline, load_dataset, split_train_test

EVIDENCIAS = PROJECT_ROOT / "docs" / "evidencias"


def _json_safe(obj):
    """Convierte numpy scalars/arrays y tipos raros a tipos nativos de Python."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def make_drifted_batch(baseline: pd.DataFrame, n: int = 1000,
                       random_state: int = 7) -> pd.DataFrame:
    """Genera un lote con drift evidente en varias columnas."""
    rng = np.random.default_rng(random_state)
    df = baseline.sample(n=n, replace=True, random_state=random_state).reset_index(drop=True)

    df["age"] = (df["age"] + 20).clip(18, 100).astype(int)
    df["balance"] = (df["balance"] * 5).astype(int)
    df["duration"] = (df["duration"] * 2).astype(int)
    df["campaign"] = (df["campaign"] + 3).astype(int)
    df["job"] = rng.choice(["student", "retired"], size=len(df))
    df["month"] = "dec"
    df["poutcome"] = "success"
    return df


def make_normal_batch(baseline: pd.DataFrame, n: int = 1000,
                      random_state: int = 7) -> pd.DataFrame:
    """Lote del mismo origen que el baseline (debe pasar en verde)."""
    return baseline.sample(n=n, replace=True, random_state=random_state).reset_index(drop=True)


def main() -> None:
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print(" FASE 4 - DATA DRIFT + CONCEPT DRIFT")
    print("=" * 90)

    baseline = load_baseline()
    print(f"\n[1] Baseline cargado: {len(baseline):,} filas, {baseline.shape[1]} columnas")

    # ------------------------------------------------------------------
    # 6.1 DATA DRIFT - lote normal (verde) y lote derivado (rojo)
    # ------------------------------------------------------------------
    print("\n[2] Data drift - lote NORMAL (mismo origen que train)")
    normal = make_normal_batch(baseline, n=1000)
    rep_normal = detect_data_drift(normal, baseline)
    print_report(rep_normal)
    assert not rep_normal.has_drift, "El lote normal no deberia disparar drift"

    print("\n[3] Data drift - lote DERIVADO (inyectado a proposito)")
    drifted = make_drifted_batch(baseline, n=1000)
    rep_drifted = detect_data_drift(drifted, baseline)
    print_report(rep_drifted)
    assert rep_drifted.has_drift, "El lote derivado deberia disparar drift"

    with open(EVIDENCIAS / "fase4_data_drift.json", "w", encoding="utf-8") as f:
        json.dump(
            _json_safe({"normal": rep_normal.to_dict(), "drifted": rep_drifted.to_dict()}),
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n    Evidencia guardada: {EVIDENCIAS / 'fase4_data_drift.json'}")

    # ------------------------------------------------------------------
    # 6.2 CONCEPT DRIFT - lotes sucesivos con etiquetas invertidas
    # ------------------------------------------------------------------
    print("\n[4] Concept drift - cargando modelo champion desde el registry")
    model = load_model()
    df = binarize_target(load_dataset())
    _, X_test, _, y_test = split_train_test(df)
    print(f"    Test: {len(X_test):,} filas")

    print("\n[5] Simulando lotes con etiquetas invertidas crecientes (batch_size=2000)...")
    results = simulate_concept_drift(model, X_test, y_test, n_batches=8, batch_size=2000)
    criterion = check_retraining_criterion(results)
    print_concept_report(results, criterion)

    results.to_csv(EVIDENCIAS / "fase4_concept_drift.csv", index=False)
    with open(EVIDENCIAS / "fase4_concept_drift.json", "w", encoding="utf-8") as f:
        json.dump(
            _json_safe({
                "criterion": criterion,
                "results": results.to_dict(orient="records"),
            }),
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n    Evidencia guardada: {EVIDENCIAS / 'fase4_concept_drift.csv'}")
    print(f"    Evidencia guardada: {EVIDENCIAS / 'fase4_concept_drift.json'}")

    print("\n" + "=" * 90)
    print(" FASE 4 COMPLETADA")
    print("=" * 90)


if __name__ == "__main__":
    main()