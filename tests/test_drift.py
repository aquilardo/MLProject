"""Fase 4 - Tests de data drift y concept drift.

Los tests reproducen lo exigido en el enunciado:
- Data drift: pasa (verde) con datos del mismo origen y falla (rojo)
  con datos derivados inyectados deliberadamente.
- Concept drift: la metrica principal se degrada al invertir etiquetas y
  el criterio de reentrenamiento se dispara cuando corresponde.
"""

from __future__ import annotations

import numpy as np

from src.drift.concept_drift import (
    CONSECUTIVE_BATCHES,
    F1_DROP_THRESHOLD,
    check_retraining_criterion,
    simulate_concept_drift,
)
from src.drift.data_drift import detect_data_drift
from src.inference.model_loader import load_model
from src.utils import binarize_target, load_baseline, load_dataset, split_train_test


def _make_drifted_batch(baseline, n=500, random_state=7):
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


def test_data_drift_green_on_same_source():
    """Verde: un lote del mismo origen que el baseline no debe disparar drift."""
    baseline = load_baseline()
    same = baseline.sample(n=500, random_state=1).reset_index(drop=True)
    report = detect_data_drift(same, baseline)
    assert not report.has_drift, f"Falso positivo: {report.drifted_columns}"


def test_data_drift_red_on_injected_drift():
    """Rojo: un lote derivado a proposito debe disparar drift."""
    baseline = load_baseline()
    drifted = _make_drifted_batch(baseline, n=500)
    report = detect_data_drift(drifted, baseline)
    assert report.has_drift, "El lote derivado no disparo drift (falso negativo)"
    kinds = {c.kind for c in report.columns if c.drifted}
    assert "numeric" in kinds
    assert "categorical" in kinds


def test_concept_drift_degrades_f1_and_triggers_retraining():
    """La inversion de etiquetas degrada F1 y dispara el criterio de reentrenamiento.

    Usamos lotes grandes (2000 filas) para que los primeros lotes (sin flip)
    queden cerca del F1 de referencia y no disparen falsas alarmas.
    """
    model = load_model()
    df = binarize_target(load_dataset())
    _, X_test, _, y_test = split_train_test(df)
    results = simulate_concept_drift(
        model, X_test, y_test, n_batches=6, batch_size=2000,
        flip_schedule=[0.0, 0.0, 0.10, 0.25, 0.45, 0.65],
    )
    # Los primeros lotes (sin flip) no deben alertar
    assert not results.iloc[0]["alert"], (
        f"Falso positivo en lote sin flip (drop={results.iloc[0]['drop_pct']:.2%})"
    )
    assert not results.iloc[1]["alert"], (
        f"Falso positivo en lote sin flip (drop={results.iloc[1]['drop_pct']:.2%})"
    )
    # Los ultimos lotes (flip alto) deben alertar
    assert results.iloc[-1]["alert"]
    assert results.iloc[-2]["alert"]
    # El criterio debe dispararse con suficientes lotes alertados
    criterion = check_retraining_criterion(results)
    assert criterion["retraining_triggered"] is True
    assert criterion["triggered_at_batch"] is not None