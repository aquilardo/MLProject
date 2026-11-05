"""Prueba 1 - Smoke test del dataset y pipeline base."""

import pandas as pd
import pytest

from src.utils import (
    BASELINE_FILE, CATEGORICAL_COLS, EXPECTED_COLUMNS, NUMERIC_COLS,
    TARGET_COL, binarize_target, load_baseline, load_dataset,
    save_baseline, split_train_test,
)


def test_dataset_loads_and_has_expected_columns():
    df = load_dataset()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 1000
    assert set(EXPECTED_COLUMNS).issubset(set(df.columns))
    assert set(NUMERIC_COLS).issubset(set(df.columns))
    assert set(CATEGORICAL_COLS).issubset(set(df.columns))


def test_target_binarization_and_imbalance():
    df = binarize_target(load_dataset())
    assert df[TARGET_COL].isin([0, 1]).all()
    positive_rate = df[TARGET_COL].mean()
    assert 0.05 < positive_rate < 0.25


def test_split_is_reproducible_and_stratified():
    df = binarize_target(load_dataset())
    X_tr1, X_te1, y_tr1, y_te1 = split_train_test(df)
    X_tr2, X_te2, y_tr2, y_te2 = split_train_test(df)
    assert X_tr1.equals(X_tr2)
    assert y_tr1.equals(y_tr2)
    assert abs(y_tr1.mean() - y_te1.mean()) < 0.01
    assert abs(len(X_te1) / len(df) - 0.20) < 0.01


def test_baseline_saves_and_reloads(tmp_path, monkeypatch):
    from src import utils
    fake_baseline = tmp_path / "baseline.csv"
    monkeypatch.setattr(utils, "BASELINE_FILE", fake_baseline)
    monkeypatch.setattr(utils, "DATA_BASELINE_DIR", tmp_path)

    df = binarize_target(load_dataset())
    X_train, _, _, _ = split_train_test(df)
    path = save_baseline(X_train, n_rows=200)
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == 200
    assert set(NUMERIC_COLS + CATEGORICAL_COLS).issubset(set(reloaded.columns))
    assert TARGET_COL not in reloaded.columns


def test_scripts_have_expected_real_baseline():
    if not BASELINE_FILE.exists():
        pytest.skip("Baseline real no generado aun. Ejecuta scripts/run_prueba1.py")
    baseline = load_baseline()
    assert len(baseline) > 0
    assert TARGET_COL not in baseline.columns
    assert set(NUMERIC_COLS + CATEGORICAL_COLS).issubset(set(baseline.columns))
