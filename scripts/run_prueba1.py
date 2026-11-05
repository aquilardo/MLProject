"""Ejecuta la Prueba 1 y genera el baseline real."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.utils import (
    BASELINE_FILE, DATA_RAW, binarize_target, load_dataset,
    save_baseline, split_train_test,
)


def main():
    print("=" * 60)
    print(" PRUEBA 1 - Smoke test del dataset y pipeline base")
    print("=" * 60)

    print(f"\n[1] Cargando dataset desde: {DATA_RAW}")
    df = load_dataset()
    print(f"    Filas: {len(df):,} | Columnas: {df.shape[1]}")

    print("\n[2] Binarizando target 'y'")
    df = binarize_target(df)
    counts = df["y"].value_counts().sort_index()
    for label, count in counts.items():
        name = "yes" if label == 1 else "no"
        print(f"      {label} ({name}): {count:,} ({100*count/len(df):.2f}%)")
    assert 0.05 < df["y"].mean() < 0.25

    print("\n[3] Split train/test (80/20 estratificado, random_state=42)")
    X_train, X_test, y_train, y_test = split_train_test(df)
    print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")
    assert abs(y_train.mean() - y_test.mean()) < 0.01

    print(f"\n[4] Generando baseline en: {BASELINE_FILE}")
    path = save_baseline(X_train, n_rows=5000)
    print(f"    Guardado en: {path}")
    print(f"    Tamano: {path.stat().st_size / 1024:.1f} KB")

    baseline = pd.read_csv(path)
    print(f"    Releido: {len(baseline)} filas, {baseline.shape[1]} columnas")
    assert "y" not in baseline.columns

    print("\n" + "=" * 60)
    print(" PRUEBA 1 SUPERADA")
    print("=" * 60)


if __name__ == "__main__":
    main()
