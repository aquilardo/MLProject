"""Utilidades comunes: carga, split y baseline."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COL = "y"
CSV_SEP = ";"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "bank-full.csv"
DATA_BASELINE_DIR = PROJECT_ROOT / "data" / "baseline"
BASELINE_FILE = DATA_BASELINE_DIR / "baseline.csv"

EXPECTED_COLUMNS = [
    "age", "job", "marital", "education", "default", "balance",
    "housing", "loan", "contact", "day", "month", "duration",
    "campaign", "pdays", "previous", "poutcome", "y",
]

NUMERIC_COLS = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
CATEGORICAL_COLS = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "poutcome",
]


def load_dataset(path=None) -> pd.DataFrame:
    csv_path = Path(path) if path is not None else DATA_RAW
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el CSV en: {csv_path}")
    df = pd.read_csv(csv_path, sep=CSV_SEP)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas esperadas: {missing}")
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"yes": 1, "no": 0}).astype(int)
    return df


def split_train_test(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


def save_baseline(X_train, n_rows=5000, random_state=RANDOM_STATE):
    DATA_BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    n = min(n_rows, len(X_train))
    baseline = X_train.sample(n=n, random_state=random_state).reset_index(drop=True)
    baseline.to_csv(BASELINE_FILE, index=False)
    return BASELINE_FILE


def load_baseline() -> pd.DataFrame:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"No existe baseline en {BASELINE_FILE}. Ejecuta scripts/run_prueba1.py"
        )
    return pd.read_csv(BASELINE_FILE)
