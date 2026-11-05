# =====================================================================
#  fix-fase0-fase1.ps1
#  Reconstruye la estructura faltante de Fase 0 + Prueba 1 + Fase 1
#  Ejecutar desde la raíz del proyecto:
#     .\fix-fase0-fase1.ps1
# =====================================================================

$ErrorActionPreference = "Stop"
$Root = (Get-Location).Path
Write-Host "Raiz del proyecto: $Root" -ForegroundColor Cyan

# ---- 1. Carpetas ----
Write-Host "`n[1/12] Creando carpetas..." -ForegroundColor Cyan
$folders = @(
    "data\raw",
    "data\baseline",
    "src\inference",
    "src\drift",
    "tests",
    "docker",
    "k8s",
    "docs\evidencias",
    "scripts",
    ".github\workflows"
)
foreach ($f in $folders) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root $f) | Out-Null
}

# ---- 2. .gitignore ----
Write-Host "[2/12] .gitignore" -ForegroundColor Cyan
@"
__pycache__/
*.py[cod]
*.so
.venv/
env/
venv/
build/
dist/
*.egg-info/
.pytest_cache/
mlruns/
mlflow.db
mlartifacts/
*.pkl
*.joblib
*.log
.vscode/
.idea/
.DS_Store
Thumbs.db
.env
"@ | Set-Content -Path (Join-Path $Root ".gitignore") -Encoding UTF8

# ---- 3. .env.example ----
Write-Host "[3/12] .env.example" -ForegroundColor Cyan
@"
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=bank-classification
MLFLOW_MODEL_NAME=bank_deposit_classifier
MLFLOW_MODEL_ALIAS=champion
"@ | Set-Content -Path (Join-Path $Root ".env.example") -Encoding UTF8

# ---- 4. pytest.ini ----
Write-Host "[4/12] pytest.ini" -ForegroundColor Cyan
@"
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
"@ | Set-Content -Path (Join-Path $Root "pytest.ini") -Encoding UTF8

# ---- 5. conftest.py ----
Write-Host "[5/12] conftest.py" -ForegroundColor Cyan
@"
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
"@ | Set-Content -Path (Join-Path $Root "conftest.py") -Encoding UTF8

# ---- 6. README.md ----
Write-Host "[6/12] README.md" -ForegroundColor Cyan
@"
# Bank Deposit Classifier - Proyecto MLOps

Proyecto end-to-end: MLflow + Docker + Kubernetes + Drift.

## Dataset
- Fuente: data/raw/bank-full.csv (UCI Bank Marketing).
- Problema: clasificacion binaria (y: yes/no).
- Registros: ~45211.

## Fases
- [ ] Fase 1 - Modelo + MLflow
- [ ] Fase 2 - Docker
- [ ] Fase 3 - Kubernetes
- [ ] Fase 4 - Drift
- [ ] Fase 5 (extra) - UI

## Reparto de trabajo
| Integrante | Componentes |
|------------|-------------|
| Nombre 1   | Fase 1      |
| Nombre 2   | Fases 2-3   |
| Nombre 3   | Fase 4      |
"@ | Set-Content -Path (Join-Path $Root "README.md") -Encoding UTF8

# ---- 7. __init__.py en paquetes ----
Write-Host "[7/12] __init__.py" -ForegroundColor Cyan
Set-Content -Path (Join-Path $Root "src\__init__.py")           -Value "# Paquete src"       -Encoding UTF8
Set-Content -Path (Join-Path $Root "src\inference\__init__.py") -Value "# Paquete inference" -Encoding UTF8
Set-Content -Path (Join-Path $Root "src\drift\__init__.py")     -Value "# Paquete drift"     -Encoding UTF8
Set-Content -Path (Join-Path $Root "tests\__init__.py")         -Value "# Paquete tests"     -Encoding UTF8

# ---- 8. src/utils.py ----
Write-Host "[8/12] src/utils.py" -ForegroundColor Cyan
@'
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
'@ | Set-Content -Path (Join-Path $Root "src\utils.py") -Encoding UTF8

# ---- 9. tests/test_smoke.py ----
Write-Host "[9/12] tests/test_smoke.py" -ForegroundColor Cyan
@'
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
'@ | Set-Content -Path (Join-Path $Root "tests\test_smoke.py") -Encoding UTF8

# ---- 10. scripts/run_prueba1.py ----
Write-Host "[10/12] scripts/run_prueba1.py" -ForegroundColor Cyan
@'
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
'@ | Set-Content -Path (Join-Path $Root "scripts\run_prueba1.py") -Encoding UTF8

# ---- 11. src/train.py (Fase 1) ----
Write-Host "[11/12] src/train.py" -ForegroundColor Cyan
@'
"""Fase 1 - Entrenamiento con MLflow + Model Registry."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (CATEGORICAL_COLS, NUMERIC_COLS, RANDOM_STATE,
                       binarize_target, load_dataset, save_baseline,
                       split_train_test)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "bank-classification")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "bank_deposit_classifier")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def build_preprocessor():
    numeric = Pipeline(steps=[("scaler", StandardScaler())])
    categorical = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        transformers=[("num", numeric, NUMERIC_COLS), ("cat", categorical, CATEGORICAL_COLS)],
        remainder="drop",
    )


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }


def get_configs():
    return [
        {"name": "logreg_C1.0",
         "estimator": LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
         "params": {"model_type": "logreg", "C": 1.0, "max_iter": 1000}},
        {"name": "logreg_C0.1",
         "estimator": LogisticRegression(C=0.1, max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1),
         "params": {"model_type": "logreg", "C": 0.1, "max_iter": 1000}},
        {"name": "rf_100",
         "estimator": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
         "params": {"model_type": "random_forest", "n_estimators": 100}},
        {"name": "rf_300_depth10",
         "estimator": RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1),
         "params": {"model_type": "random_forest", "n_estimators": 300, "max_depth": 10}},
        {"name": "gb_150_lr0.05",
         "estimator": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=RANDOM_STATE),
         "params": {"model_type": "gradient_boosting", "n_estimators": 150, "learning_rate": 0.05}},
    ]


def register_model(run_id, model_name, description):
    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    client = MlflowClient()
    client.update_model_version(name=model_name, version=result.version, description=description)
    return model_uri, int(result.version)


def set_alias(model_name, version, alias):
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=str(version))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=5)
    args = parser.parse_args()

    print("=" * 70)
    print(" FASE 1 - Entrenamiento con MLflow + Model Registry")
    print("=" * 70)
    print(f" Tracking URI   : {TRACKING_URI}")
    print(f" Experiment     : {EXPERIMENT_NAME}")
    print(f" Model Registry : {MODEL_NAME}")
    print(f" Alias final    : {MODEL_ALIAS}")
    print(f" Runs a ejecutar: {args.n_runs}")
    print("=" * 70)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("\n[1] Cargando dataset...")
    df = binarize_target(load_dataset())
    X_train, X_test, y_train, y_test = split_train_test(df)
    print(f"    Train: {len(X_train):,} | Test: {len(X_test):,}")

    print("[2] Guardando baseline para drift (Fase 4)...")
    baseline_path = save_baseline(X_train, n_rows=5000)
    print(f"    OK -> {baseline_path}")

    configs = get_configs()[: args.n_runs]
    runs_summary = []

    for i, cfg in enumerate(configs, start=1):
        print(f"\n[3.{i}] Entrenando: {cfg['name']}")
        preprocessor = build_preprocessor()
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("clf", cfg["estimator"])])

        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_params(cfg["params"])
            mlflow.log_param("random_state", RANDOM_STATE)
            pipeline.fit(X_train, y_train)
            metrics = evaluate(pipeline, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("phase", "fase1")
            mlflow.set_tag("dataset", "bank-full")
            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path="model")
            runs_summary.append({"name": cfg["name"], "run_id": run.info.run_id,
                                 "f1": metrics["f1"], "roc_auc": metrics["roc_auc"]})
            print(f"    f1={metrics['f1']:.4f} | roc_auc={metrics['roc_auc']:.4f} | run_id={run.info.run_id[:8]}...")

    runs_summary.sort(key=lambda r: r["f1"], reverse=True)
    best = runs_summary[0]
    second = runs_summary[1] if len(runs_summary) > 1 else runs_summary[0]

    print("\n[4] Ranking por F1:")
    for pos, r in enumerate(runs_summary, start=1):
        print(f"    {pos}. {r['name']:<20} f1={r['f1']:.4f} roc_auc={r['roc_auc']:.4f} run={r['run_id'][:8]}...")

    print("\n[5] Registrando versiones en el Model Registry...")
    _, v1 = register_model(second["run_id"], MODEL_NAME,
                           f"v1 (baseline) - {second['name']} - f1={second['f1']:.4f}")
    print(f"    Version {v1} <- run {second['run_id'][:8]} ({second['name']})")
    _, v2 = register_model(best["run_id"], MODEL_NAME,
                           f"v2 (best) - {best['name']} - f1={best['f1']:.4f}")
    print(f"    Version {v2} <- run {best['run_id'][:8]} ({best['name']})")

    print(f"\n[6] Marcando version {v2} con alias '{MODEL_ALIAS}'...")
    set_alias(MODEL_NAME, v2, MODEL_ALIAS)
    print(f"    OK: {MODEL_NAME}@{MODEL_ALIAS} -> version {v2}")

    print("\n[7] Verificando carga por alias...")
    uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    loaded = mlflow.sklearn.load_model(uri)
    loaded_metrics = evaluate(loaded, X_test, y_test)
    print(f"    Cargado OK desde {uri}")
    print(f"    f1 del modelo cargado: {loaded_metrics['f1']:.4f}")

    print("\n" + "=" * 70)
    print(" FASE 1 COMPLETADA")
    print("=" * 70)
    print(f" Abre MLflow UI: {TRACKING_URI}")
    print(f" - Experiment: {EXPERIMENT_NAME} ({len(runs_summary)} runs)")
    print(f" - Registry  : {MODEL_NAME} (versiones {v1}, {v2})")
    print(f" - Alias     : {MODEL_NAME}@{MODEL_ALIAS} = version {v2}")
    print("=" * 70)


if __name__ == "__main__":
    main()
'@ | Set-Content -Path (Join-Path $Root "src\train.py") -Encoding UTF8

# ---- 12. src/inference/model_loader.py + scripts/check_registry.py ----
Write-Host "[12/12] src/inference/model_loader.py + scripts/check_registry.py" -ForegroundColor Cyan
@'
"""Carga del modelo desde MLflow Model Registry por ALIAS."""

from __future__ import annotations
import os
from functools import lru_cache
import mlflow
import mlflow.sklearn

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "bank_deposit_classifier")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def _model_uri() -> str:
    return f"models:/{MODEL_NAME}@{MODEL_ALIAS}"


@lru_cache(maxsize=1)
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.sklearn.load_model(_model_uri())


def get_model_info() -> dict:
    from mlflow import MlflowClient
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        return {"model_name": MODEL_NAME, "alias": MODEL_ALIAS,
                "version": mv.version, "run_id": mv.run_id, "source": mv.source}
    except Exception as exc:
        return {"model_name": MODEL_NAME, "alias": MODEL_ALIAS, "error": str(exc)}
'@ | Set-Content -Path (Join-Path $Root "src\inference\model_loader.py") -Encoding UTF8

@'
"""Diagnostico del Model Registry."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "bank_deposit_classifier")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    print("=" * 70)
    print(f" Registry: {MODEL_NAME}  (tracking: {TRACKING_URI})")
    print("=" * 70)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No hay versiones registradas. Ejecuta src/train.py primero.")
        return

    versions = sorted(versions, key=lambda v: int(v.version))
    print(f"\nVersiones registradas: {len(versions)}\n")

    for mv in versions:
        alias_set = []
        try:
            full = client.get_model_version(name=MODEL_NAME, version=mv.version)
            alias_set = list(getattr(full, "aliases", []) or [])
        except Exception:
            pass
        marker = " <== " + MODEL_ALIAS if MODEL_ALIAS in alias_set else ""
        print(f"  v{mv.version:<3} run={mv.run_id[:8]}...  status={mv.status}{marker}")
        if mv.description:
            print(f"         desc: {mv.description}")

    print("\nAlias:")
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        print(f"  {MODEL_NAME}@{MODEL_ALIAS} -> version {champion.version} (run {champion.run_id[:8]}...)")
        run = client.get_run(champion.run_id)
        print("\nMetricas del run champion:")
        for k, v in sorted(run.data.metrics.items()):
            print(f"  {k:<10} = {v:.4f}")
    except Exception as exc:
        print(f"  [aviso] no se pudo leer alias: {exc}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
'@ | Set-Content -Path (Join-Path $Root "scripts\check_registry.py") -Encoding UTF8

# ---- Verificacion final ----
Write-Host "`n============================================" -ForegroundColor Green
Write-Host " ESTRUCTURA CREADA" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green

Write-Host "`nArbol del proyecto (excluyendo .git y .pytest_cache):" -ForegroundColor Yellow
Get-ChildItem -Recurse -Force |
    Where-Object { $_.FullName -notmatch '\\\.git\\' -and $_.FullName -notmatch '\\\.pytest_cache' -and $_.FullName -notmatch '\\__pycache__' } |
    ForEach-Object {
        $rel = $_.FullName.Substring($Root.Length + 1)
        if ($_.PSIsContainer) { Write-Host "  [DIR]  $rel" -ForegroundColor DarkCyan }
        else                  { Write-Host "  [FILE] $rel" }
    }

Write-Host "`nVerificacion del CSV:" -ForegroundColor Yellow
if (Test-Path "data\raw\bank-full.csv") {
    $csv = Get-Item "data\raw\bank-full.csv"
    Write-Host ("  OK: " + $csv.FullName + " -> " + [math]::Round($csv.Length/1MB,2) + " MB") -ForegroundColor Green
} else {
    Write-Host "  FALTA: data\raw\bank-full.csv" -ForegroundColor Red
    Write-Host "  Copia bank-full.csv a esa ruta antes de continuar." -ForegroundColor Yellow
}

Write-Host "`nSiguientes pasos:" -ForegroundColor Magenta
Write-Host "  1) python -m pytest tests -q      (debe dar 4 passed, 1 skipped)"
Write-Host "  2) python scripts\run_prueba1.py  (debe decir PRUEBA 1 SUPERADA)"
Write-Host "  3) python -m pytest tests -q      (debe dar 5 passed)"
Write-Host ""