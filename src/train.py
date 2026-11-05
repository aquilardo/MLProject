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
