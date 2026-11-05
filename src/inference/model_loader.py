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
