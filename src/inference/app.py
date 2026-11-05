"""
Servicio de inferencia - FastAPI.

Endpoints:
- GET  /health   -> liveness/readiness
- GET  /info     -> metadatos del modelo cargado (version, run_id, pod)
- POST /predict  -> prediccion a partir de un JSON con las 16 features

Devuelve el HOSTNAME (nombre del pod en K8s) en cada respuesta para
demostrar balanceo de carga entre replicas.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path
from typing import Literal

# Permitir ejecutar el modulo desde cualquier cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.model_loader import get_model_info, load_model

# Identificador del pod / contenedor
POD_NAME = os.getenv("HOSTNAME", socket.gethostname())

app = FastAPI(
    title="Bank Deposit Classifier API",
    description="Servicio de inferencia para predecir contratacion de deposito a plazo.",
    version="1.0.0",
)


# ---------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------
class PredictRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    job: str = Field(..., example="management")
    marital: str = Field(..., example="married")
    education: str = Field(..., example="tertiary")
    default: Literal["yes", "no"] = Field(..., example="no")
    balance: int = Field(..., example=1500)
    housing: Literal["yes", "no"] = Field(..., example="yes")
    loan: Literal["yes", "no"] = Field(..., example="no")
    contact: str = Field(..., example="cellular")
    day: int = Field(..., ge=1, le=31, example=15)
    month: str = Field(..., example="may")
    duration: int = Field(..., ge=0, example=200)
    campaign: int = Field(..., ge=1, example=1)
    pdays: int = Field(..., example=-1)
    previous: int = Field(..., ge=0, example=0)
    poutcome: str = Field(..., example="unknown")


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    probability_yes: float
    pod: str
    model_version: str | None = None


# ---------------------------------------------------------------------
# Carga del modelo al arrancar (falla rapido si no esta)
# ---------------------------------------------------------------------
_model = None


@app.on_event("startup")
def _startup() -> None:
    global _model
    try:
        _model = load_model()
    except Exception as exc:  # noqa: BLE001
        # Dejamos que el contenedor arranque, pero /predict fallara
        print(f"[startup] ERROR cargando modelo: {exc}", flush=True)


# ---------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "pod": POD_NAME}


@app.get("/info")
def info() -> dict:
    return {"pod": POD_NAME, **get_model_info()}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    global _model
    if _model is None:
        try:
            _model = load_model()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"Modelo no disponible: {exc}")

    # Convertir a DataFrame de una fila respetando nombres de columnas
    import pandas as pd

    payload = req.model_dump()
    X = pd.DataFrame([payload])

    proba = float(_model.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)

    info_dict = get_model_info()
    return PredictResponse(
        prediction=pred,
        prediction_label="yes" if pred == 1 else "no",
        probability_yes=round(proba, 6),
        pod=POD_NAME,
        model_version=str(info_dict.get("version")) if info_dict.get("version") else None,
    )