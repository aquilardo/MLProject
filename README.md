# Bank Deposit Classifier - Proyecto MLOps

Proyecto end-to-end de clasificación binaria con trazabilidad en MLflow, empaquetado en Docker,
despliegue en Kubernetes con réplicas, monitoreo de drift y UI de consumo.

## Dataset
- Fuente: `data/raw/bank-full.csv` (UCI Bank Marketing).
- Problema: clasificación binaria (`y`: yes/no).
- Registros: ~45.211.
- Métrica principal: **F1-score**. Secundaria: ROC-AUC.

## Fases
- [x] Fase 1 - Modelo + MLflow (experimentos, registry, alias)
- [x] Fase 2 - Docker (imagen reproducible)
- [x] Fase 3 - Kubernetes (3 réplicas, balanceo, autorreparación, escalado)
- [x] Fase 4 - Drift (data drift + concept drift)
- [x] Fase 5 (extra) - UI Streamlit contra el servicio en K8s

## Reparto de trabajo

| Integrante | Componentes asignados | Commits principales |
|------------|-----------------------|---------------------|
| **Grover Plaza Quiroga** | Fase 1: `src/train.py`, `src/utils.py`, `src/inference/model_loader.py`, `scripts/check_registry.py`, `docs/architecture.md`.<br>Fase 4: `src/drift/data_drift.py`, `src/drift/concept_drift.py`, `scripts/run_fase4.py`, `tests/test_drift.py`. | `feat: fase 1 entrenamiento y registry`<br>`feat: fase 4 deteccion de data y concept drift` |
| **Aquilardo Caricari Cala** | Fase 2: `docker/Dockerfile`, `requirements.txt`, `scripts/build_docker.ps1`.<br>Fase 3: `k8s/*.yaml`, `scripts/deploy_k8s.ps1`, `scripts/test_k8s.ps1`.<br>Fase 5 (extra): `ui/app.py`, `ui/requirements.txt`, `scripts/run_ui.ps1`. | `feat: fase 2 dockerizacion`<br>`feat: fase 3 despliegue k8s`<br>`feat: ui streamlit` |

## Requisitos previos
- Python 3.11 con el entorno `bank-mlops` (conda o venv).
- Docker Desktop con Kubernetes habilitado.
- MLflow 2.16.2, dependencias en `requirements.txt`.
- Dataset en `data/raw/bank-full.csv`.

## Cómo reproducir

### 1. Entorno

```powershell
conda create -n bank-mlops python=3.11 -y
conda activate bank-mlops
pip install -r requirements.txt
