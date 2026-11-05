# Ejecuta el servidor MLflow local (SQLite + artifacts gestionados por MLflow)
# Uso (desde bank-mlops/):
#   .\scripts\run_mlflow_server.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " MLflow Tracking Server" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Backend  : sqlite:///mlflow.db"
Write-Host " Artifacts: mlflow-artifacts:/  (servidos por HTTP)"
Write-Host " URL      : http://localhost:5000"
Write-Host " (Ctrl+C para detener)"
Write-Host "==========================================================" -ForegroundColor Cyan

mlflow server `
  --host 0.0.0.0 `
  --port 5000 `
  --backend-store-uri "sqlite:///mlflow.db" `
  --default-artifact-root "mlflow-artifacts:/"