# Orquesta la Fase 1: verifica MLflow, entrena y registra.
# Uso (desde bank-mlops/):
#   .\scripts\run_fase1.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Activando entorno virtual..." -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "ERROR: no existe .venv. Crea el entorno primero." -ForegroundColor Red
    exit 1
}
. .\.venv\Scripts\Activate.ps1

Write-Host "==> Comprobando servidor MLflow en http://localhost:5000 ..." -ForegroundColor Cyan
try {
    $resp = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 5
    if ($resp.StatusCode -ne 200) { throw "Health no OK" }
    Write-Host "    Servidor MLflow OK" -ForegroundColor Green
} catch {
    Write-Host "ERROR: MLflow no responde. Abre otra terminal y ejecuta:" -ForegroundColor Red
    Write-Host "   .\scripts\run_mlflow_server.ps1" -ForegroundColor Yellow
    exit 1
}

Write-Host "==> Ejecutando entrenamiento (5 runs + Registry + alias)..." -ForegroundColor Cyan
python src\train.py --n-runs 5