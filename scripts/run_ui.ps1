# Lanza la UI Streamlit contra el servicio desplegado en Kubernetes.
# Uso (desde bank-mlops/):
#   .\scripts\run_ui.ps1
#   .\scripts\run_ui.ps1 -ApiBaseUrl "http://localhost:30080"

param(
    [string]$ApiBaseUrl = "http://localhost:30080"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Verificando que el servicio responde en $ApiBaseUrl ..." -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$ApiBaseUrl/health" -Method Get -TimeoutSec 5
    Write-Host "    Servicio OK. Pod que respondio: $($health.pod)" -ForegroundColor Green
} catch {
    Write-Host "ERROR: el servicio no responde en $ApiBaseUrl" -ForegroundColor Red
    Write-Host "Comprueba que el deployment de K8s esta arriba:" -ForegroundColor Yellow
    Write-Host "  kubectl get pods -n bank-mlops" -ForegroundColor Yellow
    exit 1
}

Write-Host "==> Instalando dependencias de la UI (si faltan)..." -ForegroundColor Cyan
pip install -q -r ui/requirements.txt

Write-Host "==> Lanzando Streamlit..." -ForegroundColor Cyan
$env:API_BASE_URL = $ApiBaseUrl
streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0