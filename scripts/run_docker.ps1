# Ejecuta el contenedor del servicio de inferencia usando --mount (evita el error de "too many colons" en Windows).
# Uso (desde bank-mlops/):
#   .\scripts\run_docker.ps1

param(
    [string]$Tag = "bank-inference:v1",
    [int]$HostPort = 8000,
    [string]$MlflowUri = "http://host.docker.internal:5000"
)

$ErrorActionPreference = "Stop"

# Ruta absoluta del proyecto y de los artefactos en el host
$ProjectRoot = (Get-Location).Path
$ArtifactsHost = Join-Path $ProjectRoot "mlartifacts"

if (-not (Test-Path $ArtifactsHost)) {
    Write-Host "ERROR: no existe la carpeta de artefactos: $ArtifactsHost" -ForegroundColor Red
    exit 1
}

# Convertir la ruta Windows (D:\...) al formato que Docker entiende (/D:/...)
$HostPathForDocker = "/" + ($ArtifactsHost -replace '\\', '/' -replace '^([A-Za-z]):', '$1:')

# La ruta dentro del contenedor debe ser la misma que la registrada en MLflow
$ContainerPath = $HostPathForDocker

Write-Host "==> Parando contenedor previo (si existe)..." -ForegroundColor Cyan
docker rm -f bank-inference 2>$null | Out-Null

Write-Host "==> Montando artefactos con --mount:" -ForegroundColor Cyan
Write-Host "    source : $HostPathForDocker"
Write-Host "    target : $ContainerPath"

Write-Host "==> Lanzando $Tag en el puerto $HostPort ..." -ForegroundColor Cyan
docker run -d --name bank-inference `
    -p "${HostPort}:8000" `
    --mount "type=bind,source=$HostPathForDocker,target=$ContainerPath" `
    -e "MLFLOW_TRACKING_URI=$MlflowUri" `
    -e "MLFLOW_MODEL_NAME=bank_deposit_classifier" `
    -e "MLFLOW_MODEL_ALIAS=champion" `
    $Tag

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: docker run fallo." -ForegroundColor Red
    exit 1
}

Write-Host "==> Esperando a que el servicio este listo..." -ForegroundColor Cyan
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:${HostPort}/health" -UseBasicParsing -TimeoutSec 3
        if ($r.StatusCode -eq 200) { $ok = $true; break }
    } catch { }
}

if (-not $ok) {
    Write-Host "ERROR: el servicio no respondio a tiempo. Revisa logs:" -ForegroundColor Red
    docker logs bank-inference --tail 50
    exit 1
}

Write-Host "==> Servicio arriba en http://localhost:${HostPort}" -ForegroundColor Green
docker ps --filter "name=bank-inference"