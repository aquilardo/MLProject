# Construye la imagen Docker del servicio de inferencia.
# Uso (desde bank-mlops/):
#   .\scripts\build_docker.ps1

param(
    [string]$Tag = "bank-inference:v1"
)

$ErrorActionPreference = "Stop"

Write-Host "==> Verificando Docker..." -ForegroundColor Cyan
docker --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Docker no esta disponible. Abre Docker Desktop." -ForegroundColor Red
    exit 1
}

Write-Host "==> Construyendo imagen $Tag desde docker/Dockerfile ..." -ForegroundColor Cyan
docker build -f docker/Dockerfile -t $Tag .

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: build fallo." -ForegroundColor Red
    exit 1
}

Write-Host "==> Imagen construida:" -ForegroundColor Green
docker images $Tag