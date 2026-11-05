# Elimina todos los recursos de la Fase 3 del cluster.
# Uso (desde bank-mlops/):
#   .\scripts\teardown_k8s.ps1

$ErrorActionPreference = "Continue"
$Namespace = "bank-mlops"

Write-Host "==> Borrando Deployment y Service..." -ForegroundColor Cyan
kubectl delete -f k8s/deployment.yaml --ignore-not-found
kubectl delete -f k8s/service.yaml --ignore-not-found

Write-Host "==> Borrando Namespace..." -ForegroundColor Cyan
kubectl delete -f k8s/namespace.yaml --ignore-not-found

Write-Host "==> Verificando limpieza..." -ForegroundColor Green
kubectl get all -n $Namespace 2>$null
Write-Host "Listo." -ForegroundColor Green