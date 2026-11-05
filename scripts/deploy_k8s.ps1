# Despliega la infraestructura en Kubernetes.
# Uso (desde bank-mlops/):
#   .\scripts\deploy_k8s.ps1

$ErrorActionPreference = "Stop"

Write-Host "==> Aplicando Namespace..." -ForegroundColor Cyan
kubectl apply -f k8s/namespace.yaml

Write-Host "==> Aplicando Deployment (3 replicas)..." -ForegroundColor Cyan
kubectl apply -f k8s/deployment.yaml

Write-Host "==> Aplicando Service (NodePort 30080)..." -ForegroundColor Cyan
kubectl apply -f k8s/service.yaml

Write-Host "==> Esperando a que los pods esten listos..." -ForegroundColor Cyan
kubectl wait --for=condition=available --timeout=120s deployment/bank-inference -n bank-mlops

Write-Host "==> Estado de los pods:" -ForegroundColor Green
kubectl get pods -n bank-mlops -o wide

Write-Host "==> Estado del Service:" -ForegroundColor Green
kubectl get svc -n bank-mlops

Write-Host "==> El servicio esta disponible en http://localhost:30080" -ForegroundColor Magenta