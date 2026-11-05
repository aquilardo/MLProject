# Demuestra las 4 pruebas exigidas en la Fase 3.
# Uso (desde bank-mlops/):
#   .\scripts\test_k8s.ps1

param(
    [string]$BaseUrl = "http://localhost:30080",
    [string]$Namespace = "bank-mlops"
)

$ErrorActionPreference = "Stop"

function Get-PredictBody {
    return @{
        age = 35; job = "management"; marital = "married"; education = "tertiary"
        default = "no"; balance = 1500; housing = "yes"; loan = "no"
        contact = "cellular"; day = 15; month = "may"; duration = 200
        campaign = 1; pdays = -1; previous = 0; poutcome = "unknown"
    } | ConvertTo-Json
}

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " FASE 3 - Pruebas de Kubernetes" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------
# DEMO 1: 3 replicas Running
# ---------------------------------------------------------------
Write-Host "`n[DEMO 1] Verificando 3 replicas en estado Running..." -ForegroundColor Yellow
$pods = kubectl get pods -n $Namespace -l app=bank-inference -o json | ConvertFrom-Json
$running = ($pods.items | Where-Object { $_.status.phase -eq "Running" }).Count
kubectl get pods -n $Namespace -l app=bank-inference
if ($running -ge 3) {
    Write-Host "  OK: $running replicas Running" -ForegroundColor Green
} else {
    Write-Host "  ERROR: solo $running replicas Running (se esperaban 3)" -ForegroundColor Red
}

# ---------------------------------------------------------------
# DEMO 2: Balanceo entre pods
# ---------------------------------------------------------------
Write-Host "`n[DEMO 2] Balanceo de carga (10 peticiones al Service)..." -ForegroundColor Yellow
$body = Get-PredictBody
$podsSeen = @{}
for ($i = 1; $i -le 10; $i++) {
    try {
        $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 5
        $pod = $r.pod
        if ($podsSeen.ContainsKey($pod)) { $podsSeen[$pod]++ } else { $podsSeen[$pod] = 1 }
        Write-Host ("  Peticion {0,2}: atendida por {1}" -f $i, $pod)
    } catch {
        Write-Host "  Peticion $i falló: $_" -ForegroundColor Red
    }
}
Write-Host "`n  Resumen de pods que atendieron:" -ForegroundColor Cyan
$podsSeen.GetEnumerator() | ForEach-Object { Write-Host ("    {0} -> {1} peticiones" -f $_.Key, $_.Value) }
if ($podsSeen.Count -ge 2) {
    Write-Host "  OK: el trafico se distribuyo entre $($podsSeen.Count) pods distintos" -ForegroundColor Green
} else {
    Write-Host "  ADVERTENCIA: solo un pod atendio. Revisa el Service." -ForegroundColor DarkYellow
}

# ---------------------------------------------------------------
# DEMO 3: Autorreparacion
# ---------------------------------------------------------------
Write-Host "`n[DEMO 3] Autorreparacion (borrando un pod)..." -ForegroundColor Yellow
$podToDelete = (kubectl get pods -n $Namespace -l app=bank-inference -o jsonpath="{.items[0].metadata.name}")
Write-Host "  Borrando pod: $podToDelete"
kubectl delete pod $podToDelete -n $Namespace --wait=false

Write-Host "  Esperando a que Kubernetes lo recree..." -ForegroundColor Cyan
Start-Sleep -Seconds 8

Write-Host "  Estado actual de pods:" -ForegroundColor Cyan
kubectl get pods -n $Namespace -l app=bank-inference

Write-Host "  Verificando que el servicio sigue respondiendo..." -ForegroundColor Cyan
Start-Sleep -Seconds 5
try {
    $r = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $body -ContentType "application/json" -TimeoutSec 5
    Write-Host "  OK: el servicio responde tras eliminar el pod (atendio $($r.pod))" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: el servicio no responde tras eliminar el pod: $_" -ForegroundColor Red
}

# ---------------------------------------------------------------
# DEMO 4: Escalado
# ---------------------------------------------------------------
Write-Host "`n[DEMO 4] Escalado (3 -> 5 replicas)..." -ForegroundColor Yellow
Write-Host "  Escalando a 5 replicas..."
kubectl scale deployment bank-inference -n $Namespace --replicas=5

Write-Host "  Esperando a que los nuevos pods esten listos..." -ForegroundColor Cyan
Start-Sleep -Seconds 10
kubectl get pods -n $Namespace -l app=bank-inference

$podsAfter = kubectl get pods -n $Namespace -l app=bank-inference -o json | ConvertFrom-Json
$runningAfter = ($podsAfter.items | Where-Object { $_.status.phase -eq "Running" }).Count
if ($runningAfter -ge 5) {
    Write-Host "  OK: $runningAfter replicas Running tras el escalado" -ForegroundColor Green
} else {
    Write-Host "  ADVERTENCIA: solo $runningAfter replicas Running" -ForegroundColor DarkYellow
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " FASE 3 - TODAS LAS DEMOS COMPLETADAS" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green