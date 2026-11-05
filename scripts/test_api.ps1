# Prueba end-to-end del servicio de inferencia en Docker.
# Uso (desde bank-mlops/):
#   .\scripts\test_api.ps1

param(
    [string]$BaseUrl = "http://localhost:8000"
)

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Test API - $BaseUrl" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. /health
Write-Host "`n[1] GET /health" -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "$BaseUrl/health" -Method Get
$health | ConvertTo-Json -Depth 3

# 2. /info
Write-Host "`n[2] GET /info" -ForegroundColor Yellow
$info = Invoke-RestMethod -Uri "$BaseUrl/info" -Method Get
$info | ConvertTo-Json -Depth 3

# 3. /predict - caso 1 (perfil con probable 'no')
Write-Host "`n[3] POST /predict  (caso 1)" -ForegroundColor Yellow
$body1 = @{
    age = 35; job = "management"; marital = "married"; education = "tertiary"
    default = "no"; balance = 1500; housing = "yes"; loan = "no"
    contact = "cellular"; day = 15; month = "may"; duration = 200
    campaign = 1; pdays = -1; previous = 0; poutcome = "unknown"
} | ConvertTo-Json
$r1 = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $body1 -ContentType "application/json"
$r1 | ConvertTo-Json -Depth 3

# 4. /predict - caso 2 (cliente joven estudiante, suele ser 'yes')
Write-Host "`n[4] POST /predict  (caso 2)" -ForegroundColor Yellow
$body2 = @{
    age = 22; job = "student"; marital = "single"; education = "secondary"
    default = "no"; balance = 500; housing = "no"; loan = "no"
    contact = "cellular"; day = 20; month = "apr"; duration = 450
    campaign = 1; pdays = -1; previous = 0; poutcome = "unknown"
} | ConvertTo-Json
$r2 = Invoke-RestMethod -Uri "$BaseUrl/predict" -Method Post -Body $body2 -ContentType "application/json"
$r2 | ConvertTo-Json -Depth 3

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " API OK - el contenedor respondio correctamente" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green