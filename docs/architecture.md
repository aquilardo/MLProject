# Documento de arquitectura

## 1. Problema y dataset
- Dataset: `bank-full.csv` (UCI Bank Marketing), 45.211 registros, 16 features + target.
- Target: `y` (yes/no) -> 1/0. Desbalance ~88/12.
- Métrica principal: **F1-score**. Secundaria: ROC-AUC.

## 2. Flujo de arquitectura
## 4. Modelo desplegado
| Campo | Valor |
|-------|-------|
| Model Registry | `bank_deposit_classifier` |
| Alias desplegado | `champion` |
| Versión desplegada | **v2** |
| Run ID asociado | `3b5087a3...` (cópialo entero de la UI o de `check_registry.py`) |
| Configuración elegida | `rf_100` (RandomForest n_estimators=100) |
| F1 (test) | **0.4924** |
| ROC-AUC (test) | _revisar en check_registry.py_ |

### Justificación de la elección
- `rf_100` obtuvo el mayor F1 en test entre los 5 runs (comparación en UI MLflow).
- Se descartaron los modelos lineales por F1 claramente inferior.
- `gb_150_lr0.05` quedó muy cerca y se registró como v1 (baseline).
- El servicio de inferencia consume `models:/bank_deposit_classifier@champion`, no una ruta de archivo.