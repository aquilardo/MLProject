"""Fase 4 - Data drift: compara un lote nuevo contra el baseline de entrenamiento.

Pruebas estadisticas por tipo de variable:
- Numericas: Kolmogorov-Smirnov (KS) de dos muestras.
  Se elige KS porque no asume normalidad, es sensible a cambios en la forma
  de la distribucion y funciona con muestras de tamanos distintos (baseline
  vs lote).
- Categoricas: Chi-cuadrado de homogeneidad.
  Se elige chi^2 porque compara las frecuencias observadas por categoria
  entre baseline y lote; es la prueba estandar para variables nominales.

Umbral de alerta:
- KS: p-value < 0.05 => drift (nivel de significancia habitual, 5%).
- Chi^2: p-value < 0.05 => drift.
- Ademas se calcula PSI (Population Stability Index) como metrica de magnitud:
    PSI < 0.10  -> sin cambio
    0.10 <= PSI < 0.25 -> cambio moderado
    PSI >= 0.25 -> cambio critico
  Estos umbrales son los convencionales en scorecards de riesgo (Siddiqi, 2006).
  Se alerta si PSI >= 0.25 o si p-value < 0.05 en KS/chi^2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from src.utils import CATEGORICAL_COLS, NUMERIC_COLS, load_baseline

ALPHA = 0.05
PSI_CRITICAL = 0.25
PSI_MODERATE = 0.10
N_BINS_PSI = 10


@dataclass
class ColumnDrift:
    column: str
    kind: str            # "numeric" | "categorical"
    test: str            # "ks" | "chi2"
    statistic: float
    p_value: float
    psi: float
    drifted: bool
    reason: str


@dataclass
class DriftReport:
    n_baseline: int
    n_new: int
    columns: List[ColumnDrift] = field(default_factory=list)

    @property
    def drifted_columns(self) -> List[str]:
        return [c.column for c in self.columns if c.drifted]

    @property
    def has_drift(self) -> bool:
        return len(self.drifted_columns) > 0

    def to_dict(self) -> Dict:
        return {
            "n_baseline": int(self.n_baseline),
            "n_new": int(self.n_new),
            "has_drift": bool(self.has_drift),
            "drifted_columns": [str(c) for c in self.drifted_columns],
            "columns": [
                {
                    "column": str(c.column),
                    "kind": str(c.kind),
                    "test": str(c.test),
                    "statistic": float(c.statistic),
                    "p_value": float(c.p_value),
                    "psi": float(c.psi),
                    "drifted": bool(c.drifted),
                    "reason": str(c.reason),
                }
                for c in self.columns
            ],
        }


def _psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = N_BINS_PSI) -> float:
    expected = expected.dropna()
    actual = actual.dropna()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    try:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    except Exception:
        return 0.0
    if len(edges) < 2:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = exp_counts / max(exp_counts.sum(), 1)
    act_pct = act_counts / max(act_counts.sum(), 1)
    exp_pct = np.where(exp_pct == 0, 1e-6, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-6, act_pct)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    expected = expected.astype(str).fillna("__NA__")
    actual = actual.astype(str).fillna("__NA__")
    categories = sorted(set(expected.unique()) | set(actual.unique()))
    exp_pct = expected.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    act_pct = actual.value_counts(normalize=True).reindex(categories, fill_value=0.0)
    exp_pct = exp_pct.replace(0.0, 1e-6)
    act_pct = act_pct.replace(0.0, 1e-6)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def _test_numeric(col: str, baseline: pd.Series, new: pd.Series) -> ColumnDrift:
    b = baseline.dropna().astype(float)
    n = new.dropna().astype(float)
    stat, p = stats.ks_2samp(b, n)
    psi = _psi_numeric(b, n)
    drifted = bool((p < ALPHA) or (psi >= PSI_CRITICAL))
    reason_parts = []
    if p < ALPHA:
        reason_parts.append(f"KS p={p:.4f} < {ALPHA}")
    if psi >= PSI_CRITICAL:
        reason_parts.append(f"PSI={psi:.3f} >= {PSI_CRITICAL}")
    reason = " y ".join(reason_parts) if reason_parts else "sin cambio significativo"
    return ColumnDrift(col, "numeric", "ks", float(stat), float(p), float(psi), drifted, reason)


def _test_categorical(col: str, baseline: pd.Series, new: pd.Series) -> ColumnDrift:
    b = baseline.astype(str).fillna("__NA__")
    n = new.astype(str).fillna("__NA__")
    categories = sorted(set(b.unique()) | set(n.unique()))
    b_counts = b.value_counts().reindex(categories, fill_value=0).values
    n_counts = n.value_counts().reindex(categories, fill_value=0).values
    table = np.vstack([b_counts, n_counts])
    keep = table.sum(axis=0) > 0
    table = table[:, keep]
    if table.shape[1] < 2:
        stat, p = 0.0, 1.0
    else:
        stat, p, _, _ = stats.chi2_contingency(table, correction=False)
    psi = _psi_categorical(b, n)
    drifted = bool((p < ALPHA) or (psi >= PSI_CRITICAL))
    reason_parts = []
    if p < ALPHA:
        reason_parts.append(f"chi2 p={p:.4f} < {ALPHA}")
    if psi >= PSI_CRITICAL:
        reason_parts.append(f"PSI={psi:.3f} >= {PSI_CRITICAL}")
    reason = " y ".join(reason_parts) if reason_parts else "sin cambio significativo"
    return ColumnDrift(col, "categorical", "chi2", float(stat), float(p), float(psi), drifted, reason)


def detect_data_drift(new_data: pd.DataFrame, baseline: pd.DataFrame | None = None) -> DriftReport:
    """Compara new_data contra baseline. Si baseline es None, carga data/baseline/baseline.csv."""
    if baseline is None:
        baseline = load_baseline()

    report = DriftReport(n_baseline=len(baseline), n_new=len(new_data))

    for col in NUMERIC_COLS:
        if col in new_data.columns and col in baseline.columns:
            report.columns.append(_test_numeric(col, baseline[col], new_data[col]))

    for col in CATEGORICAL_COLS:
        if col in new_data.columns and col in baseline.columns:
            report.columns.append(_test_categorical(col, baseline[col], new_data[col]))

    return report


def print_report(report: DriftReport) -> None:
    print("=" * 78)
    print(f" DATA DRIFT  (baseline={report.n_baseline:,}  nuevo={report.n_new:,})")
    print("=" * 78)
    print(f"{'columna':<12}{'tipo':<14}{'prueba':<8}{'stat':>10}{'p-value':>10}{'PSI':>8}{'estado':>10}")
    print("-" * 78)
    for c in report.columns:
        estado = "DRIFT" if c.drifted else "ok"
        print(f"{c.column:<12}{c.kind:<14}{c.test:<8}{c.statistic:>10.4f}"
              f"{c.p_value:>10.4f}{c.psi:>8.3f}{estado:>10}")
    print("-" * 78)
    if report.has_drift:
        print(f"RESULTADO: DRIFT detectado en {len(report.drifted_columns)} columnas: "
              f"{report.drifted_columns}")
    else:
        print("RESULTADO: sin drift significativo")
    print("=" * 78)