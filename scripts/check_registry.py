"""Diagnostico del Model Registry."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "bank_deposit_classifier")
MODEL_ALIAS = os.getenv("MLFLOW_MODEL_ALIAS", "champion")


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    print("=" * 70)
    print(f" Registry: {MODEL_NAME}  (tracking: {TRACKING_URI})")
    print("=" * 70)

    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No hay versiones registradas. Ejecuta src/train.py primero.")
        return

    versions = sorted(versions, key=lambda v: int(v.version))
    print(f"\nVersiones registradas: {len(versions)}\n")

    for mv in versions:
        alias_set = []
        try:
            full = client.get_model_version(name=MODEL_NAME, version=mv.version)
            alias_set = list(getattr(full, "aliases", []) or [])
        except Exception:
            pass
        marker = " <== " + MODEL_ALIAS if MODEL_ALIAS in alias_set else ""
        print(f"  v{mv.version:<3} run={mv.run_id[:8]}...  status={mv.status}{marker}")
        if mv.description:
            print(f"         desc: {mv.description}")

    print("\nAlias:")
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        print(f"  {MODEL_NAME}@{MODEL_ALIAS} -> version {champion.version} (run {champion.run_id[:8]}...)")
        run = client.get_run(champion.run_id)
        print("\nMetricas del run champion:")
        for k, v in sorted(run.data.metrics.items()):
            print(f"  {k:<10} = {v:.4f}")
    except Exception as exc:
        print(f"  [aviso] no se pudo leer alias: {exc}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
