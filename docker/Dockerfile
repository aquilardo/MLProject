# Imagen base reproducible
FROM python:3.11-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencias del sistema minimas
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python (fijadas)
COPY requirements.txt /app/requirements.txt
RUN pip install --only-binary=:all: -r requirements.txt

# Copiar codigo
COPY src /app/src
COPY conftest.py /app/conftest.py

# Variables por defecto (se sobreescriben en runtime)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
    MLFLOW_MODEL_NAME=bank_deposit_classifier \
    MLFLOW_MODEL_ALIAS=champion \
    PORT=8000

EXPOSE 8000

# Healthcheck del contenedor
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# Arrancar el servicio
CMD ["uvicorn", "src.inference.app:app", "--host", "0.0.0.0", "--port", "8000"]