"""Extra - UI Streamlit que consume el servicio desplegado en Kubernetes.

Uso:
    streamlit run ui/app.py

La URL por defecto es http://localhost:30080 (NodePort del Service en K8s).
Se puede sobreescribir con la variable de entorno API_BASE_URL.
"""

from __future__ import annotations

import os
from collections import Counter

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:30080")

JOB_OPTIONS = [
    "admin.", "unknown", "unemployed", "management", "housemaid",
    "entrepreneur", "student", "blue-collar", "self-employed", "retired",
    "technician", "services",
]
MARITAL_OPTIONS = ["married", "divorced", "single"]
EDUCATION_OPTIONS = ["unknown", "secondary", "primary", "tertiary"]
YES_NO = ["yes", "no"]
CONTACT_OPTIONS = ["unknown", "telephone", "cellular"]
MONTH_OPTIONS = ["jan", "feb", "mar", "apr", "may", "jun",
                 "jul", "aug", "sep", "oct", "nov", "dec"]
POUTCOME_OPTIONS = ["unknown", "other", "failure", "success"]


def _api_get(path: str, timeout: float = 5.0) -> dict:
    r = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _api_predict(payload: dict, timeout: float = 10.0) -> dict:
    r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()


st.set_page_config(page_title="Bank Deposit Classifier", page_icon="🏦", layout="wide")

st.title("🏦 Bank Deposit Classifier")
st.caption(f"Consumiendo el servicio en **{API_BASE_URL}** (Kubernetes Service, NodePort 30080)")

# ---------------------------------------------------------------------
# Sidebar: estado del servicio y pod que atendió
# ---------------------------------------------------------------------
with st.sidebar:
    st.header("Estado del servicio")
    if st.button("Refrescar estado"):
        st.rerun()

    try:
        health = _api_get("/health")
        st.success("Servicio OK")
        st.json(health)
    except Exception as exc:
        st.error(f"No se pudo contactar el servicio: {exc}")
        st.stop()

    try:
        info = _api_get("/info")
        st.subheader("Modelo cargado")
        st.json(info)
    except Exception as exc:
        st.warning(f"/info no disponible: {exc}")

    st.divider()
    st.subheader("Balanceo de carga")
    st.write(
        "Cada predicción devuelve el **pod** que la atendió. "
        "Haz varias predicciones y verás cómo se reparten entre réplicas."
    )
    if "pods_seen" not in st.session_state:
        st.session_state["pods_seen"] = Counter()

    if st.session_state["pods_seen"]:
        st.write("Pods que han atendido:")
        for pod, count in st.session_state["pods_seen"].most_common():
            st.write(f"- `{pod}`: {count}")
        if st.button("Resetear contador"):
            st.session_state["pods_seen"] = Counter()
            st.rerun()

# ---------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------
st.header("Perfil del cliente")

with st.form("predict_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("age", min_value=18, max_value=100, value=35)
        job = st.selectbox("job", JOB_OPTIONS, index=JOB_OPTIONS.index("management"))
        marital = st.selectbox("marital", MARITAL_OPTIONS, index=MARITAL_OPTIONS.index("married"))
        education = st.selectbox("education", EDUCATION_OPTIONS, index=EDUCATION_OPTIONS.index("tertiary"))
        default = st.selectbox("default", YES_NO, index=YES_NO.index("no"))
        balance = st.number_input("balance", min_value=-100000, max_value=1000000, value=1500)

    with col2:
        housing = st.selectbox("housing", YES_NO, index=YES_NO.index("yes"))
        loan = st.selectbox("loan", YES_NO, index=YES_NO.index("no"))
        contact = st.selectbox("contact", CONTACT_OPTIONS, index=CONTACT_OPTIONS.index("cellular"))
        day = st.number_input("day", min_value=1, max_value=31, value=15)
        month = st.selectbox("month", MONTH_OPTIONS, index=MONTH_OPTIONS.index("may"))
        duration = st.number_input("duration (seg)", min_value=0, max_value=10000, value=200)

    with col3:
        campaign = st.number_input("campaign", min_value=1, max_value=100, value=1)
        pdays = st.number_input("pdays", min_value=-1, max_value=1000, value=-1)
        previous = st.number_input("previous", min_value=0, max_value=100, value=0)
        poutcome = st.selectbox("poutcome", POUTCOME_OPTIONS, index=POUTCOME_OPTIONS.index("unknown"))

    submitted = st.form_submit_button("Predecir", type="primary")

if submitted:
    payload = {
        "age": int(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": int(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "day": int(day),
        "month": month,
        "duration": int(duration),
        "campaign": int(campaign),
        "pdays": int(pdays),
        "previous": int(previous),
        "poutcome": poutcome,
    }
    try:
        result = _api_predict(payload)
    except Exception as exc:
        st.error(f"Error al predecir: {exc}")
    else:
        pod = result.get("pod", "desconocido")
        st.session_state["pods_seen"][pod] += 1

        st.subheader("Resultado")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicción", result.get("prediction_label", "?"))
        c2.metric("Probabilidad (yes)", f"{result.get('probability_yes', 0):.4f}")
        c3.metric("Versión del modelo", str(result.get("model_version", "?")))

        if result.get("prediction") == 1:
            st.success(f"El cliente **probablemente contratará** el depósito. Atendido por `{pod}`.")
        else:
            st.info(f"El cliente **probablemente no contratará** el depósito. Atendido por `{pod}`.")

        st.caption("Payload enviado")
        st.json(payload)

st.divider()
st.caption(
    "Tip: pulsa **Predecir** varias veces y observa en la barra lateral cómo el campo `pod` "
    "cambia entre réplicas. Eso demuestra el balanceo de carga del Service de Kubernetes."
)