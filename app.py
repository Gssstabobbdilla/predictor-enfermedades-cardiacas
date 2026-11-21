import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Estimación de Enfermedad Cardíaca",
    layout="centered",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "pipe_logreg_no_reg.joblib"
META_PATH = BASE_DIR / "modelo" / "metadata_logreg.json"


@st.cache_resource
def cargar_recursos():
    modelo = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "No se encontraron `modelo/pipe_logreg_no_reg.joblib` o `modelo/metadata_logreg.json`. "
        "Ejecuta primero la celda que guarda el pipeline y los metadatos."
    )
    st.stop()

modelo, metadata = cargar_recursos()
feature_names = metadata["feature_names"]
dtypes = metadata.get("dtypes", {})

st.title("Prototipo de Estimación — Regresión Logística")
st.caption(
    "Aplicación de apoyo estadístico/académico basada en el dataset de enfermedad cardíaca. "
    "No constituye un diagnóstico médico."
)

st.subheader("Datos clínicos")

entradas = {}
for col in feature_names:
    if col == "age":
        entradas[col] = st.number_input("Edad (`age`)", min_value=18, max_value=110, value=55, step=1)
    elif col == "sex":
        entradas[col] = st.selectbox(
            "Sexo (`sex`)",
            options=[0, 1],
            index=1,
            format_func=lambda x: "0 - Femenino" if x == 0 else "1 - Masculino",
        )
    elif col == "cp":
        entradas[col] = st.selectbox("Tipo de dolor torácico (`cp`)", options=[0, 1, 2, 3], index=0)
    elif col == "trestbps":
        entradas[col] = st.number_input(
            "Presión arterial en reposo (`trestbps`)", min_value=60, max_value=260, value=130, step=1
        )
    elif col == "chol":
        entradas[col] = st.number_input(
            "Colesterol sérico (`chol`)", min_value=80, max_value=700, value=240, step=1
        )
    elif col == "fbs":
        entradas[col] = st.selectbox(
            "Glucemia en ayunas > 120 mg/dl (`fbs`)",
            options=[0, 1],
            index=0,
            format_func=lambda x: "0 - No" if x == 0 else "1 - Sí",
        )
    elif col == "restecg":
        entradas[col] = st.selectbox("ECG en reposo (`restecg`)", options=[0, 1, 2], index=0)
    elif col == "thalach":
        entradas[col] = st.number_input(
            "Frecuencia cardíaca máxima (`thalach`)", min_value=40, max_value=250, value=150, step=1
        )
    elif col == "exang":
        entradas[col] = st.selectbox(
            "Angina inducida por ejercicio (`exang`)",
            options=[0, 1],
            index=0,
            format_func=lambda x: "0 - No" if x == 0 else "1 - Sí",
        )
    elif col == "oldpeak":
        entradas[col] = st.number_input(
            "Depresión del ST (`oldpeak`)", min_value=0.0, max_value=10.0, value=1.0, step=0.1, format="%.1f"
        )
    elif col == "slope":
        entradas[col] = st.selectbox("Pendiente ST (`slope`)", options=[0, 1, 2], index=1)
    elif col == "ca":
        entradas[col] = st.selectbox("Vasos principales (`ca`)", options=[0, 1, 2, 3, 4], index=0)
    elif col == "thal":
        entradas[col] = st.selectbox("Talasemia (`thal`)", options=[0, 1, 2, 3], index=2)
    else:
        entradas[col] = st.number_input(col, value=0)

df_entrada = pd.DataFrame([entradas])[feature_names]
for col, dtype_str in dtypes.items():
    if col in df_entrada.columns:
        try:
            df_entrada[col] = df_entrada[col].astype(dtype_str)
        except (ValueError, TypeError):
            pass

if st.button("Predecir", type="primary"):
    pred = int(modelo.predict(df_entrada)[0])
    proba = float(modelo.predict_proba(df_entrada)[0, 1])

    st.divider()
    c1, c2 = st.columns(2)
    c1.metric("Probabilidad (clase 1)", f"{proba:.2%}")
    c2.metric("Clase predicha", f"{pred}")