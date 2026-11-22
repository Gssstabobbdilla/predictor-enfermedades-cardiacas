from datetime import datetime
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sistema de Estimación Clínica Cardiovascular",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
        .hero-banner {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #172554 100%);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 1.5rem 1.8rem;
            margin-bottom: 1.1rem;
        }
        .hero-title {
            margin: 0 0 0.35rem 0;
            font-size: 1.7rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .hero-subtitle {
            margin: 0;
            font-size: 0.96rem;
            color: #cbd5e1;
            line-height: 1.45;
        }
        .notice-box {
            background-color: rgba(245, 158, 11, 0.10);
            border-left: 4px solid #f59e0b;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 1.25rem;
            font-size: 0.9rem;
        }
        .section-tag {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #60a5fa;
            margin-bottom: 0.2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modelo" / "pipe_logreg_no_reg.joblib"
META_PATH = BASE_DIR / "modelo" / "metadata_logreg.json"

LABELS_VARIABLES = {
    "age": "Edad (años)",
    "sex": "Sexo biológico",
    "cp": "Tipo de dolor torácico (cp)",
    "trestbps": "Presión arterial en reposo (mm Hg)",
    "chol": "Colesterol sérico (mg/dl)",
    "fbs": "Glucemia en ayunas > 120 mg/dl",
    "restecg": "Electrocardiograma en reposo (restecg)",
    "thalach": "Frecuencia cardíaca máxima alcanzada (lpm)",
    "exang": "Angina inducida por ejercicio",
    "oldpeak": "Depresión del segmento ST (oldpeak)",
    "slope": "Pendiente del segmento ST en ejercicio",
    "ca": "Número de vasos principales coloreados (0–4)",
    "thal": "Resultado de talasemia (thal)",
}

DESCRIPCIONES_CATEGORIAS = {
    "sex": {0: "Femenino", 1: "Masculino"},
    "cp": {
        0: "Angina típica",
        1: "Angina atípica",
        2: "Dolor no anginoso",
        3: "Asintomático",
    },
    "fbs": {0: "No (≤ 120 mg/dl)", 1: "Sí (> 120 mg/dl)"},
    "restecg": {
        0: "Normal",
        1: "Anomalía de onda ST-T",
        2: "Hipertrofia ventricular izquierda probable/definida",
    },
    "exang": {0: "No", 1: "Sí"},
    "slope": {0: "Ascendente", 1: "Plana", 2: "Descendente"},
    "thal": {
        0: "Normal / Nulo",
        1: "Defecto fijo",
        2: "Defecto reversible",
        3: "Otro / Sin especificar",
    },
}


@st.cache_resource
def cargar_recursos():
    modelo = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return modelo, metadata


def formatear_valor_clinico(col, val):
    if col in DESCRIPCIONES_CATEGORIAS and int(val) in DESCRIPCIONES_CATEGORIAS[col]:
        return f"{int(val)} ({DESCRIPCIONES_CATEGORIAS[col][int(val)]})"
    if col == "oldpeak":
        return f"{float(val):.1f}"
    return f"{val}"


def calcular_factores_influyentes(pipeline, df_entrada, feature_names):
    """
    Calcula la contribución local de cada variable al log-odds (z_i * beta_i)
    usando el escalador y el estimador lineal del Pipeline entrenado.
    """
    if not hasattr(pipeline, "named_steps") or len(pipeline.named_steps) < 2:
        return pd.DataFrame()

    pasos = list(pipeline.named_steps.values())
    transformador = pasos[0]
    clasificador = pasos[-1]

    if not hasattr(clasificador, "coef_"):
        return pd.DataFrame()

    X_proc = transformador.transform(df_entrada)
    z_vals = np.asarray(X_proc)[0]
    betas = np.asarray(clasificador.coef_)[0]
    contribuciones = z_vals * betas

    filas = []
    for i, col in enumerate(feature_names):
        val_raw = df_entrada.iloc[0][col]
        contrib = float(contribuciones[i])
        filas.append(
            {
                "variable": col,
                "nombre_clinico": LABELS_VARIABLES.get(col, col),
                "valor_registrado": formatear_valor_clinico(col, val_raw),
                "contribucion_log_odds": contrib,
                "impacto_abs": abs(contrib),
                "efecto": (
                    "Incrementa probabilidad"
                    if contrib > 0.02
                    else ("Reduce probabilidad" if contrib < -0.02 else "Efecto neutro")
                ),
            }
        )

    df_factores = pd.DataFrame(filas).sort_values(by="impacto_abs", ascending=False).reset_index(drop=True)
    return df_factores


def construir_reporte_clinico(
    fecha_str,
    paciente_id,
    proba,
    clasificacion_txt,
    pred_clase,
    umbral,
    df_entrada,
    feature_names,
    df_factores,
):
    top_factores = df_factores.head(5) if not df_factores.empty else pd.DataFrame()

    lineas_factores = []
    if not top_factores.empty:
        for idx, row in top_factores.iterrows():
            signo = "+" if row["contribucion_log_odds"] >= 0 else ""
            lineas_factores.append(
                f"  {idx + 1}. {row['nombre_clinico']} [{row['variable']} = {row['valor_registrado']}]: "
                f"{row['efecto']} (aporte al logit: {signo}{row['contribucion_log_odds']:.3f})"
            )
    else:
        lineas_factores.append("  No se pudieron extraer los coeficientes del modelo.")

    lineas_historia = []
    for col in feature_names:
        nombre = LABELS_VARIABLES.get(col, col)
        val_fmt = formatear_valor_clinico(col, df_entrada.iloc[0][col])
        lineas_historia.append(f"  - {nombre} ({col}): {val_fmt}")

    return f"""====================================================================
REPORTE CLÍNICO ORIENTATIVO — EVALUACIÓN DE RIESGO CARDIOVASCULAR
====================================================================
Fecha y hora de emisión : {fecha_str}
Identificador / Caso    : {paciente_id if paciente_id.strip() else "No especificado"}
Modelo estadístico      : Regresión Logística (Pipeline + StandardScaler)

1. RESULTADO DE LA ESTIMACIÓN
--------------------------------------------------------------------
- Probabilidad estimada (Clase 1) : {proba:.2%}
- Clasificación resultante        : {clasificacion_txt} (Clase {pred_clase}, umbral {umbral:.2f})

2. FACTORES QUE MÁS INFLUYEN EN ESTA ESTIMACIÓN
--------------------------------------------------------------------
{chr(10).join(lineas_factores)}

3. RESUMEN DE HISTORIA CLÍNICA REGISTRADA (VARIABLES 2.1)
--------------------------------------------------------------------
{chr(10).join(lineas_historia)}

4. NOTA DE ALCANCE Y RESPONSABILIDAD
--------------------------------------------------------------------
Este reporte ha sido generado automáticamente por un prototipo de
aprendizaje estadístico con fines académicos y de apoyo exploratorio.
No constituye un diagnóstico médico definitivo ni reemplaza el criterio
ni la evaluación integral de un profesional de la salud.
====================================================================
"""


if not MODEL_PATH.exists() or not META_PATH.exists():
    st.error(
        "No se encontraron `modelo/pipe_logreg_no_reg.joblib` o `modelo/metadata_logreg.json`. "
        "Verifica que ambos archivos existan dentro de la carpeta `modelo/`."
    )
    st.stop()

modelo, metadata = cargar_recursos()
feature_names = metadata["feature_names"]
dtypes = metadata.get("dtypes", {})

with st.sidebar:
    st.header("Configuración del Sistema")
    st.markdown("**Modelo activo:** Regresión Logística (`penalty=None`)")
    st.markdown("**Preprocesamiento:** `StandardScaler` integrado")
    st.markdown(f"**Variables clínicas (2.1):** {len(feature_names)}")
    st.divider()
    umbral = st.slider(
        "Umbral de decisión (Clase 1)",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Punto de corte sobre la probabilidad estimada para asignar Clase 1.",
    )
    st.caption(
        "RF3: Validación de esquema y tipos.\n"
        "RF4: Registro estructurado y visualización.\n"
        "RF5: Reporte clínico explicativo en texto."
    )

st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">Sistema de Estimación Clínica Cardiovascular</p>
        <p class="hero-subtitle">
            Evaluación de historia clínica mediante Regresión Logística estandarizada,
            visualización de probabilidad estimada y generación de reporte clínico explicativo.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="notice-box">
        <strong>Aviso importante:</strong> Herramienta de apoyo estadístico y académico.
        Los resultados y el reporte generado son orientativos y no sustituyen un diagnóstico médico profesional.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-tag">RF3 & RF4 · Registro de Historia Clínica</div>', unsafe_allow_html=True)
st.subheader("Formulario de Datos Clínicos del Paciente")

with st.form("form_historia_clinica"):
    paciente_id = st.text_input(
        "Identificador del caso / paciente (opcional)",
        placeholder="Ej. PAC-001",
    )

    tab_demo, tab_lab, tab_ecg = st.tabs(
        [
            "1. Demografía y Síntomas",
            "2. Signos Vitales y Laboratorio",
            "3. Electrocardiografía y Esfuerzo",
        ]
    )

    entradas = {}

    with tab_demo:
        c1, c2 = st.columns(2)
        with c1:
            entradas["age"] = st.number_input(
                LABELS_VARIABLES["age"], min_value=18, max_value=110, value=55, step=1
            )
            entradas["cp"] = st.selectbox(
                LABELS_VARIABLES["cp"],
                options=[0, 1, 2, 3],
                index=0,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['cp'][x]}",
            )
        with c2:
            entradas["sex"] = st.selectbox(
                LABELS_VARIABLES["sex"],
                options=[0, 1],
                index=1,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['sex'][x]}",
            )
            entradas["exang"] = st.selectbox(
                LABELS_VARIABLES["exang"],
                options=[0, 1],
                index=0,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['exang'][x]}",
            )

    with tab_lab:
        c1, c2 = st.columns(2)
        with c1:
            entradas["trestbps"] = st.number_input(
                LABELS_VARIABLES["trestbps"], min_value=60, max_value=260, value=130, step=1
            )
            entradas["fbs"] = st.selectbox(
                LABELS_VARIABLES["fbs"],
                options=[0, 1],
                index=0,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['fbs'][x]}",
            )
        with c2:
            entradas["chol"] = st.number_input(
                LABELS_VARIABLES["chol"], min_value=80, max_value=700, value=240, step=1
            )
            entradas["thal"] = st.selectbox(
                LABELS_VARIABLES["thal"],
                options=[0, 1, 2, 3],
                index=2,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['thal'][x]}",
            )

    with tab_ecg:
        c1, c2 = st.columns(2)
        with c1:
            entradas["restecg"] = st.selectbox(
                LABELS_VARIABLES["restecg"],
                options=[0, 1, 2],
                index=0,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['restecg'][x]}",
            )
            entradas["thalach"] = st.number_input(
                LABELS_VARIABLES["thalach"], min_value=40, max_value=250, value=150, step=1
            )
            entradas["ca"] = st.selectbox(
                LABELS_VARIABLES["ca"],
                options=[0, 1, 2, 3, 4],
                index=0,
            )
        with c2:
            entradas["oldpeak"] = st.number_input(
                LABELS_VARIABLES["oldpeak"],
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                step=0.1,
                format="%.1f",
            )
            entradas["slope"] = st.selectbox(
                LABELS_VARIABLES["slope"],
                options=[0, 1, 2],
                index=1,
                format_func=lambda x: f"{x} - {DESCRIPCIONES_CATEGORIAS['slope'][x]}",
            )

    submitted = st.form_submit_button(
        "Analizar historia clínica y generar reporte",
        type="primary",
        use_container_width=True,
    )

if submitted:
    faltantes = [col for col in feature_names if col not in entradas]
    if faltantes:
        st.error(f"Faltan variables requeridas por el modelo: {', '.join(faltantes)}")
        st.stop()

    df_entrada = pd.DataFrame([entradas])[feature_names]
    for col, dtype_str in dtypes.items():
        if col in df_entrada.columns:
            try:
                df_entrada[col] = df_entrada[col].astype(dtype_str)
            except (ValueError, TypeError):
                pass

    proba = float(modelo.predict_proba(df_entrada)[0, 1])
    pred_clase = int(proba >= umbral)

    if pred_clase == 1:
        clasificacion_txt = "Positivo estimado (Presencia de enfermedad cardíaca)"
        estado_color = "error"
    else:
        clasificacion_txt = "Negativo estimado (Ausencia de enfermedad cardíaca)"
        estado_color = "success"

    df_factores = calcular_factores_influyentes(modelo, df_entrada, feature_names)
    fecha_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    reporte_texto = construir_reporte_clinico(
        fecha_str=fecha_str,
        paciente_id=paciente_id,
        proba=proba,
        clasificacion_txt=clasificacion_txt,
        pred_clase=pred_clase,
        umbral=umbral,
        df_entrada=df_entrada,
        feature_names=feature_names,
        df_factores=df_factores,
    )

    st.divider()
    st.markdown('<div class="section-tag">RF4 · Resultado de la Predicción</div>', unsafe_allow_html=True)
    st.subheader("Resultado de la Evaluación")

    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilidad estimada (Clase 1)", f"{proba:.2%}")
    m2.metric("Clasificación (Clase)", f"{pred_clase}")
    m3.metric("Umbral aplicado", f"{umbral:.2f}")

    st.progress(min(max(proba, 0.0), 1.0))

    if estado_color == "error":
        st.error(f"**Clasificación:** {clasificacion_txt} — Probabilidad estimada: **{proba:.2%}**")
    else:
        st.success(f"**Clasificación:** {clasificacion_txt} — Probabilidad estimada: **{proba:.2%}**")

    if not df_factores.empty:
        st.markdown("#### Factores clínicos con mayor influencia en este caso")
        top5 = df_factores.head(5)[
            ["nombre_clinico", "valor_registrado", "efecto", "contribucion_log_odds"]
        ].rename(
            columns={
                "nombre_clinico": "Variable clínica",
                "valor_registrado": "Valor registrado",
                "efecto": "Dirección del efecto",
                "contribucion_log_odds": "Contribución (log-odds)",
            }
        )
        st.dataframe(top5, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown('<div class="section-tag">RF5 · Reporte Clínico en Texto</div>', unsafe_allow_html=True)
    st.subheader("Reporte Clínico Generado")

    st.text_area(
        "Vista previa del reporte en texto",
        value=reporte_texto,
        height=340,
    )

    nombre_archivo = (
        f"reporte_clinico_{paciente_id.strip()}.txt"
        if paciente_id.strip()
        else "reporte_clinico_cardiovascular.txt"
    )
    st.download_button(
        label="Descargar reporte clínico (.txt)",
        data=reporte_texto.encode("utf-8"),
        file_name=nombre_archivo,
        mime="text/plain",
        use_container_width=True,
    )