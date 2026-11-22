# Predictor de Enfermedades Cardíacas

**Predicción de la presencia de enfermedad cardíaca a partir de factores clínicos mediante técnicas de aprendizaje estadístico.**

Proyecto del curso **Aprendizaje Estadístico** · Programa de Ingeniería de Sistemas e Inteligencia Artificial · Facultad de Ingeniería · Universidad Privada Antenor Orrego (UPAO) · Trujillo, Perú · 2026-20

---

## Tabla de contenidos

1. [Descripción](#-descripción)
2. [Objetivos](#-objetivos)
3. [Dataset](#-dataset)
4. [Metodología](#-metodología)
5. [Modelos evaluados](#-modelos-evaluados)
6. [Resultados](#-resultados)
7. [Modelo seleccionado](#-modelo-seleccionado)
8. [Instalación y ejecución](#-instalación-y-ejecución)
9. [Uso del sistema](#-uso-del-sistema)
10. [Requerimientos del sistema](#-requerimientos-del-sistema)
11. [Autores](#-autores)
12. [Referencias](#-referencias)

---

## Descripción

Las enfermedades cardiovasculares son una de las principales causas de muerte en el mundo: según la OMS (2025), en 2022 se estimaron cerca de 19.8 millones de muertes por esta causa (≈ 32 % de las defunciones globales).

Este proyecto aborda el problema como una **clasificación binaria supervisada**: dadas 13 características clínicas y demográficas de un paciente, estimar si presenta (`target = 1`) o no (`target = 0`) enfermedad cardíaca. Más allá del rendimiento predictivo, el trabajo pone énfasis en:

- Validación rigurosa **sin fuga de información** (*data leakage*).
- Comparación de modelos con distinto nivel de flexibilidad (lineales regularizados, ensambles y SVM).
- **Interpretabilidad** de los factores que influyen en la predicción.
- Cuantificación de la incertidumbre (intervalos de confianza *bootstrap*).

El sistema integra: carga y depuración de datos → preprocesamiento → modelo de clasificación → módulo de entrada que valida los datos de un paciente, calcula la probabilidad de enfermedad y genera un **reporte clínico en texto**.

> **Aviso:** este proyecto tiene fines académicos. **No es un dispositivo médico** ni sustituye el diagnóstico de un profesional de la salud.

## Objetivos

**General:** desarrollar y evaluar modelos de aprendizaje estadístico para clasificar la presencia o ausencia de enfermedad cardíaca, comparando su capacidad predictiva, generalización e interpretabilidad.

**Específicos:**

1. Preparar los datos para aprendizaje supervisado (duplicados, valores faltantes, estandarización dentro de la validación, sin fuga de información).
2. Analizar la relación entre las características clínicas y el diagnóstico, identificando variables informativas y redundancias.
3. Construir modelos con distintos niveles de flexibilidad (regresión logística y sus versiones regularizadas, Random Forest, SVM, XGBoost).
4. Implementar un sistema de clasificación con **precisión > 80 %** y **sensibilidad > 80 %** en el conjunto de prueba, manteniendo una baja tasa de falsos negativos.

## Dataset

- **Fuente:** [Heart Disease Dataset (Kaggle)](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset), basado en el repositorio [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).
- **Tamaño original:** 1 025 filas × 14 columnas.
- **Tras `drop_duplicates()`:** **302 registros únicos** (se eliminaron 723 duplicados, 70.5 %).
- **Valores nulos:** ninguno.
- **Balance de clases:** 164 con enfermedad (54.3 %) y 138 sin enfermedad (45.7 %); no se aplicó sobre/submuestreo.

| Variable | Descripción | Valores / Rango |
|---|---|---|
| `age` | Edad (años) | 29 – 77 |
| `sex` | Sexo | 0 = mujer, 1 = hombre |
| `cp` | Tipo de dolor torácico | 0 = angina típica, 1 = atípica, 2 = no anginoso, 3 = asintomático |
| `trestbps` | Presión arterial en reposo (mm Hg) | 94 – 200 |
| `chol` | Colesterol sérico (mg/dl) | 126 – 564 |
| `fbs` | Glucosa en ayunas > 120 mg/dl | 0 / 1 |
| `restecg` | ECG en reposo | 0 = normal, 1 = onda ST-T anormal, 2 = hipertrofia ventricular |
| `thalach` | Frecuencia cardíaca máxima (bpm) | 71 – 202 |
| `exang` | Angina inducida por ejercicio | 0 / 1 |
| `oldpeak` | Depresión del segmento ST inducida por ejercicio | 0.0 – 6.2 |
| `slope` | Pendiente del segmento ST en el pico del ejercicio | 0, 1, 2 |
| `ca` | Vasos principales coloreados por fluoroscopia | 0 – 4 |
| `thal` | Resultado de la prueba de talasemia | 0, 1, 2, 3 |
| **`target`** | **Variable objetivo** | 0 = sin enfermedad, 1 = con enfermedad |

## Metodología

### Preprocesamiento
- Eliminación de duplicados exactos y verificación de integridad (`df.info()`, `df.isnull().sum()`).
- Estandarización (`StandardScaler`) de las variables continuas: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`.
- El escalador va **dentro de un `Pipeline` de scikit-learn** y se ajusta solo con los datos de entrenamiento de cada pliegue, evitando fuga de información.

### Particionado y validación

| Conjunto | Tamaño | Rol |
|---|---|---|
| Entrenamiento | 241 obs. (80 %) | Ajuste de modelos y búsqueda de hiperparámetros |
| Validación cruzada (5 folds, estratificada) | ~48–49 obs. por fold | Selección de hiperparámetros (`GridSearchCV`, métrica F1) |
| Validación cruzada repetida 5 × 5 | 25 evaluaciones por modelo | Comparación exploratoria de familias de modelos |
| Validación cruzada anidada (5 externos × 5 internos) | — | **Estimación principal** del desempeño |
| Prueba | 61 obs. (20 %: 33 enfermos, 28 sanos) | Evaluación final, usada **una sola vez** |

- Partición con `train_test_split(stratify=y, random_state=42)`.
- El conjunto de prueba no participó en ningún ajuste ni selección; se usó el umbral de decisión por defecto (0.5).
- Intervalos de confianza al 95 % mediante *bootstrap* (1 000 remuestreos) sobre el conjunto de prueba.

### Criterio de selección del modelo (definido antes de evaluar en prueba)
1. Mayor F1 medio en validación anidada.
2. Criterio clínico secundario: sensibilidad (el falso negativo es el error más costoso).
3. Equivalencia: modelos a menos de una desviación estándar del mejor se consideran equivalentes.
4. Desempate: el modelo **más simple e interpretable**.
5. Verificación pareada con la prueba de Wilcoxon.

## Modelos evaluados

| # | Modelo | Tipo | Propósito |
|---|---|---|---|
| 1 | Regresión logística sin regularización | Lineal / probabilístico | Línea base interpretable |
| 2 | Regresión logística L2 (Ridge) | Lineal regularizado | Estabilizar coeficientes |
| 3 | Regresión logística L1 (Lasso) | Lineal con selección de variables | Evaluar anulación de variables |
| 4 | Random Forest | Ensamble / bagging | Reducir varianza, capturar interacciones |
| 5 | XGBoost | Ensamble / boosting | Ajuste secuencial regularizado |
| 6 | SVM (kernel RBF) | Margen máximo / kernel | Fronteras no lineales |

**Hiperparámetros óptimos (GridSearchCV):**

- Regresión logística L1/L2: `C = 4.83`
- Random Forest: `max_depth=3, min_samples_leaf=3, max_features='sqrt', n_estimators=200`
- XGBoost: `max_depth=2, learning_rate=0.03, n_estimators=300, subsample=1.0, colsample_bytree=0.8, reg_lambda=1`
- SVM (RBF): `C=100, gamma=0.003, probability=True`

## Resultados

### Validación cruzada anidada (estimación principal, F1)

| Modelo | F1 medio | Desv. est. |
|---|---|---|
| Random Forest | 0.870 | 0.074 |
| SVM (RBF) | 0.860 | 0.052 |
| LR sin regularización | 0.859 | 0.064 |
| LR L2 (Ridge) | 0.850 | 0.072 |
| XGBoost | 0.839 | 0.050 |
| LR L1 (Lasso) | 0.838 | 0.062 |

Ninguna diferencia frente a la regresión logística sin regularización resultó estadísticamente significativa (Wilcoxon, p ≥ 0.25). Con solo 5 pliegues externos, el p-valor mínimo alcanzable es ≈ 0.06, por lo que la prueba no puede demostrar igualdad: los seis modelos se consideran **equivalentes** dentro del ruido muestral.

### Conjunto de prueba (N = 61)

| Modelo | Accuracy | Precision | Recall | Especificidad | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| **LR sin regularización** | 0.787 | 0.778 | 0.848 | 0.714 | 0.812 | 0.868 |
| LR L2 (Ridge) | 0.803 | 0.800 | 0.848 | 0.750 | 0.824 | 0.871 |
| LR L1 (Lasso) | 0.803 | 0.800 | 0.848 | 0.750 | 0.824 | 0.870 |
| Random Forest | 0.803 | 0.800 | 0.848 | 0.750 | 0.824 | 0.873 |
| XGBoost | 0.721 | 0.750 | 0.727 | 0.714 | 0.738 | 0.850 |
| SVM (RBF) | 0.820 | 0.806 | 0.879 | 0.750 | 0.841 | 0.870 |

Modelo desplegado. Para la regresión logística sin regularización, los intervalos de confianza al 95 % (bootstrap) son: accuracy 0.672–0.885, precision 0.639–0.903, **recall 0.719–0.968**, especificidad 0.536–0.879, F1 0.704–0.900 y ROC-AUC 0.764–0.949. Los intervalos son amplios debido al tamaño reducido de la prueba (1 paciente ≈ 1.6 puntos porcentuales).


## Modelo seleccionado

**Regresión logística sin regularización.** Se elige por interpretabilidad y sencillez dentro de un grupo de modelos estadísticamente equivalentes, no por una ventaja predictiva:

- **Interpretable:** cada coeficiente indica dirección y magnitud del efecto, lo que permite que el reporte clínico explique los factores que influyen en la predicción.
- **Liviano:** 13 coeficientes + intercepto + escalador; convergió en 23 iteraciones; entrega probabilidades directas por sigmoide.
- **Sin hiperparámetros** y sin el optimismo de la selección sobre los mismos pliegues.

### Coeficientes (variables continuas estandarizadas)

| Variable | Coeficiente | Odds ratio | Efecto |
|---|---|---|---|
| `cp` | +1.021 | 2.776 | ↑ probabilidad de enfermedad |
| `thalach` | +0.925 | 2.521 | ↑ |
| `slope` | +0.544 | 1.724 | ↑ |
| `restecg` | +0.457 | 1.579 | ↑ |
| `age` | +0.129 | 1.137 | ≈ nulo |
| `chol` | −0.327 | 0.721 | ↓ |
| `fbs` | −0.367 | 0.693 | ↓ |
| `trestbps` | −0.406 | 0.666 | ↓ |
| `oldpeak` | −0.408 | 0.665 | ↓ |
| `ca` | −0.758 | 0.469 | ↓ |
| `thal` | −0.893 | 0.409 | ↓ |
| `exang` | −1.129 | 0.323 | ↓ |
| `sex` | −2.076 | 0.125 | ↓ |

> Estas son **asociaciones dentro de la muestra, no relaciones causales**. Random Forest y XGBoost coinciden en que `cp` y `thal` son las variables más usadas, y `fbs` tiene importancia prácticamente nula.


## Instalación y ejecución

```bash
# 1. Clonar el repositorio
git clone https://github.com/Gssstabobbdilla/predictor-enfermedades-cardiacas.git
cd predictor-enfermedades-cardiacas

# 2. (Opcional) Crear entorno virtual
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

Dependencias principales: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `scipy`, `matplotlib`, `seaborn`, `kagglehub`.

**Notebook en Google Colab:** [Abrir notebook](https://colab.research.google.com/drive/1kqBjqKRfV5RpkI1znaIAqdEdaQ1Jcgrz#scrollTo=rtlCjCjw)

> El dataset se descarga con `kagglehub` y se depura con `drop_duplicates()` hasta obtener los 302 registros únicos.

## Uso del sistema

<!-- AJUSTAR: nombres de módulos/archivos según el repositorio -->

El sistema expone dos funciones principales:

- `predecir_paciente(datos)`: valida los datos de un paciente (convierte formatos, completa faltantes con advertencia o rechaza la entrada con un mensaje claro) y devuelve la **probabilidad** de enfermedad y la **clase** predicha.
- `reporte_clinico(...)`: genera un texto con la probabilidad estimada, la clasificación, los tres factores que más aumentan y los tres que más reducen la probabilidad, las advertencias de la entrada y una nota de uso.

Ejemplo ilustrativo:

```python
paciente = {
    "age": 55, "sex": 1, "cp": 0, "trestbps": 140, "chol": 240,
    "fbs": 0, "restecg": 1, "thalach": 150, "exang": 1,
    "oldpeak": 1.5, "slope": 1, "ca": 1, "thal": 2
}

resultado = predecir_paciente(paciente)
print(reporte_clinico(paciente))
```

**Aplicación web:** [Abrir aplicacion]([https://colab.research.google.com/drive/1kqBjqKRfV5RpkI1znaIAqdEdaQ1Jcgrz#scrollTo=rtlCjCjw](https://predictor-enfermedades-cardiacas-hb5oyajz8pk7wltw5nlbhr.streamlit.app/))

## Requerimientos del sistema

| Código | Requerimiento | Estado |
|---|---|---|
| RF1 | Incorporar y procesar las observaciones del dataset de Kaggle | Cumplido |
| RF2 | Módulo de preprocesamiento (duplicados, nulos, estandarización) | Cumplido |
| RF3 | Analizar las variables de la historia clínica | Cumplido |
| RF4 | Interfaz con formulario para registrar datos y ver resultados | Cumplido |
| RF5 | Reporte clínico en texto | Cumplido (notebook) |
| RNF1 | Tiempo de respuesta < 1 minuto | Del orden de milisegundos |
| RNF2 | Entrenamiento y validación que garanticen precisión adecuada | Cumplido |
| RNF3 | Robustez ante datos incompletos o con otros formatos | 11 casos de prueba, 0 excepciones no controladas |


## Autores

Universidad Privada Antenor Orrego · Ingeniería de Sistemas e Inteligencia Artificial

- Alva Bobadilla, Gustavo Paolo
- Chuquiviguel Castillo, Darieé
- Gotera Campos, Manuel Arturo
- Manrique Urteaga, Luana Arelis
- Romero Herrera, Jhoselin Idania
- Sacramento Chigne, Joel Miguel
- Salinas Castrillon, Sebastian Emanuel

## Referencias

- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794. https://doi.org/10.1145/2939672.2939785
- Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *The American Journal of Cardiology, 64*(5), 304–310. https://doi.org/10.1016/0002-9149(89)90524-9
- Lapp, D. (2019). *Heart Disease Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- Organización Mundial de la Salud. (2025). *Cardiovascular diseases (CVDs)*. https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)
- Osei-Nkwantabisa, A. S., & Ntumy, R. (2024). *Classification and prediction of heart diseases using machine learning algorithms*. arXiv. https://doi.org/10.48550/arXiv.2409.03697
- UCI Machine Learning Repository. (1988). *Heart Disease*. https://archive.ics.uci.edu/dataset/45/heart+disease

