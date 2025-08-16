# Telco Churn – Proyecto End-to-End (EDA ➜ Modelo ➜ App ➜ Deploy temporal)

Este repositorio muestra un caso completo de **predicción de churn en telecomunicaciones**:
- **Dataset** (sintético/limpio): comportamiento de clientes, uso del servicio y bajas.
- **Notebook de Colab**: EDA, preprocesamiento, modelado (LogisticRegression/RandomForest), métricas y serialización.
- **App Streamlit (`app.py`)**: formulario con las mismas features del entrenamiento, probabilidad y decisión con **umbral ajustable**, y recomendaciones de negocio.
- **Deploy temporal con ngrok**: publicación rápida desde Colab.

## 1. Problema de negocio
Reducir el **churn** prediciendo qué clientes tienen mayor probabilidad de darse de baja y habilitar acciones de retención (ofertas, mejoras de servicio, etc.).  
**KPI**: F1 / AUC (clasificación binaria), tasa de retención.

## 2. Datos
- ~1.5k filas, ~19 columnas (numéricas, categóricas, fechas y texto corto `nps_text`).
- Imperfecciones controladas: nulos, outliers, categorías raras.
- Target balanceado (≈40–60%).
- Features derivadas: `days_since_signup`, `days_since_last_interaction`.

> Si no incluyes el CSV aquí, el notebook puede **generarlo** o bien cargar desde un archivo subido en Colab.

## 3. Arquitectura (alto nivel)

```mermaid
flowchart LR
    %% Estilos por color
    classDef data fill=#4DB6AC,stroke=#00695C,color=white;
    classDef process fill=#64B5F6,stroke=#1565C0,color=white;
    classDef model fill=#81C784,stroke=#2E7D32,color=white;
    classDef app fill=#FFD54F,stroke=#F57F17,color=black;
    classDef user fill=#E57373,stroke=#C62828,color=white;

    subgraph PREP[Preprocesamiento & Validación]
        A[📂 CSV / Datos]:::data --> B[🔎 Validación & EDA]:::process
        B --> C[⚙️ Preprocesamiento num/cat/txt]:::process
    end

    subgraph TRAIN[Entrenamiento & Modelo]
        C --> D[🤖 Entrenamiento CV + métricas]:::model
        D --> E[💾 Serialización model.joblib]:::model
    end

    subgraph DEPLOY[Despliegue]
        E --> F[🌐 Streamlit app.py]:::app
        F --> G[🖥️ Usuario (via ngrok/web)]:::user
    end



