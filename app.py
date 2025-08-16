import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# -------------------------------------------------
# 1) Funciones necesarias para cargar el modelo
#    (esta función existía cuando entrenaste el pipeline)
# -------------------------------------------------
def select_text_col(X: pd.DataFrame) -> pd.Series:
    # Mismo nombre/función que usaste al entrenar
    # Devuelve la columna de texto 'nps_text' con nulos como cadena vacía
    return X["nps_text"].fillna("")

# -------------------------------------------------
# 2) Cargar el pipeline entrenado
# -------------------------------------------------
@st.cache_resource
def load_model(path="model.joblib"):
    # Gracias a que definimos select_text_col arriba,
    # joblib puede deserializar el pipeline sin error.
    return joblib.load(path)

pipe = load_model()

# -------------------------------------------------
# 3) UI de Streamlit
#    Coincide con las FEATURES usadas al entrenar
#    (numéricas, categóricas y texto)
# -------------------------------------------------
st.set_page_config(page_title="Churn Telco", page_icon="📶", layout="centered")
st.title("📶 Predicción de Churn (Telco)")

st.caption("Modelo cargado desde **model.joblib**")

with st.form("form_inputs"):
    st.subheader("Datos del cliente")

    # Fechas para calcular 'days_since_*'
    signup_date = st.date_input("signup_date", value=date(2023, 1, 15))
    last_interaction_date = st.date_input("last_interaction_date", value=date.today())

    # Numéricas
    tenure_months = st.number_input("tenure_months", min_value=0, max_value=240, value=24, step=1)
    monthly_charge = st.number_input("monthly_charge", min_value=0.0, value=65.0, step=0.1)
    total_charges = st.number_input("total_charges", min_value=0.0, value=1560.0, step=0.1)
    support_tickets_30d = st.number_input("support_tickets_30d", min_value=0, max_value=30, value=1, step=1)
    num_services = st.number_input("num_services", min_value=1, max_value=5, value=3, step=1)
    avg_download_mbps = st.number_input("avg_download_mbps", min_value=1.0, value=180.0, step=1.0)
    downtime_hrs_30d = st.number_input("downtime_hrs_30d", min_value=0.0, value=1.5, step=0.1)
    late_payments_12m = st.number_input("late_payments_12m", min_value=0, max_value=24, value=0, step=1)

    # Categóricas (valores ejemplo; usa los que tengas en tu dataset)
    contract_type = st.selectbox("contract_type", ["Mes a mes", "1 año", "2 años"], index=0)
    payment_method = st.selectbox("payment_method", ["Tarjeta", "Débito", "Efectivo", "Billetera", "Cheque"], index=0)
    internet_service = st.selectbox("internet_service", ["Fibra", "Cable", "DSL", "Satélite"], index=0)
    promo_applied = st.selectbox("promo_applied", ["Sí", "No"], index=1)
    region = st.selectbox("region", ["Norte", "Centro", "Sur", "Oriente", "Lima Metropolitana"], index=4)
    device_type = st.selectbox("device_type", ["Modem", "Router", "Combo", "ONT", "Otro"], index=1)

    # Texto
    nps_text = st.text_input("nps_text (comentario breve)", "Todo bien")

    # Umbral
    threshold = st.slider("Umbral de decisión (churn si prob ≥ umbral)",
                          0.05, 0.95, 0.50, 0.01)

    submitted = st.form_submit_button("Predecir")

# -------------------------------------------------
# 4) Construcción de features como en el entrenamiento
#    (incluye derivadas days_since_signup / days_since_last_interaction)
# -------------------------------------------------
def build_features() -> pd.DataFrame:
    REFDATE = pd.Timestamp.today().normalize()
    s = pd.to_datetime(str(signup_date))
    li = pd.to_datetime(str(last_interaction_date))
    data = {
        # Numéricas originales
        "tenure_months": tenure_months,
        "monthly_charge": monthly_charge,
        "total_charges": total_charges,
        "support_tickets_30d": support_tickets_30d,
        "num_services": num_services,
        "avg_download_mbps": avg_download_mbps,
        "downtime_hrs_30d": downtime_hrs_30d,
        "late_payments_12m": late_payments_12m,
        # Derivadas de fecha
        "days_since_signup": (REFDATE - s).days if pd.notna(s) else None,
        "days_since_last_interaction": (REFDATE - li).days if pd.notna(li) else None,
        # Categóricas
        "contract_type": contract_type,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "promo_applied": promo_applied,
        "region": region,
        "device_type": device_type,
        # Texto
        "nps_text": nps_text
    }
    return pd.DataFrame([data])

# -------------------------------------------------
# 5) Predicción
# -------------------------------------------------
if submitted:
    X_infer = build_features()
    proba = float(pipe.predict_proba(X_infer)[:, 1][0])
    pred = int(proba >= threshold)

    st.markdown("### Resultado")
    st.write(f"**Probabilidad de churn:** {proba:.3f}")
    st.write(f"**Predicción (umbral {threshold:.2f}):** {'Churn' if pred==1 else 'No churn'}")

    st.markdown("### Recomendaciones")
    recs = []
    if pred == 1:
        recs.append("- Ofrecer contrato de 1–2 años con beneficios (upgrade, descuento 3–6 meses).")
        recs.append("- Reducir downtime y TMR de soporte; ticket preventivo si hubo caídas.")
        recs.append("- Ajustar tarifa/paquete (bundle) y promociones personalizadas.")
        if late_payments_12m > 0:
            recs.append("- Recordatorios y fraccionamiento; incentivar pago automático.")
        if avg_download_mbps < 80:
            recs.append("- Proponer upgrade de plan/tecnología.")
    else:
        recs.append("- Cliente estable: habilitar cross-sell suave (upgrade de velocidad).")
        recs.append("- Mantener NPS con comunicaciones proactivas y estabilidad del servicio.")

    for r in recs:
        st.write(r)

st.caption("© Telco Churn Demo – Streamlit")
