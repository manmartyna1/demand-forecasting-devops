import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime

st.set_page_config(
    page_title="Prognozowanie popytu",
    page_icon="📦",
    layout="centered"
)

# Lekki styling
st.markdown("""
    <style>
    .main {
        padding-top: 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        font-size: 16px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }
    </style>
""", unsafe_allow_html=True)

# Wczytanie modelu i preprocessing
model = joblib.load("rf_week.pkl")
encoder = joblib.load("encoder_week.pkl")
scaler = joblib.load("scaler_week.pkl")
feature_cols = joblib.load("feature_cols_week.pkl")

categorical_cols = ["Product_Code", "Warehouse", "Product_Category"]
numeric_cols = [
    "Year", "Month", "WeekOfYear",
    "lag_1", "lag_2", "lag_4",
    "rolling_mean_4", "rolling_std_4"
]

st.title("📦 System prognozowania tygodniowego popytu")
st.write("Wprowadź dane produktu oraz historię popytu, aby uzyskać prognozę na wybrany tydzień.")

st.markdown("## Dane podstawowe")
col1, col2 = st.columns(2)

with col1:
    product_code = st.text_input("Kod produktu", "Product_0979")
    product_category = st.text_input("Kategoria produktu", "Category_028")

with col2:
    warehouse = st.text_input("Magazyn", "Whse_J")

st.markdown("## Dane czasowe")
col3, col4, col5 = st.columns(3)

with col3:
    year = st.number_input("Rok", value=2017, step=1)
with col4:
    month = st.number_input("Miesiąc", value=1, step=1)
with col5:
    week_of_year = st.number_input("Numer tygodnia", value=1, step=1)

st.markdown("## Historia popytu")
col6, col7 = st.columns(2)

with col6:
    lag_1 = st.number_input("Popyt z poprzedniego tygodnia", value=500.0)
    lag_2 = st.number_input("Popyt sprzed 2 tygodni", value=500.0)
    lag_4 = st.number_input("Popyt sprzed 4 tygodni", value=500.0)

with col7:
    rolling_mean_4 = st.number_input("Średni popyt z 4 tygodni", value=500.0)
    rolling_std_4 = st.number_input("Zmienność popytu z 4 tygodni", value=0.0)

if st.button("🔮 Prognozuj"):
    validation_errors = []

    if not product_code.strip():
        validation_errors.append("Pole „Kod produktu” nie może być puste.")
    if not warehouse.strip():
        validation_errors.append("Pole „Magazyn” nie może być puste.")
    if not product_category.strip():
        validation_errors.append("Pole „Kategoria produktu” nie może być puste.")
    if not (1 <= month <= 12):
        validation_errors.append("Miesiąc musi mieć wartość od 1 do 12.")
    if not (1 <= week_of_year <= 53):
        validation_errors.append("Numer tygodnia musi mieć wartość od 1 do 53.")
    if lag_1 < 0:
        validation_errors.append("Popyt z poprzedniego tygodnia nie może być ujemny.")
    if lag_2 < 0:
        validation_errors.append("Popyt sprzed 2 tygodni nie może być ujemny.")
    if lag_4 < 0:
        validation_errors.append("Popyt sprzed 4 tygodni nie może być ujemny.")
    if rolling_mean_4 < 0:
        validation_errors.append("Średni popyt z 4 tygodni nie może być ujemny.")
    if rolling_std_4 < 0:
        validation_errors.append("Zmienność popytu z 4 tygodni nie może być ujemna.")

    if validation_errors:
        for err in validation_errors:
            st.error(err)
    else:
        input_df = pd.DataFrame([{
            "Product_Code": product_code,
            "Warehouse": warehouse,
            "Product_Category": product_category,
            "Year": year,
            "Month": month,
            "WeekOfYear": week_of_year,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_4": lag_4,
            "rolling_mean_4": rolling_mean_4,
            "rolling_std_4": rolling_std_4
        }])

        input_df[categorical_cols] = encoder.transform(input_df[categorical_cols])
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        pred_log = model.predict(input_df[feature_cols])
        prediction = np.expm1(pred_log[0])

        st.markdown("## 🔮 Wynik prognozy")
        st.success(f"Przewidywany tygodniowy popyt: {prediction:.2f}")
        st.info("Wynik oznacza przewidywany tygodniowy popyt dla podanych danych wejściowych.")

        log_row = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Product_Code": product_code,
            "Warehouse": warehouse,
            "Product_Category": product_category,
            "Year": year,
            "Month": month,
            "WeekOfYear": week_of_year,
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_4": lag_4,
            "rolling_mean_4": rolling_mean_4,
            "rolling_std_4": rolling_std_4,
            "Predicted_Order_Demand": float(prediction)
        }])

        log_file = "prediction_log.csv"

        if os.path.exists(log_file):
            log_row.to_csv(log_file, mode="a", header=False, index=False)
        else:
            log_row.to_csv(log_file, mode="w", header=True, index=False)

if os.path.exists("prediction_log.csv"):
    st.markdown("## 📋 Historia wykonanych prognoz")
    log_df = pd.read_csv("prediction_log.csv")
    st.dataframe(log_df.tail(10), use_container_width=True)