# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

st.set_page_config(page_title="Dropout Predictor", layout="centered")

st.title("Prediksi Risiko Dropout, Logistic Regression")

# Load model dan metadata
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
with open("info.json", "r") as f:
    info = json.load(f)

selected_features = info["selected_features"]
target_map = info["target_map"]
inv_target_map = {v: k for k, v in target_map.items()}

st.write("Akurasi model (test):", round(info.get("accuracy", 0), 4))

# Jika dataset tersedia, gunakan untuk ambang input
try:
    df = pd.read_csv("students_dropout_academic_success.csv")
except Exception:
    df = None

# Input user
st.sidebar.header("Masukkan data")
inputs = {}
for feat in selected_features:
    if feat in ["Debtor", "Tuition fees up to date", "Scholarship holder"]:
        # categorical binary 0 atau 1
        val = st.sidebar.selectbox(fmt := feat, options=[0, 1], index=0)
        inputs[feat] = int(val)
    else:
        # numeric, set range dari dataset jika tersedia
        if df is not None and feat in df.columns:
            minv = float(df[feat].min())
            maxv = float(df[feat].max())
            meanv = float(df[feat].median())
            step = (maxv - minv) / 100 if maxv > minv else 1.0
            val = st.sidebar.number_input(feat, value=meanv, min_value=minv, max_value=maxv, step=step)
            inputs[feat] = float(val)
        else:
            # default range
            val = st.sidebar.number_input(feat, value=0.0)
            inputs[feat] = float(val)

# Predict button
if st.sidebar.button("Prediksi"):
    input_df = pd.DataFrame([inputs], columns=selected_features)
    X_scaled = scaler.transform(input_df)
    probs = model.predict_proba(X_scaled)[0]
    pred = int(model.predict(X_scaled)[0])
    st.subheader("Hasil Prediksi")
    st.write("Prediksi kelas:", inv_target_map[pred])
    # Tampilkan probabilitas per kelas
    prob_df = pd.DataFrame({
        "kelas": [inv_target_map[i] for i in range(len(probs))],
        "probabilitas": [float(round(p, 4)) for p in probs]
    })
    st.table(prob_df)

    # Penjelasan singkat
    st.write("Catatan: Gunakan data kualitatif dan nilai akademik yang valid. Model ini berfungsi sebagai panduan.")
