import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.title("Prediksi Dropout / Academic Success")

@st.cache_data
def load_data():
    return pd.read_csv("students_dropout_academic_success.csv")

df = load_data()

st.subheader("Preview Dataset")
st.write(df.head())

# ==========================================
# AUTOMATIC TARGET DETECTION
# ==========================================
possible_targets = ["Target", "target", "status", "Status", "Dropout", "Graduate"]

detected_target = None
for col in df.columns:
    if col in possible_targets:
        detected_target = col
        break

if detected_target is None:
    st.error("Kolom target tidak ditemukan. Pastikan nama kolom target benar.")
    st.stop()

st.success(f"Kolom target terdeteksi: {detected_target}")

# ==========================================
# PREPROCESSING
# ==========================================
X = df.drop(columns=[detected_target])
y = df[detected_target]

# Pastikan hanya kolom numerik
X = X.select_dtypes(include=["int64", "float64"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================================
# MODEL
# ==========================================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler, list(X.columns)), f)

# ==========================================
# STREAMLIT INPUT FORM
# ==========================================
st.subheader("Input Fitur")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, value=float(df[col].mean()))

input_df = pd.DataFrame([user_input])

# ==========================================
# PREDIKSI
# ==========================================
if st.button("Prediksi"):
    with open("model.pkl", "rb") as f:
        model, scaler, cols = pickle.load(f)

    input_scaled = scaler.transform(input_df[cols])
    pred = model.predict(input_scaled)[0]

    st.success(f"Prediksi Model: {pred}")
