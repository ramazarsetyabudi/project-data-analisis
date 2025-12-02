import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ==========================
# LOAD DATASET
# ==========================
@st.cache_data
def load_data():
    df = pd.read_csv("students_dropout_academic_success.csv")
    return df

df = load_data()

# ==========================
# PREPROCESSING
# ==========================
# Menghapus ID atau kolom yang tidak dipakai jika ada
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Tentukan target (misal: 'Target')
# Ganti sesuai dataset Anda
target_column = "Target"

X = df.drop(columns=[target_column])
y = df[target_column]

# Standarisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# SPLIT DATA
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Simpan model
with open("model_rf.pkl", "wb") as f:
    pickle.dump((model, scaler, list(X.columns)), f)

# ==========================
# STREAMLIT APP
# ==========================
st.title("Prediksi Dropout / Academic Success")

st.write("Aplikasi ini melakukan prediksi berdasarkan dataset mahasiswa.")

st.subheader("Input Fitur")

# Buat input otomatis berdasarkan kolom
user_input = {}
for col in X.columns:
    val = st.number_input(f"{col}", value=float(df[col].mean()))
    user_input[col] = val

# Convert ke dataframe
input_df = pd.DataFrame([user_input])

# ==========================
# PREDICT
# ==========================
if st.button("Prediksi"):
    # Load model
    with open("model_rf.pkl", "rb") as f:
        saved_model, saved_scaler, saved_cols = pickle.load(f)

    # Urutkan kolom
    input_scaled = saved_scaler.transform(input_df[saved_cols])

    pred = saved_model.predict(input_scaled)[0]

    st.subheader("Hasil Prediksi")
    st.success(f"Hasil Model: {pred}")
