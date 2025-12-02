import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.title("Aplikasi Prediksi Dropout / Academic Success Menggunakan KNN")

@st.cache_data
def load_data():
    return pd.read_csv("students_dropout_academic_success.csv")

df = load_data()

st.subheader("Preview Dataset")
st.write(df.head())


possible_targets = [
    "Target", "target", "Status", "status",
    "Dropout", "dropout", "Graduate", "graduate", "Outcome"
]

detected_target = None
for col in df.columns:
    if col in possible_targets:
        detected_target = col
        break

if detected_target is None:
    st.error("Kolom target tidak ditemukan. Pastikan nama kolom target benar.")
    st.stop()

st.success(f"Kolom target terdeteksi: {detected_target}")


X = df.drop(columns=[detected_target])
y = df[detected_target]


X = X.select_dtypes(include=["int64", "float64"])


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='minkowski'
)

knn.fit(X_train, y_train)


with open("model_knn.pkl", "wb") as f:
    pickle.dump((knn, scaler, list(X.columns)), f)

st.subheader("Input Fitur untuk Prediksi")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, value=float(df[col].mean()))

input_df = pd.DataFrame([user_input])


if st.button("Prediksi"):
    with open("model_knn.pkl", "rb") as f:
        knn_model, knn_scaler, cols = pickle.load(f)

    input_scaled = knn_scaler.transform(input_df[cols])
    pred = knn_model.predict(input_scaled)[0]

    st.subheader("Hasil Prediksi")
    st.success(f"Hasil Model KNN: {pred}")
