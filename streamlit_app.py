import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(page_title="Prediksi Risiko Dropout (KNN)", layout="wide")
st.title("Prediksi Risiko Dropout Mahasiswa (KNN)")

uploaded = st.file_uploader("Upload dataset mahasiswa (CSV)", type=["csv"])

if uploaded is None:
    st.info("Silakan upload file CSV yang mengandung kolom 'target' untuk memulai.")
    st.stop()

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df_raw = pd.read_csv(uploaded)
st.subheader("Preview dataset")
st.dataframe(df_raw.head())

if "target" not in df_raw.columns:
    st.error("Dataset HARUS memiliki kolom bernama 'target'. Ubah nama kolom target menjadi 'target'.")
    st.stop()

st.success("Kolom 'target' ditemukan.")

# -------------------------------------------------------------------
# PENGATURAN MODE KLASIFIKASI
# -------------------------------------------------------------------
st.sidebar.header("Pengaturan Model")
mode = st.sidebar.radio(
    "Mode Klasifikasi:",
    ["3 Kelas (Dropout / Enrolled / Graduate)", "Binary (Dropout vs Tidak Dropout)"]
)

df = df_raw.copy()

# Jika kolom target berupa teks, simpan dulu untuk keperluan mapping
target_raw = df["target"].copy()

# Mode binary: 0 -> 1 (dropout), 1/2 -> 0 (tidak dropout)
if mode == "Binary (Dropout vs Tidak Dropout)":
    # Asumsi: 0=Dropout, 1/2=Enrolled/Graduate
    df["target"] = df["target"].apply(lambda x: 1 if x == 0 else 0)

# -------------------------------------------------------------------
# ENCODING LABEL y -> ANGKA
# -------------------------------------------------------------------
# y_raw: label asli (bisa numerik atau string)
y_raw = df["target"]

if y_raw.dtype == "O":  # object/string -> encode manual
    unique_labels = sorted(y_raw.unique())
    label2id = {lab: idx for idx, lab in enumerate(unique_labels)}
    id2label = {idx: lab for lab, idx in label2id.items()}
    y = y_raw.map(label2id)
else:
    y = y_raw.copy()
    id2label = None  # tidak perlu mapping balik

X = df.drop(columns=["target"])

# -------------------------------------------------------------------
# TIPE FITUR
# -------------------------------------------------------------------
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# -------------------------------------------------------------------
# HYPERPARAMETER KNN
# -------------------------------------------------------------------
st.sidebar.subheader("Hyperparameter KNN")
k_val = st.sidebar.slider("n_neighbors (k)", 1, 21, 5)
weight_val = st.sidebar.selectbox("weights", ["uniform", "distance"])
p_val = st.sidebar.selectbox("Metric P", [1, 2])

# -------------------------------------------------------------------
# PIPELINE PREPROCESSING + MODEL
# -------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("knn", KNeighborsClassifier(
        n_neighbors=k_val,
        weights=weight_val,
        p=p_val
    ))
])

# -------------------------------------------------------------------
# TRAIN / TEST SPLIT & TRAINING
# -------------------------------------------------------------------
strat = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=strat,
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Kalau ada mapping id2label, konversi ke label teks untuk ditampilkan
if id2label is not None:
    y_test_disp = y_test.map(id2label)
    y_pred_disp = pd.Series(y_pred).map(id2label)
else:
    y_test_disp = y_test
    y_pred_disp = pd.Series(y_pred)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
cm = confusion_matrix(y_test, y_pred)

# -------------------------------------------------------------------
# TABS: EDA, Evaluasi, Prediksi
# -------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Evaluasi Model", "🎯 Prediksi Manual"])

# ================= TAB 1: EDA =================
with tab1:
    st.subheader("Exploratory Data Analysis (EDA)")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Dimensi dataset:", df.shape)
        st.write("Missing values per kolom:")
        st.dataframe(df.isnull().sum().to_frame("missing"))
    with c2:
        st.write("Distribusi target (label asli):")
        st.bar_chart(target_raw.value_counts())

# ================= TAB 2: Evaluasi Model =================
with tab2:
    st.subheader("Evaluasi Model (Test Set)")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Akurasi", f"{acc:.4f}")
    a2.metric("Precision (weighted)", f"{prec:.4f}")
    a3.metric("Recall (weighted)", f"{rec:.4f}")
    a4.metric("F1-score (weighted)", f"{f1:.4f}")

    st.markdown("---")
    st.write("Confusion Matrix (berdasarkan label numerik):")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

    st.markdown("---")
    st.write("Classification report (berdasarkan label numerik):")
    st.text(classification_report(y_test, y_pred, zero_division=0))

# ================= TAB 3: Prediksi Manual =================
with tab3:
    st.subheader("Prediksi Mahasiswa Baru (Input Manual)")

    st.info("Fitur numerik diisi ANGKA, fitur kategorikal diisi TEKS.")

    with st.expander("Isi data mahasiswa baru"):
        input_data = {}

        st.write("### Fitur Numerik")
        for col in numeric_cols:
            default_val = float(X[col].median())
            input_data[col] = st.number_input(col, value=default_val)

        st.write("### Fitur Kategorikal")
        for col in cat_cols:
            input_data[col] = st.text_input(col, value="MISSING")

        if st.button("Prediksi"):
            # Bangun df_new dengan kolom yang sama dan urutan sama seperti X
            df_new = pd.DataFrame([input_data])
            df_new = df_new[X.columns]  # pastikan urutan kolom sama

            # Biarkan pipeline menangani imputer/encoder; kita hanya menjaga tipe dasar
            # numerik sudah number_input -> float; kategori sudah string

            st.write("Preview input:")
            st.dataframe(df_new)

            try:
                pred_num = model.predict(df_new)[0]
                if id2label is not None:
                    pred_label = id2label[pred_num]
                    st.success(f"Hasil prediksi (label): {pred_label}  (kode: {pred_num})")
                else:
                    st.success(f"Hasil prediksi (kode numerik): {pred_num}")
            except Exception as e:
                st.error("Error memproses input baru:")
                st.code(str(e))
                st.info("Periksa kembali isian fitur (khususnya kolom numerik vs teks).")
