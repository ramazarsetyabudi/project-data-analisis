# streamlit_app.py  (VERSI FINAL - Perbaikan prediksi manual)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA

st.set_page_config(page_title="Prediksi Risiko Dropout (KNN)", layout="wide")
st.title("Prediksi Risiko Dropout Mahasiswa (KNN) — Final")

uploaded = st.file_uploader("Upload dataset mahasiswa (CSV)", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.subheader("Preview dataset (5 baris pertama)")
    st.dataframe(df_raw.head())

    # cek kolom target
    if "target" not in df_raw.columns:
        st.error("Dataset harus memiliki kolom bernama 'target'. Ganti nama kolom target menjadi 'target' (huruf kecil).")
        st.stop()

    st.success("Kolom 'target' ditemukan.")

    # --------------------------------------------------
    # Pengaturan
    # --------------------------------------------------
    st.sidebar.header("Pengaturan Model")
    mode = st.sidebar.radio("Mode klasifikasi:",
                            ["3 Kelas (Dropout / Enrolled / Graduate)", "Binary (Dropout vs Tidak Dropout)"])

    df = df_raw.copy()
    if mode == "Binary (Dropout vs Tidak Dropout)":
        # convert: 0 -> 1 (dropout), 1/2 -> 0 (tidak dropout)
        df["target"] = df["target"].apply(lambda x: 1 if x == 0 else 0)

    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # --------------------------------------------------
    # Hyperparameter
    # --------------------------------------------------
    st.sidebar.subheader("Hyperparameter KNN")
    k_val = st.sidebar.slider("n_neighbors (k)", 1, 21, 5)
    weight_val = st.sidebar.selectbox("weights", ["uniform", "distance"])
    p_val = st.sidebar.selectbox("Metric P", [1, 2])

    # --------------------------------------------------
    # Preprocessing pipeline (robust)
    # --------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    model = Pipeline(steps=[
        ("prep", preprocessor),
        ("knn", KNeighborsClassifier(n_neighbors=k_val, weights=weight_val, p=p_val))
    ])

    # --------------------------------------------------
    # Split & train
    # --------------------------------------------------
    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=strat, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # --------------------------------------------------
    # UI: Tabs
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Evaluasi Model", "🎯 Prediksi Manual"])

    # TAB 1: EDA
    with tab1:
        st.subheader("Exploratory Data Analysis")
        colA, colB = st.columns(2)
        with colA:
            st.write("Dimensi dataset:", df.shape)
            st.write("Missing values per kolom:")
            st.dataframe(df.isnull().sum().to_frame("missing").T)
        with colB:
            st.write("Distribusi target:")
            st.bar_chart(df["target"].value_counts().sort_index())

    # TAB 2: Evaluasi
    with tab2:
        st.subheader("Evaluasi Model")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Akurasi", f"{acc:.4f}")
        c2.metric("Precision (weighted)", f"{prec:.4f}")
        c3.metric("Recall (weighted)", f"{rec:.4f}")
        c4.metric("F1 (weighted)", f"{f1:.4f}")

        st.markdown("---")
        st.write("Confusion Matrix:")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, cm[i][j], ha="center", va="center")
        st.pyplot(fig)

        st.markdown("---")
        st.write("Classification report:")
        st.text(classification_report(y_test, y_pred))

        # ================= TAB 3: Prediksi Manual =================
    with tab3:
        st.subheader("Prediksi Mahasiswa Baru")

        st.info("Kolom numerik hanya boleh diisi ANGKA. Kolom kategorikal hanya boleh TEKS.")

        with st.expander("Isi fitur mahasiswa baru"):
            input_data = {}

            st.write("### Fitur Numerik")
            for col in numeric_cols:
                default_val = float(X[col].median())
                input_data[col] = st.number_input(col, value=default_val)

            st.write("### Fitur Kategorikal")
            for col in cat_cols:
                input_data[col] = st.text_input(col, value="MISSING")

            if st.button("Prediksi"):
                # bangun dataframe input
                df_new = pd.DataFrame([input_data])

                # pastikan kolom lengkap
                for col in X_train.columns:
                    if col not in df_new.columns:
                        df_new[col] = np.nan

                df_new = df_new[X_train.columns]

                # pastikan kolom numerik -> numeric
                for col in numeric_cols:
                    df_new[col] = pd.to_numeric(df_new[col], errors="coerce")

                # pastikan kolom kategorikal -> string
                for col in cat_cols:
                    df_new[col] = df_new[col].astype(str)

                st.write("Preview input:")
                st.dataframe(df_new)

                try:
                    X_new_trans = model.named_steps["prep"].transform(df_new)
                    pred_new = model.named_steps["knn"].predict(X_new_trans)[0]
                    st.success(f"Hasil prediksi: {pred_new}")

                except Exception as e:
                    st.error("Error memproses input:")
                    st.code(str(e))
                    st.info("Periksa apakah Anda memasukkan teks ke kolom numerik.")
else:
    st.info("Silakan upload file CSV yang mengandung kolom 'target' untuk memulai.")
