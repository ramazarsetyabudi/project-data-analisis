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

st.set_page_config(
    page_title="Prediksi Risiko Dropout Mahasiswa",
    layout="wide"
)

st.title("Prediksi Risiko Dropout Mahasiswa (KNN)")

uploaded = st.file_uploader("Upload dataset mahasiswa (CSV)", type=["csv"])

if uploaded is not None:

    df_raw = pd.read_csv(uploaded)
    st.subheader("Preview Dataset")
    st.dataframe(df_raw.head())

    if "target" not in df_raw.columns:
        st.error("Dataset harus memiliki kolom bernama 'target'.")
        st.stop()

    st.success("Kolom target ditemukan.")

    st.sidebar.header("Pengaturan Model")

    mode = st.sidebar.radio(
        "Tipe Klasifikasi:",
        ["3 Kelas (Dropout / Enrolled / Graduate)",
         "Binary (Dropout vs Tidak Dropout)"]
    )

    df = df_raw.copy()

    if mode == "Binary (Dropout vs Tidak Dropout)":
        df["target"] = df["target"].apply(lambda x: 1 if x == 0 else 0)

    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    st.sidebar.subheader("Hyperparameter KNN")

    k_val = st.sidebar.slider("n_neighbors (k)", 1, 21, 5)
    weight_val = st.sidebar.selectbox("weights", ["uniform", "distance"])
    p_val = st.sidebar.selectbox("Metric P", [1, 2])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
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

    strat = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=strat,
        random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)


    tab1, tab2, tab3 = st.tabs(["EDA", "Evaluasi Model", "Prediksi Manual"])


    with tab1:
        st.subheader("Exploratory Data Analysis")

        colA, colB = st.columns(2)

        with colA:
            st.write("**Dimensi Dataset:**")
            st.write(df.shape)

            st.write("**Missing Values:**")
            st.dataframe(df.isnull().sum().to_frame("missing"))

        with colB:
            st.write("**Distribusi Target:**")
            st.bar_chart(df["target"].value_counts().sort_index())


    with tab2:
        st.subheader("Evaluasi Model KNN")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall", f"{rec:.4f}")
        col4.metric("F1", f"{f1:.4f}")

        st.markdown("---")

        st.write("### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(cm, cmap="Blues")
        for i in range(len(cm)):
            for j in range(len(cm[i])):
                ax.text(j, i, cm[i][j], ha="center", va="center")
        st.pyplot(fig)

        st.markdown("---")

        st.write("### Classification Report")
        st.text(classification_report(y_test, y_pred))


    with tab3:
        st.subheader("Prediksi untuk Mahasiswa Baru")

        with st.expander("Isi Data Mahasiswa"):
            input_data = {}

            st.write("### Fitur Numerik")
            for col in numeric_cols:
                input_data[col] = st.number_input(
                    col, value=float(X[col].median())
                )

            st.write("### Fitur Kategorikal")
            for col in cat_cols:
                input_data[col] = st.text_input(col, "MISSING")

            if st.button("Prediksi"):
                df_new = pd.DataFrame([input_data])

                for col in X_train.columns:
                    if col not in df_new.columns:
                        df_new[col] = np.nan

                df_new = df_new[X_train.columns]

                pred_new = model.predict(df_new)[0]

                st.success(f"Hasil Prediksi: {pred_new}")

else:
    st.info("students_dropout_academic_success.csv")
