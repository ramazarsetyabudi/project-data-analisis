import streamlit as st
import pandas as pd
import numpy as np
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
    page_title="Prediksi Risiko Dropout Mahasiswa (KNN)",
    layout="wide"
)

st.title("Prediksi Risiko Dropout Mahasiswa (KNN)")

uploaded = st.file_uploader("Upload dataset mahasiswa (CSV)", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
    st.subheader("Preview Dataset")
    st.dataframe(df_raw.head())

    if "target" not in df_raw.columns:
        st.error("Kolom target 'target' tidak ditemukan dalam dataset.")
        st.stop()

    st.success("Kolom target ditemukan: 'target'")

    st.sidebar.header("Pengaturan Model")

    mode = st.sidebar.radio(
        "Tipe Klasifikasi:",
        ["3 Kelas (Dropout / Enrolled / Graduate)", "Binary (Dropout vs Tidak Dropout)"]
    )

    df = df_raw.copy()

    if mode == "Binary (Dropout vs Tidak Dropout)":

        df["target"] = df["target"].apply(lambda x: 1 if x == 0 else 0)

    X = df.drop(columns=["target"])
    y = df["target"]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    st.sidebar.subheader("Hyperparameter KNN")
    k_val = st.sidebar.slider("n_neighbors (k)", 1, 21, 5, step=2)
    weight_val = st.sidebar.selectbox("weights", ["uniform", "distance"])
    p_val = st.sidebar.selectbox("Metric P", [1, 2], format_func=lambda x: f"{x} (Manhattan)" if x == 1 else f"{x} (Euclidean)")

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

    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        stratify=strat,
        random_state=42
    )

    k_list = list(range(1, 22, 2))
    cv_scores = []
    for k in k_list:
        tmp_model = Pipeline(steps=[
            ("prep", preprocessor),
            ("knn", KNeighborsClassifier(
                n_neighbors=k,
                weights=weight_val,
                p=p_val
            ))
        ])
        scores = cross_val_score(tmp_model, X_train, y_train, cv=4, scoring="accuracy")
        cv_scores.append(scores.mean())

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    tab1, tab2, tab3 = st.tabs(["📊 EDA", "🧠 Training & Evaluasi", "🎯 Prediksi Manual"])

    with tab1:
        st.subheader("Exploratory Data Analysis (EDA)")

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Dimensi Data**")
            st.write(f"{df.shape[0]} baris, {df.shape[1]} kolom")

            st.markdown("**Tipe Data per Kolom**")
            st.dataframe(df.dtypes.to_frame("dtype"))

        with colB:
            st.markdown("**Missing Values per Kolom**")
            st.dataframe(df.isnull().sum().to_frame("missing"))

            st.markdown("**Distribusi Kelas (target)**")
            st.bar_chart(df["target"].value_counts().sort_index())

    with tab2:
        st.subheader("Metode KNN & Evaluasi Model")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Akurasi (test)", f"{acc:.4f}")
        col2.metric("Precision (weighted)", f"{prec:.4f}")
        col3.metric("Recall (weighted)", f"{rec:.4f}")
        col4.metric("F1-Score (weighted)", f"{f1:.4f}")

        st.markdown("---")
        st.markdown("### Confusion Matrix")

        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        im = ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_title("Confusion Matrix")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax_cm.text(j, i, cm[i, j], ha="center", va="center")
        fig_cm.colorbar(im, ax=ax_cm)
        st.pyplot(fig_cm)

        st.markdown("---")
        st.markdown("### Akurasi vs Jumlah Tetangga (k) — Cross Validation")

        fig_k, ax_k = plt.subplots(figsize=(6, 4))
        ax_k.plot(k_list, cv_scores, marker="o")
        ax_k.set_xlabel("k (n_neighbors)")
        ax_k.set_ylabel("Mean CV Accuracy")
        ax_k.set_title("Mean CV Accuracy vs k")
        ax_k.grid(True)
        st.pyplot(fig_k)

        st.markdown("---")
        st.markdown("### Visualisasi Decision Boundary (PCA 2D)")

        X_all = pd.concat([X_train, X_test], axis=0)
        y_all = pd.concat([y_train, y_test], axis=0)
        X_all_trans = model.named_steps["prep"].transform(X_all)

        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_all_trans)
        df_pca = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "label": y_all.reset_index(drop=True)
        })
        st.scatter_chart(df_pca, x="PC1", y="PC2", color="label")

        st.markdown("---")
        st.markdown("### Classification Report (detail per kelas)")
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report_dict).T)

    with tab3:
        st.subheader("Prediksi Mahasiswa Baru (Input Manual)")

        with st.expander("Isi fitur mahasiswa baru di bawah ini"):
            input_data = {}

            st.markdown("**Fitur Numerik**")
            for col in numeric_cols:
                default_val = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                input_data[col] = st.number_input(col, value=default_val)

            st.markdown("---")
            st.markdown("**Fitur Kategorikal**")
            for col in cat_cols:
                input_data[col] = st.text_input(col, value="MISSING")

            if st.button("Prediksi"):
                df_new = pd.DataFrame([input_data])

                for col in X_train.columns:
                    if col not in df_new.columns:
                        df_new[col] = np.nan

                df_new = df_new[X_train.columns]

                pred_new = model.predict(df_new)[0]
                st.success(f"Hasil Prediksi untuk data baru: {pred_new}")

else:
    st.info("Silakan upload file CSV terlebih dahulu.")
