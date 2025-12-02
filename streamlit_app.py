import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.decomposition import PCA

st.set_page_config(page_title="Prediksi Risiko Dropout Mahasiswa", layout="wide")

st.title("Prediksi Risiko Dropout Mahasiswa (KNN)")

uploaded = st.file_uploader("Upload dataset mahasiswa (CSV)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    if "target" not in df.columns:
        st.error("Kolom target 'target' tidak ditemukan dalam dataset.")
        st.stop()

    st.success("Kolom target ditemukan: 'target'")

    st.sidebar.title("Pengaturan Model")
    mode = st.sidebar.radio(
        "Tipe Klasifikasi:",
        ["3 Kelas (Dropout / Enrolled / Graduate)", "Binary (Dropout vs Tidak Dropout)"]
    )

    df_proc = df.copy()

    if mode == "Binary (Dropout vs Tidak Dropout)":
        df_proc["target"] = df_proc["target"].apply(lambda x: 1 if x != 0 else 0)

    X = df_proc.drop(columns=["target"])
    y = df_proc["target"]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    for c in numeric_cols:
        X[c] = X[c].fillna(X[c].median())
    for c in cat_cols:
        X[c] = X[c].fillna("MISSING")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        ]
    )

    st.sidebar.title("Hyperparameter KNN")
    k_val = st.sidebar.slider("n_neighbors", 1, 21, 5)
    weight_val = st.sidebar.selectbox("weights", ["uniform", "distance"])
    p_val = st.sidebar.selectbox("Metric P (1=Manhattan, 2=Euclidean)", [1, 2])

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("knn", KNeighborsClassifier(n_neighbors=k_val, weights=weight_val, p=p_val))
        ]
    )

    strat = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=strat, random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.subheader("Evaluasi Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy_score(y_test, preds):.4f}")
    col2.metric("Precision", f"{precision_score(y_test, preds, average='weighted'):.4f}")
    col3.metric("Recall", f"{recall_score(y_test, preds, average='weighted'):.4f}")
    col4.metric("F1-Score", f"{f1_score(y_test, preds, average='weighted'):.4f}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    st.pyplot(fig)

    st.subheader("Visualisasi PCA (2D)")
    if st.checkbox("Tampilkan PCA Plot"):
        X_trans = model.named_steps["prep"].transform(X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_trans)

        df_pca = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            "Label": y
        })

        st.scatter_chart(df_pca, x="PC1", y="PC2", color="Label")

    st.subheader("Prediksi Manual")
    with st.expander("Input fitur mahasiswa baru"):
        input_data = {}
        for col in numeric_cols:
            input_data[col] = st.number_input(f"{col}", value=float(X[col].median()))

        for col in cat_cols:
            input_data[col] = st.text_input(f"{col}", "MISSING")

        if st.button("Prediksi"):
            df_new = pd.DataFrame([input_data])
            pred_new = model.predict(df_new)[0]
            st.success(f"Hasil Prediksi: {pred_new}")
