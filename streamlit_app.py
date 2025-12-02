import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

st.title("Prediksi Risiko Dropout Mahasiswa (KNN)")

uploaded = st.file_uploader("Upload dataset mahasiswa (CSV)", type=["csv"])

TARGET_COL = "students_dropout_academic_success"

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(df.head())

    if TARGET_COL not in df.columns:
        st.error(f"Target column '{TARGET_COL}' tidak ditemukan dalam file.")
    else:
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
        cat_cols = X.select_dtypes(include=['object','category']).columns.tolist()

        for c in numeric_cols:
            X[c] = X[c].fillna(X[c].median())
        for c in cat_cols:
            X[c] = X[c].fillna("MISSING")

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
            ]
        )

        st.sidebar.subheader("Hyperparameter KNN")
        k_val = st.sidebar.slider("n_neighbors", 1, 21, 5)
        weight_val = st.sidebar.selectbox("weights", ["uniform", "distance"])
        p_val = st.sidebar.selectbox("p (distance metric)", [1, 2])

        model = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("knn", KNeighborsClassifier(n_neighbors=k_val, weights=weight_val, p=p_val))
            ]
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, stratify=y if y.value_counts().min() >= 2 else None
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.write("Prediksi vs Aktual:")
        st.write(pd.DataFrame({"Actual": y_test.values, "Predicted": preds}).head())

        if st.checkbox("Tampilkan PCA Visualization"):
            X_trans = model.named_steps["prep"].transform(X)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_trans)
            st.scatter_chart(
                pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "label": y})
            )
