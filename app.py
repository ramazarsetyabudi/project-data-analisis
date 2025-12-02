import streamlit as st
import pandas as pd
from prediction import predict_with_proba, get_classes

st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("Iris Flower Classifier")
st.write("Pilih fitur sepal/petal lalu klik **Predict**. Aplikasi akan menampilkan prediksi dan diagram probabilitas.")

st.header("Plant Features")
col1, col2 = st.columns(2)

with col1:
    sepal_l = st.slider("Sepal length (cm)", 1.0, 8.0, 5.0, 0.1)
    sepal_w = st.slider("Sepal width (cm)", 1.0, 5.0, 3.0, 0.1)

with col2:
    petal_l = st.slider("Petal length (cm)", 0.0, 7.0, 4.0, 0.1)
    petal_w = st.slider("Petal width (cm)", 0.0, 2.5, 1.0, 0.1)

if st.button("Predict"):
 
    pred_label, probs = predict_with_proba([sepal_l, sepal_w, petal_l, petal_w])


    st.subheader("Hasil Prediksi")
    st.success(f"Jenis bunga: **{pred_label}**")


    st.write("**Input (cm):**", {
        "sepal_length": sepal_l,
        "sepal_width": sepal_w,
        "petal_length": petal_l,
        "petal_width": petal_w
    })

  
    classes = get_classes()
    if probs is not None and len(classes) == len(probs):
        df = pd.DataFrame({"Probability": probs}, index=classes)
        df = df.sort_values("Probability", ascending=True)  
        st.subheader("Probabilitas per Kelas")
        st.bar_chart(df)   
       
        st.table(df.T)
    else:
        st.info("Model tidak menyediakan probabilitas. Hanya menampilkan label prediksi.")
