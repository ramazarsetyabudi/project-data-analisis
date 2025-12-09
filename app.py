import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    df = pd.read_csv("students_dropout_academic_success.csv")
    return df

df = load_data()

st.title("Predicting Student Dropout and Academic Success")
st.write("Aplikasi prediksi berbasis **Logistic Regression Stacking Model**.")


le = LabelEncoder()
df["target"] = le.fit_transform(df["target"])

df["SuccessRate_1"] = (
    df["Curricular units 1st sem (approved)"] /
    df["Curricular units 1st sem (enrolled)"]
).fillna(0)

X = df.drop(["target"], axis=1)
y = df["target"]

num_cols = X.select_dtypes(include="number").columns.tolist()
scaler = MinMaxScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


rf = RandomForestClassifier(n_estimators=350, max_depth=10, random_state=42)

xgb = XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    learning_rate=0.05,
    max_depth=6,
    n_estimators=500,
    eval_metric="mlogloss",
    random_state=42
)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

train_meta = np.hstack([
    rf.predict_proba(X_train),
    xgb.predict_proba(X_train)
])

test_meta = np.hstack([
    rf.predict_proba(X_test),
    xgb.predict_proba(X_test)
])

meta_lr = LogisticRegression(max_iter=800, multi_class="multinomial")
meta_lr.fit(train_meta, y_train)

def stacked_predict(X):
    """Generate predictions using RF + XGB → Logistic Regression meta model."""
    meta_features = np.hstack([
        rf.predict_proba(X),
        xgb.predict_proba(X)
    ])
    return np.argmax(meta_lr.predict_proba(meta_features), axis=1)

st.header("Dataset Overview")
st.dataframe(df.head())

st.subheader("Distribusi Target Mahasiswa")
fig = plt.figure()
sns.countplot(x=df["target"], palette="viridis")
plt.xticks([0, 1, 2], le.classes_)
st.pyplot(fig)

st.header("Coba Prediksi Status Mahasiswa")

input_data = {}

for col in X.columns:
    default_val = float(df[col].median())
    input_data[col] = st.number_input(f"{col}", value=default_val)

if st.button("Prediksi"):
    user_df = pd.DataFrame([input_data])

    user_df[num_cols] = scaler.transform(user_df[num_cols])

    pred = stacked_predict(user_df)[0]
    label = le.inverse_transform([pred])[0]

    st.success(f"Hasil Prediksi: **{label}**")


st.header("Akurasi Model (Logistic Regression Stacking)")

test_pred = stacked_predict(X_test)
accuracy = (test_pred == y_test).mean()

st.metric("Akurasi Model", f"{accuracy * 100:.2f}%")
