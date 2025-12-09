# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json

# Load dataset
df = pd.read_csv("students_dropout_academic_success.csv")

# Pastikan kolom target sesuai label pada dataset
# Mapping target ke angka: 0 Dropout, 1 Enrolled, 2 Graduate
target_map = {"Dropout": 0, "Enrolled": 1, "Graduate": 2}
df["target"] = df["target"].map(target_map)

# Pilih fitur penting untuk antarmuka Streamlit
selected_features = [
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (approved)",
    "Age at enrollment",
    "Debtor",
    "Tuition fees up to date",
    "Scholarship holder",
    "Admission grade"
]

# Isi nan jika ada
df[selected_features] = df[selected_features].fillna(df[selected_features].median())

X = df[selected_features]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Latih Logistic Regression multi kelas
model = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=5000)
model.fit(X_train_scaled, y_train)

# Evaluasi
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("Accuracy:", round(acc, 4))
print("Classification report (summary):")
for k, v in report.items():
    if k in ["accuracy", "macro avg", "weighted avg"]:
        print(k, v)

# Simpan model, scaler, dan info
import os
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

info = {
    "selected_features": selected_features,
    "accuracy": float(acc),
    "target_map": target_map
}
with open("model/info.json", "w") as f:
    json.dump(info, f)

print("Model dan scaler tersimpan di folder model/")
