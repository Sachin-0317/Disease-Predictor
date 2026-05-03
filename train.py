import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import joblib, os

os.makedirs("models", exist_ok=True)

# --- Diabetes (Pima Indians dataset via URL) ---
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)
X, y = df.drop("Outcome", axis=1), df["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_d = StandardScaler()
X_train = scaler_d.fit_transform(X_train)
X_test = scaler_d.transform(X_test)
clf_d = RandomForestClassifier(n_estimators=100, random_state=42)
clf_d.fit(X_train, y_train)
print(f"Diabetes accuracy: {clf_d.score(X_test, y_test):.2f}")
joblib.dump(clf_d, "models/diabetes_model.pkl")
joblib.dump(scaler_d, "models/diabetes_scaler.pkl")
joblib.dump(list(df.drop("Outcome", axis=1).columns), "models/diabetes_features.pkl")

# --- Heart Disease (Cleveland dataset via URL) ---
url2 = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
df2 = pd.read_csv(url2)
X2, y2 = df2.drop("target", axis=1), df2["target"]
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
scaler_h = StandardScaler()
X2_train = scaler_h.fit_transform(X2_train)
X2_test = scaler_h.transform(X2_test)
clf_h = RandomForestClassifier(n_estimators=100, random_state=42)
clf_h.fit(X2_train, y2_train)
print(f"Heart Disease accuracy: {clf_h.score(X2_test, y2_test):.2f}")
joblib.dump(clf_h, "models/heart_model.pkl")
joblib.dump(scaler_h, "models/heart_scaler.pkl")
joblib.dump(list(df2.drop("target", axis=1).columns), "models/heart_features.pkl")

print("Models saved!")