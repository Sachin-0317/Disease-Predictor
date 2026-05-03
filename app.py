from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import shap
import pandas as pd

app = Flask(__name__)

# Load models
diabetes_model = joblib.load("models/diabetes_model.pkl")
diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
diabetes_features = joblib.load("models/diabetes_features.pkl")

heart_model = joblib.load("models/heart_model.pkl")
heart_scaler = joblib.load("models/heart_scaler.pkl")
heart_features = joblib.load("models/heart_features.pkl")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    disease = data.get("disease")

    if disease == "diabetes":
        features = diabetes_features
        scaler = diabetes_scaler
        model = diabetes_model
    else:
        features = heart_features
        scaler = heart_scaler
        model = heart_model

    values = pd.DataFrame([[data[f] for f in features]], columns=features)
    scaled = scaler.transform(values)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][int(pred)]

    # SHAP explanation
    explainer = shap.TreeExplainer(model)
   # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled)

    if isinstance(shap_values, list):
        sv = np.array(shap_values[1][0])
    else:
        sv = np.array(shap_values[0])

    sv = sv.flatten()
    shap_dict = dict(zip(features, [round(float(v), 4) for v in sv]))

    return jsonify({
        "prediction": int(pred),
        "label": "Positive" if pred == 1 else "Negative",
        "confidence": round(float(prob) * 100, 2),
        "disease": disease,
        "shap_values": shap_dict
    })

if __name__ == "__main__":
    app.run(debug=True)