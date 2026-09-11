from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Path to trained model
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_fraud_model.joblib"
)

# Load trained model
model_data = joblib.load(MODEL_PATH)

model = model_data["model"]
threshold = model_data["threshold"]
feature_columns = model_data["feature_columns"]


# Home page
@app.route("/", methods=["GET"])
def home():
    return send_from_directory(
        os.path.dirname(__file__),
        "index.html"
    )


# Fraud prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Create dataframe from received data
        transaction = pd.DataFrame([data])

        # Ensure correct feature order
        transaction = transaction[feature_columns]

        # Get fraud probability
        probability = model.predict_proba(transaction)[0][1]

        # Fraud prediction
        prediction = int(probability >= threshold)

        # Risk level
        if probability < 0.20:
            risk = "Low"
        elif probability < 0.50:
            risk = "Medium"
        elif probability < 0.80:
            risk = "High"
        else:
            risk = "Critical"

        return jsonify({
            "fraud_probability": round(float(probability), 4),
            "prediction": prediction,
            "risk_level": risk
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400
