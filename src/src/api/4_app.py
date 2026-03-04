"""
Flask API for UPI Fraud Detection

This API loads the trained fraud detection model and
provides an endpoint to predict whether a transaction
is fraudulent or legitimate.
"""

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model_path = "../models/fraud_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return "UPI Fraud Detection API is running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = [
        data["transaction_amount"],
        data["transaction_hour"],
        data["device_type"],
        data["location_change"],
        data["account_age_days"],
        data["previous_transactions"]
    ]

    features_array = np.array(features).reshape(1, -1)

    prediction = model.predict(features_array)

    if prediction[0] == 1:
        result = "Fraudulent Transaction"
    else:
        result = "Legitimate Transaction"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(debug=True)
