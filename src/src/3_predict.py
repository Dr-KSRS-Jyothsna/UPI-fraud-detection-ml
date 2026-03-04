"""
Prediction Script
UPI Fraud Detection Project

This script loads the trained machine learning model and
predicts whether a transaction is fraudulent or legitimate.
"""

import pickle
import numpy as np


# Load trained model
model_path = "../models/fraud_model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)


def predict_transaction(features):
    """
    Predict if a transaction is fraud.

    Parameters:
    features (list): transaction feature vector

    Returns:
    str
    """

    prediction = model.predict([features])

    if prediction[0] == 1:
        return "Fraudulent Transaction"
    else:
        return "Legitimate Transaction"


if __name__ == "__main__":

    # Example transaction
    sample_transaction = [
        75000,  # transaction_amount
        2,      # transaction_hour
        0,      # device_type (0 = mobile, 1 = web)
        1,      # location_change
        5,      # account_age_days
        3       # previous_transactions
    ]

    result = predict_transaction(sample_transaction)

    print("Prediction Result:", result)
