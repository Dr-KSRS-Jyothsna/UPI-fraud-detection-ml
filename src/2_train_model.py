"""
Model Training Script
UPI Fraud Detection Project

This script trains a machine learning model to detect fraudulent transactions
and saves the trained model for later predictions.
"""

import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from data_preprocessing import load_data, preprocess_data


def train_model():

    # Load dataset
    df = load_data("../data/upi_transactions.csv")

    # Preprocess data
    X, y = preprocess_data(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initialize model
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save trained model
    with open("../models/fraud_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model saved successfully!")


if __name__ == "__main__":
    train_model()
