"""
Data Preprocessing Module
UPI Fraud Detection Project

This module loads the dataset and performs preprocessing steps
such as encoding categorical variables and feature scaling.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """
    Load dataset from CSV file.

    Parameters:
    file_path (str): Path to dataset

    Returns:
    pandas.DataFrame
    """

    df = pd.read_csv(file_path)
    return df


def encode_features(df):
    """
    Encode categorical features.
    """

    encoder = LabelEncoder()

    if "device_type" in df.columns:
        df["device_type"] = encoder.fit_transform(df["device_type"])

    return df


def scale_features(X):
    """
    Scale numerical features using StandardScaler.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled


def preprocess_data(df):
    """
    Complete preprocessing pipeline.
    """

    df = encode_features(df)

    X = df.drop("fraud_label", axis=1)
    y = df["fraud_label"]

    X_scaled = scale_features(X)

    return X_scaled, y


if __name__ == "__main__":

    # Example usage
    data_path = "../data/upi_transactions.csv"

    df = load_data(data_path)

    X, y = preprocess_data(df)

    print("Data preprocessing completed successfully")
    print("Feature shape:", X.shape)
