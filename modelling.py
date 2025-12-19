"""
Model Training dengan MLflow Autolog
Author: Dafis Nadhif Saputra
"""

import matplotlib
matplotlib.use('Agg')  # Fix CI headless environment error

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
from datetime import datetime


def load_data(data_path):
    df = pd.read_csv(data_path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def prepare_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test, experiment_name, model_params):
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()
    
    run_name = f"RF_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        run_id = run.info.run_id
    
    mlflow.autolog(disable=True)
    return model, run_id


def main():
    print("=" * 50)
    print("Model Training - Breast Cancer Classification")
    print("=" * 50)
    
    DATA_PATH = "data/breast_cancer_preprocessed.csv"
    TARGET_COLUMN = "target"
    EXPERIMENT_NAME = "SMSML_Dafis_Nadhif_Saputra"
    
    mlflow.set_tracking_uri("mlruns")
    
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(df, TARGET_COLUMN)
    
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model, run_id = train_model(
        X_train, X_test, y_train, y_test,
        EXPERIMENT_NAME, model_params
    )
    
    print(f"\nTraining complete! Run ID: {run_id}")


if __name__ == "__main__":
    main()