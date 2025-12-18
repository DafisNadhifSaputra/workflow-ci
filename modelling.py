"""
Model Training dengan MLflow Autolog
Author: Dafis Nadhif Saputra
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


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


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_feature_importance(model, feature_names, save_path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def train_with_autolog(X_train, X_test, y_train, y_test, experiment_name, model_params):
    mlflow.set_experiment(experiment_name)
    
    # Menggunakan mlflow.autolog() sesuai standar MLflow
    mlflow.autolog(log_models=True)
    
    run_name = f"RF_Autolog_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        
        # Save artifacts
        joblib.dump(scaler, "scaler.joblib")
        mlflow.log_artifact("scaler.joblib")
        
        plot_confusion_matrix(y_test, y_pred, "confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        plot_feature_importance(model, X_train.columns.tolist(), "feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # Cleanup
        for f in ["scaler.joblib", "confusion_matrix.png", "feature_importance.png"]:
            if os.path.exists(f):
                os.remove(f)
        
        run_id = run.info.run_id
    
    mlflow.autolog(disable=True)
    return model, run_id


def main():
    print("=" * 50)
    print("Model Training - Breast Cancer Classification")
    print("=" * 50)
    
    DATA_PATH = "data/breast_cancer_data.csv"
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
    
    model, run_id = train_with_autolog(
        X_train, X_test, y_train, y_test,
        EXPERIMENT_NAME, model_params
    )
    
    print(f"\nTraining complete! Run ID: {run_id}")


if __name__ == "__main__":
    main()