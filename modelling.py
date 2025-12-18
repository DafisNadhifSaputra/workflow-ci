"""
modelling.py - Model Training dengan MLflow Autolog
Proyek Akhir: Membangun Sistem Machine Learning
Dicoding Indonesia

Author: Dafis Nadhif Saputra
Version: 1.1 - 2024-12-18
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(data_path):
    """
    Memuat data yang sudah dipreprocess.
    
    Args:
        data_path: Path ke file data (CSV)
    
    Returns:
        DataFrame yang sudah dipreprocess
    """
    print(f"[INFO] Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Data berhasil dimuat. Shape: {df.shape}")
    return df


def prepare_features_target(df, target_column, test_size=0.2, random_state=42):
    """
    Mempersiapkan fitur dan target untuk training.
    
    Args:
        df: DataFrame dengan data
        target_column: Nama kolom target
        test_size: Proporsi data test
        random_state: Random seed untuk reprodusibilitas
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Encode target jika kategorikal
    if y.dtype == 'object':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"[INFO] Target encoded. Classes: {le.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[INFO] Training set size: {len(X_train)}")
    print(f"[INFO] Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def create_confusion_matrix_plot(y_true, y_pred, save_path):
    """
    Membuat dan menyimpan plot confusion matrix.
    
    Args:
        y_true: Label sebenarnya
        y_pred: Prediksi model
        save_path: Path untuk menyimpan plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Confusion matrix saved to: {save_path}")
    return save_path


def train_model_with_autolog(X_train, X_test, y_train, y_test, 
                             experiment_name="SMSML_Experiment",
                             model_params=None):
    """
    Melatih model menggunakan MLflow Autolog.
    
    Args:
        X_train, X_test, y_train, y_test: Data training dan testing
        experiment_name: Nama experiment di MLflow
        model_params: Parameter untuk RandomForestClassifier
    
    Returns:
        trained_model, run_id
    """
    # Set default parameters
    if model_params is None:
        model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': 42
        }
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    
    # Enable autolog untuk sklearn
    mlflow.sklearn.autolog(log_models=True)
    
    with mlflow.start_run(run_name=f"RF_Autolog_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        print(f"[INFO] MLflow Run ID: {run.info.run_id}")
        
        # Preprocessing dengan StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Training model
        print("[INFO] Training RandomForestClassifier dengan autolog...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_scaled)
        
        # Metrics sudah di-log otomatis oleh autolog
        accuracy = accuracy_score(y_test, y_pred)
        print(f"[INFO] Accuracy: {accuracy:.4f}")
        
        # Simpan scaler sebagai artifact tambahan
        scaler_path = "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)
        
        # Buat dan log confusion matrix
        cm_path = "confusion_matrix.png"
        create_confusion_matrix_plot(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)
        
        # Log feature importance plot
        feature_importance_path = create_feature_importance_plot(
            model, X_train.columns.tolist(), "feature_importance.png"
        )
        mlflow.log_artifact(feature_importance_path)
        
        # Cleanup local files
        os.remove(scaler_path)
        os.remove(cm_path)
        os.remove(feature_importance_path)
        
        run_id = run.info.run_id
    
    # Disable autolog setelah selesai
    mlflow.sklearn.autolog(disable=True)
    
    return model, run_id


def create_feature_importance_plot(model, feature_names, save_path):
    """
    Membuat plot feature importance.
    
    Args:
        model: Trained model
        feature_names: List nama fitur
        save_path: Path untuk menyimpan plot
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices], color="steelblue", align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Feature importance plot saved to: {save_path}")
    return save_path


def main():
    """
    Fungsi utama untuk menjalankan training pipeline.
    """
    print("=" * 60)
    print("SMSML - Model Training dengan MLflow Autolog")
    print("=" * 60)
    print("Dataset: Breast Cancer Wisconsin (Diagnostic)")
    print("=" * 60)
    
    # Konfigurasi - Menggunakan Breast Cancer Dataset
    DATA_PATH = "data/breast_cancer_data.csv"
    TARGET_COLUMN = "target"
    EXPERIMENT_NAME = "SMSML_Dafis_Nadhif_Saputra"
    
    # Untuk lokal tracking (Basic criteria)
    mlflow.set_tracking_uri("mlruns")
    
    # Untuk DagsHub (Advanced criteria) - uncomment lines below
    # import dagshub
    # dagshub.init(repo_owner='YOUR_USERNAME', repo_name='YOUR_REPO', mlflow=True)
    
    # Load Breast Cancer dataset
    print("\n[INFO] Loading Breast Cancer Wisconsin Dataset...")
    df = load_preprocessed_data(DATA_PATH)
    
    print("\n[INFO] Dataset Info:")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Total features: {len(df.columns) - 1}")
    print(f"  - Target distribution:")
    print(f"    • Malignant (0): {(df['target'] == 0).sum()}")
    print(f"    • Benign (1): {(df['target'] == 1).sum()}")
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_features_target(
        df, TARGET_COLUMN, test_size=0.2, random_state=42
    )
    
    # Model parameters
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # Train model with autolog
    model, run_id = train_model_with_autolog(
        X_train, X_test, y_train, y_test,
        experiment_name=EXPERIMENT_NAME,
        model_params=model_params
    )
    
    print("\n" + "=" * 60)
    print("Training selesai!")
    print(f"MLflow Run ID: {run_id}")
    print("Untuk melihat MLflow UI, jalankan: mlflow ui")
    print("=" * 60)


if __name__ == "__main__":
    main()