"""
modelling_tuning.py - Model Training dengan MLflow Manual Logging
Proyek Akhir: Membangun Sistem Machine Learning
Dicoding Indonesia

Kriteria Skilled: Menggunakan manual logging dengan metrics yang sama dengan autolog

Author: Dafis Nadhif Saputra
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import mlflow
import mlflow.sklearn
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')


class MLflowManualLogger:
    """
    Class untuk melakukan manual logging ke MLflow dengan metrics 
    yang setara dengan autolog.
    """
    
    def __init__(self, experiment_name, tracking_uri=None, use_dagshub=False,
                 dagshub_username=None, dagshub_repo=None):
        """
        Inisialisasi logger.
        
        Args:
            experiment_name: Nama experiment di MLflow
            tracking_uri: URI untuk MLflow tracking (default: mlruns lokal)
            use_dagshub: Apakah menggunakan DagsHub untuk remote tracking
            dagshub_username: Username DagsHub
            dagshub_repo: Nama repository DagsHub
        """
        self.experiment_name = experiment_name
        
        if use_dagshub and dagshub_username and dagshub_repo:
            # Setup DagsHub untuk Advanced criteria
            import dagshub
            dagshub.init(repo_owner=dagshub_username, repo_name=dagshub_repo, mlflow=True)
            print(f"[INFO] Connected to DagsHub: {dagshub_username}/{dagshub_repo}")
        elif tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("mlruns")
        
        mlflow.set_experiment(experiment_name)
        self.run_id = None
        self.artifacts_dir = "mlflow_artifacts"
        os.makedirs(self.artifacts_dir, exist_ok=True)
    
    def log_params(self, params):
        """Log parameters ke MLflow."""
        for key, value in params.items():
            mlflow.log_param(key, value)
        print(f"[INFO] Logged {len(params)} parameters")
    
    def log_metrics(self, metrics, step=None):
        """Log metrics ke MLflow."""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
        print(f"[INFO] Logged {len(metrics)} metrics")
    
    def log_model_artifact(self, model, artifact_path="model"):
        """Log model sebagai artifact."""
        mlflow.sklearn.log_model(model, artifact_path)
        print(f"[INFO] Model logged to: {artifact_path}")
    
    def log_artifact_file(self, local_path, artifact_path=None):
        """Log file sebagai artifact."""
        mlflow.log_artifact(local_path, artifact_path)
        print(f"[INFO] Artifact logged: {local_path}")
    
    def log_figure(self, fig, filename):
        """Log matplotlib figure sebagai artifact."""
        filepath = os.path.join(self.artifacts_dir, filename)
        fig.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close(fig)
        mlflow.log_artifact(filepath)
        print(f"[INFO] Figure logged: {filename}")
        return filepath


def load_preprocessed_data(data_path):
    """Memuat data yang sudah dipreprocess."""
    print(f"[INFO] Memuat data dari: {data_path}")
    df = pd.read_csv(data_path)
    print(f"[INFO] Data shape: {df.shape}")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


def preprocess_for_training(df, target_column, test_size=0.2, random_state=42):
    """
    Preprocessing data untuk training.
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder (jika ada)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"[INFO] Target encoded. Classes: {label_encoder.classes_}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Konversi kembali ke DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    print(f"[INFO] Training samples: {len(X_train)}")
    print(f"[INFO] Test samples: {len(X_test)}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder


def calculate_all_metrics(y_true, y_pred, y_pred_proba=None):
    """
    Menghitung semua metrics yang diperlukan (setara dengan autolog).
    
    Returns:
        Dictionary berisi semua metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
        metrics[f'precision_class_{i}'] = p
        metrics[f'recall_class_{i}'] = r
        metrics[f'f1_class_{i}'] = f
    
    # ROC-AUC (jika binary classification dan probabilitas tersedia)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except Exception as e:
            print(f"[WARNING] Could not calculate ROC-AUC: {e}")
    
    return metrics


def create_training_plots(y_true, y_pred, y_pred_proba, feature_importances, 
                          feature_names, artifacts_dir):
    """
    Membuat dan menyimpan semua plot yang diperlukan.
    
    Returns:
        Dictionary berisi path ke semua plot
    """
    plots = {}
    
    # 1. Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    cm_path = os.path.join(artifacts_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    plots['confusion_matrix'] = cm_path
    
    # 2. Feature Importance
    if feature_importances is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        indices = np.argsort(feature_importances)[::-1]
        ax.bar(range(len(feature_importances)), feature_importances[indices], color='steelblue')
        ax.set_xticks(range(len(feature_importances)))
        ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        ax.set_title('Feature Importance')
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        fi_path = os.path.join(artifacts_dir, 'feature_importance.png')
        fig.savefig(fi_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        plots['feature_importance'] = fi_path
    
    # 3. ROC Curve (for binary classification)
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        roc_path = os.path.join(artifacts_dir, 'roc_curve.png')
        fig.savefig(roc_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        plots['roc_curve'] = roc_path
    
    # 4. Prediction Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(y_pred, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('Prediction Distribution')
    ax.set_xlabel('Predicted Class')
    ax.set_ylabel('Frequency')
    dist_path = os.path.join(artifacts_dir, 'prediction_distribution.png')
    fig.savefig(dist_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    plots['prediction_distribution'] = dist_path
    
    print(f"[INFO] Created {len(plots)} plots")
    return plots


def train_with_manual_logging(X_train, X_test, y_train, y_test, 
                              feature_names, experiment_name,
                              model_type='random_forest',
                              hyperparameter_tuning=True):
    """Training model dengan manual logging ke MLflow."""
    logger = MLflowManualLogger(experiment_name)
    artifacts_dir = logger.artifacts_dir
    
    # Definisikan model dan parameter
    if model_type == 'random_forest':
        base_model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        } if hyperparameter_tuning else {}
    elif model_type == 'gradient_boosting':
        base_model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        } if hyperparameter_tuning else {}
    else:
        base_model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        } if hyperparameter_tuning else {}
    
    run_name = f"{model_type}_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n[INFO] MLflow Run ID: {run.info.run_id}")
        
        # Log dataset info
        dataset_info = {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'n_classes': len(np.unique(y_train))
        }
        logger.log_params(dataset_info)
        
        # Hyperparameter tuning atau langsung training
        if hyperparameter_tuning and param_grid:
            print("[INFO] Melakukan hyperparameter tuning dengan GridSearchCV...")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            
            # Log best parameters
            for param_name, param_value in best_params.items():
                mlflow.log_param(f"best_{param_name}", param_value)
            
            mlflow.log_metric("cv_best_score", grid_search.best_score_)
            print(f"[INFO] Best CV Score: {grid_search.best_score_:.4f}")
            print(f"[INFO] Best Parameters: {best_params}")
        else:
            print("[INFO] Training model tanpa hyperparameter tuning...")
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        # Log model parameters
        model_params = best_model.get_params()
        for param_name, param_value in model_params.items():
            try:
                mlflow.log_param(f"model_{param_name}", param_value)
            except:
                pass  # Skip parameters yang tidak bisa di-log
        
        # Predictions
        y_pred = best_model.predict(X_test)
        y_pred_proba = None
        if hasattr(best_model, 'predict_proba'):
            y_pred_proba = best_model.predict_proba(X_test)
        
        # Calculate and log metrics
        metrics = calculate_all_metrics(y_test, y_pred, y_pred_proba)
        logger.log_metrics(metrics)
        
        print("\n[INFO] Training Metrics:")
        for metric_name, metric_value in metrics.items():
            print(f"  - {metric_name}: {metric_value:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_mean_accuracy", cv_scores.mean())
        mlflow.log_metric("cv_std_accuracy", cv_scores.std())
        
        # Feature importance (jika tersedia)
        feature_importances = None
        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            
            # Log feature importance sebagai JSON
            fi_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importances)}
            fi_path = os.path.join(artifacts_dir, 'feature_importance.json')
            with open(fi_path, 'w') as f:
                json.dump(fi_dict, f, indent=2)
            mlflow.log_artifact(fi_path)
        
        # Create and log plots
        plots = create_training_plots(
            y_test, y_pred, y_pred_proba, 
            feature_importances, feature_names, artifacts_dir
        )
        for plot_name, plot_path in plots.items():
            mlflow.log_artifact(plot_path)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model, 
            "model",
            registered_model_name=f"SMSML_{model_type}"
        )
        
        # Log classification report sebagai artifact
        report = classification_report(y_test, y_pred, output_dict=True)
        report_path = os.path.join(artifacts_dir, 'classification_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact(report_path)
        
        # Log tags
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("author", "Dafis Nadhif Saputra")
        mlflow.set_tag("stage", "training")
        
        run_id = run.info.run_id
    
    return best_model, run_id


def main():
    """Fungsi utama untuk menjalankan training dengan manual logging."""
    print("=" * 70)
    print("SMSML - Model Training dengan MLflow Manual Logging (Skilled Criteria)")
    print("=" * 70)
    
    # Konfigurasi
    # Konfigurasi
    DATA_PATH = "data/breast_cancer_preprocessed.csv"
    TARGET_COLUMN = "target"
    EXPERIMENT_NAME = "SMSML_Dafis_Nadhif_Saputra_Manual"
    
    # Untuk lokal tracking
    mlflow.set_tracking_uri("mlruns")
    
    # Load data
    if os.path.exists(DATA_PATH):
        df = load_preprocessed_data(DATA_PATH)
        feature_names = [col for col in df.columns if col != TARGET_COLUMN]
    else:
        # Fallback to dummy data if file not found (for demonstration only)
        print(f"[WARNING] File {DATA_PATH} not found. Using dummy data for demo.")
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        y = (X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.5) > 0
        y = y.astype(int)
        feature_names = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
    
    # Preprocessing
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_for_training(
        df, 'target', test_size=0.2, random_state=42
    )
    
    # Training dengan berbagai model
    models_to_train = ['random_forest', 'gradient_boosting']
    
    for model_type in models_to_train:
        print(f"\n{'='*70}")
        print(f"Training {model_type.upper()}")
        print('='*70)
        
        model, run_id = train_with_manual_logging(
            X_train, X_test, y_train, y_test,
            feature_names=feature_names,
            experiment_name=EXPERIMENT_NAME,
            model_type=model_type,
            hyperparameter_tuning=True
        )
        
        print(f"\n[SUCCESS] {model_type} training completed!")
        print(f"Run ID: {run_id}")
    
    # Simpan model terbaik dan scaler
    print("\n[INFO] Menyimpan artifacts tambahan...")
    joblib.dump(scaler, 'scaler_final.joblib')
    print("[INFO] Scaler saved to: scaler_final.joblib")
    
    print("\n" + "=" * 70)
    print("Semua training selesai!")
    print("Untuk melihat MLflow UI, jalankan: mlflow ui")
    print("=" * 70)


if __name__ == "__main__":
    main()
