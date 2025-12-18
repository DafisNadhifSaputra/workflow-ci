"""
Dafis-Nadhif-Saputra.py - Automated Preprocessing Script
Proyek Akhir: Membangun Sistem Machine Learning
Dicoding Indonesia

Kriteria Skilled: File automate untuk preprocessing secara otomatis

Author: Dafis Nadhif Saputra
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Class untuk melakukan preprocessing data secara otomatis.
    Menghasilkan data yang siap untuk training model.
    """
    
    def __init__(self, config=None):
        """
        Inisialisasi preprocessor dengan konfigurasi.
        
        Args:
            config: Dictionary konfigurasi preprocessing
        """
        self.config = config or self._default_config()
        self.scaler = None
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = None
        self.target_column = None
        
    def _default_config(self):
        """Konfigurasi default."""
        return {
            'scaling_method': 'standard',  # 'standard', 'minmax', or None
            'handle_missing': 'mean',  # 'mean', 'median', 'mode', 'drop'
            'encode_categorical': True,
            'remove_outliers': False,
            'outlier_threshold': 3,  # IQR multiplier
            'test_size': 0.2,
            'random_state': 42
        }
    
    def load_data(self, file_path):
        """
        Memuat data dari file.
        
        Args:
            file_path: Path ke file data (CSV, Excel, dll)
        
        Returns:
            DataFrame
        """
        logger.info(f"Loading data from: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            df = pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_extension == '.json':
            df = pd.read_json(file_path)
        elif file_extension == '.parquet':
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def explore_data(self, df):
        """
        Eksplorasi data untuk memahami struktur dan kualitas data.
        
        Args:
            df: DataFrame
        
        Returns:
            Dictionary berisi hasil eksplorasi
        """
        logger.info("Exploring data...")
        
        exploration = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numerical_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
            'unique_values': {col: df[col].nunique() for col in df.columns}
        }
        
        logger.info(f"Numerical columns: {len(exploration['numerical_columns'])}")
        logger.info(f"Categorical columns: {len(exploration['categorical_columns'])}")
        logger.info(f"Total missing values: {df.isnull().sum().sum()}")
        
        return exploration
    
    def handle_missing_values(self, df):
        """
        Menangani missing values.
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame dengan missing values yang sudah ditangani
        """
        logger.info("Handling missing values...")
        df = df.copy()
        
        if self.config['handle_missing'] == 'drop':
            initial_rows = len(df)
            df = df.dropna()
            logger.info(f"Dropped {initial_rows - len(df)} rows with missing values")
        else:
            # Handle numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    if self.config['handle_missing'] == 'mean':
                        imputer = SimpleImputer(strategy='mean')
                    elif self.config['handle_missing'] == 'median':
                        imputer = SimpleImputer(strategy='median')
                    else:
                        imputer = SimpleImputer(strategy='most_frequent')
                    
                    df[col] = imputer.fit_transform(df[[col]])
                    self.imputers[col] = imputer
            
            # Handle categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
        
        logger.info(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_features(self, df, target_column=None):
        """
        Encoding fitur kategorikal.
        
        Args:
            df: DataFrame
            target_column: Nama kolom target (tidak di-encode jika diberikan)
        
        Returns:
            DataFrame dengan fitur kategorikal yang ter-encode
        """
        logger.info("Encoding categorical features...")
        df = df.copy()
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target column from encoding if specified
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return df
    
    def remove_outliers(self, df, target_column=None):
        """
        Menghapus outliers menggunakan IQR method.
        
        Args:
            df: DataFrame
            target_column: Nama kolom target (tidak dihapus outliernya)
        
        Returns:
            DataFrame tanpa outliers
        """
        logger.info("Removing outliers...")
        df = df.copy()
        initial_rows = len(df)
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        threshold = self.config['outlier_threshold']
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"Removed {initial_rows - len(df)} outlier rows")
        return df
    
    def scale_features(self, df, target_column=None):
        """
        Scaling fitur numerik.
        
        Args:
            df: DataFrame
            target_column: Nama kolom target (tidak di-scale)
        
        Returns:
            DataFrame dengan fitur yang sudah di-scale
        """
        logger.info(f"Scaling features using {self.config['scaling_method']} method...")
        df = df.copy()
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        if self.config['scaling_method'] == 'standard':
            self.scaler = StandardScaler()
        elif self.config['scaling_method'] == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            logger.info("No scaling applied")
            return df
        
        df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        self.feature_names = numerical_cols
        
        return df
    
    def preprocess(self, df, target_column):
        """
        Pipeline preprocessing lengkap.
        
        Args:
            df: DataFrame
            target_column: Nama kolom target
        
        Returns:
            DataFrame yang sudah dipreprocess
        """
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline...")
        logger.info("=" * 50)
        
        self.target_column = target_column
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Encode categorical features
        if self.config['encode_categorical']:
            df = self.encode_categorical_features(df, target_column)
        
        # Step 3: Remove outliers (optional)
        if self.config['remove_outliers']:
            df = self.remove_outliers(df, target_column)
        
        # Step 4: Scale features
        if self.config['scaling_method']:
            df = self.scale_features(df, target_column)
        
        logger.info("=" * 50)
        logger.info(f"Preprocessing complete. Final shape: {df.shape}")
        logger.info("=" * 50)
        
        return df
    
    def get_train_test_split(self, df, target_column):
        """
        Split data menjadi training dan testing sets.
        
        Args:
            df: DataFrame yang sudah dipreprocess
            target_column: Nama kolom target
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target if needed
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
            self.label_encoders['target'] = target_encoder
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config['test_size'],
            random_state=self.config['random_state'],
            stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def save_preprocessed_data(self, df, output_path):
        """
        Menyimpan data yang sudah dipreprocess.
        
        Args:
            df: DataFrame yang sudah dipreprocess
            output_path: Path untuk menyimpan file
        """
        df.to_csv(output_path, index=False)
        logger.info(f"Preprocessed data saved to: {output_path}")
    
    def save_artifacts(self, output_dir):
        """
        Menyimpan artifacts preprocessing (scaler, encoders, dll).
        
        Args:
            output_dir: Directory untuk menyimpan artifacts
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.scaler:
            scaler_path = os.path.join(output_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to: {scaler_path}")
        
        if self.label_encoders:
            encoders_path = os.path.join(output_dir, 'label_encoders.joblib')
            joblib.dump(self.label_encoders, encoders_path)
            logger.info(f"Label encoders saved to: {encoders_path}")
        
        if self.imputers:
            imputers_path = os.path.join(output_dir, 'imputers.joblib')
            joblib.dump(self.imputers, imputers_path)
            logger.info(f"Imputers saved to: {imputers_path}")
        
        # Save config
        config_path = os.path.join(output_dir, 'preprocessing_config.joblib')
        joblib.dump(self.config, config_path)
        logger.info(f"Config saved to: {config_path}")


def preprocess_data(input_path, output_path, target_column, config=None):
    """
    Fungsi utama untuk preprocessing data dari input ke output.
    
    Args:
        input_path: Path ke file input
        output_path: Path untuk menyimpan hasil
        target_column: Nama kolom target
        config: Konfigurasi preprocessing (optional)
    
    Returns:
        X_train, X_test, y_train, y_test - Data siap latih
    """
    preprocessor = DataPreprocessor(config)
    
    # Load data
    df = preprocessor.load_data(input_path)
    
    # Explore data
    exploration = preprocessor.explore_data(df)
    
    # Preprocess
    df_preprocessed = preprocessor.preprocess(df, target_column)
    
    # Save preprocessed data
    preprocessor.save_preprocessed_data(df_preprocessed, output_path)
    
    # Get train-test split
    X_train, X_test, y_train, y_test = preprocessor.get_train_test_split(
        df_preprocessed, target_column
    )
    
    # Save artifacts
    artifacts_dir = os.path.dirname(output_path) or '.'
    artifacts_dir = os.path.join(artifacts_dir, 'preprocessing_artifacts')
    preprocessor.save_artifacts(artifacts_dir)
    
    return X_train, X_test, y_train, y_test


def main():
    """Fungsi main untuk menjalankan preprocessing dari command line."""
    parser = argparse.ArgumentParser(
        description='Automated Data Preprocessing untuk SMSML'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Path ke file input data')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Path untuk menyimpan data yang sudah dipreprocess')
    parser.add_argument('--target', '-t', type=str, required=True,
                        help='Nama kolom target')
    parser.add_argument('--scaling', type=str, default='standard',
                        choices=['standard', 'minmax', 'none'],
                        help='Metode scaling (default: standard)')
    parser.add_argument('--missing', type=str, default='mean',
                        choices=['mean', 'median', 'mode', 'drop'],
                        help='Metode handling missing values (default: mean)')
    parser.add_argument('--remove-outliers', action='store_true',
                        help='Hapus outliers dari data')
    
    args = parser.parse_args()
    
    config = {
        'scaling_method': args.scaling if args.scaling != 'none' else None,
        'handle_missing': args.missing,
        'encode_categorical': True,
        'remove_outliers': args.remove_outliers,
        'outlier_threshold': 3,
        'test_size': 0.2,
        'random_state': 42
    }
    
    X_train, X_test, y_train, y_test = preprocess_data(
        args.input, args.output, args.target, config
    )
    
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE!")
    print("=" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
