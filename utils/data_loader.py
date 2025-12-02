"""
Data loading and preprocessing utilities for Breast Cancer Wisconsin dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(file_path='data/data.csv'):
    """
    Load and preprocess the Breast Cancer Wisconsin dataset.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        X_train, X_test, y_train, y_test: Preprocessed train/test splits
        feature_names: List of feature names
    """
    # Load dataset - try with headers first (Kaggle format)
    try:
        df = pd.read_csv(file_path)
        # Check if this is UCI format (no headers) by checking if first column is numeric
        # UCI format has numeric first column (ID), Kaggle has string header
        if str(df.columns[0]).replace('.', '').isdigit() or len(df.columns) == 32:
            # UCI format without headers
            df = pd.read_csv(file_path, header=None)
            # UCI format: ID (col 0), Diagnosis (col 1), then 30 features (cols 2-31)
            df = df.drop(0, axis=1)  # Drop ID column
            df.columns = ['diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
        else:
            # Kaggle format with headers
            if 'id' in df.columns.str.lower():
                id_col = [col for col in df.columns if col.lower() == 'id'][0]
                df = df.drop(id_col, axis=1)
            elif 'Unnamed: 0' in df.columns:
                df = df.drop('Unnamed: 0', axis=1)
    except:
        # Fallback: assume UCI format (no headers)
        df = pd.read_csv(file_path, header=None)
        df = df.drop(0, axis=1)  # Drop ID column
        df.columns = ['diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    
    # Encode diagnosis: M = 1, B = 0
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].values
    
    # Get feature names
    feature_names = X.columns.tolist()
    
    # Convert to numpy array
    X = X.values
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def download_dataset():
    """
    Download the Breast Cancer Wisconsin dataset from Kaggle.
    Note: This requires kaggle API credentials. Alternatively, user can download manually.
    """
    import urllib.request
    import os
    
    url = "https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/download"
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Please download the dataset manually from:")
    print("https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")
    print("And place it as 'data/data.csv' in the project root.")
    print("\nAlternatively, if you have kaggle API set up, you can use:")
    print("kaggle datasets download -d uciml/breast-cancer-wisconsin-data -p data/")
    print("Then unzip the file and rename to data.csv")


