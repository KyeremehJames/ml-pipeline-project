# preprocess.py
"""
Data preprocessing pipeline for Iris dataset
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data():
    """Load Iris dataset from sklearn"""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['target'])
    return X, y, iris.target_names

def create_features(X):
    """Feature engineering"""
    X_engineered = X.copy()
    X_engineered['sepal_ratio'] = X_engineered['sepal length (cm)'] / X_engineered['sepal width (cm)']
    X_engineered['petal_ratio'] = X_engineered['petal length (cm)'] / X_engineered['petal width (cm)']
    X_engineered['sepal_area'] = X_engineered['sepal length (cm)'] * X_engineered['sepal width (cm)']
    X_engineered['petal_area'] = X_engineered['petal length (cm)'] * X_engineered['petal width (cm)']
    return X_engineered

def main():
    """Main preprocessing pipeline"""
    print("Loading Iris dataset...")
    X, y, target_names = load_data()
    
    print("Performing feature engineering...")
    X_engineered = create_features(X)
    
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv('data/processed/X_train.csv', index=False)
    pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False)
    y_test.to_csv('data/processed/y_test.csv', index=False)
    
    # Save preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print("Preprocessing completed!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")

if __name__ == "__main__":
    main()
