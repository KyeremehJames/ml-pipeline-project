# train.py
"""
Model training script for Iris classification
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import json

def load_processed_data():
    """Load preprocessed training data"""
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv')
    y_test = pd.read_csv('data/processed/y_test.csv')
    return X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

def train_model(X_train, y_train):
    """Train RandomForest classifier"""
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    """Main training pipeline"""
    print("Loading processed data...")
    X_train, X_test, y_train, y_test = load_processed_data()
    
    print("Training RandomForest classifier...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    
    # Save model
    joblib.dump(model, 'models/iris_classifier.joblib')
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'model_type': 'RandomForest',
        'parameters': {'n_estimators': 100, 'max_depth': 5}
    }
    
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
