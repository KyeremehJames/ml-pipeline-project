# api.py
"""
FastAPI REST API for Iris classification
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

app = FastAPI(title="Iris Classification API")

# Load model and scaler
model = joblib.load('models/iris_classifier.joblib')
scaler = joblib.load('models/scaler.joblib')

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"message": "Iris Classification API is running!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: IrisFeatures):
    # Create DataFrame with correct column names
    X = pd.DataFrame([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]], columns=['sepal length (cm)', 'sepal width (cm)', 
                 'petal length (cm)', 'petal width (cm)'])
    
    # Apply feature engineering
    X['sepal_ratio'] = X['sepal length (cm)'] / X['sepal width (cm)']
    X['petal_ratio'] = X['petal length (cm)'] / X['petal width (cm)']
    X['sepal_area'] = X['sepal length (cm)'] * X['sepal width (cm)']
    X['petal_area'] = X['petal length (cm)'] * X['petal width (cm)']
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    class_names = ['setosa', 'versicolor', 'virginica']
    
    return {
        "prediction": int(prediction),
        "class_name": class_names[prediction],
        "confidence": float(probability[prediction]),
        "probabilities": {
            class_names[i]: float(prob) for i, prob in enumerate(probability)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
