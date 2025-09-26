# ML Pipeline Project Assignment

A  machine learning pipeline for Iris flower classification.

## Project Structure

ml-pipeline-project/
├── preprocess.py # Data preprocessing
├── train.py # Model training
├── evaluate.py # Model evaluation
├── api.py # FastAPI deployment
├── requirements.txt # Dependencies
└── monitoring.md # Monitoring strategies
text


## Quick Start

1. **Install dependencies**:
pip install -r requirements.txt

    Run complete pipeline:

bash

python run_pipeline.py
    Start API server:

bash

uvicorn api:app --reload


API Endpoints

   GET /health - Service health check

  POST /predict - Make predictions
    
