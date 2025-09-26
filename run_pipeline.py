# run_pipeline.py
"""
Main script to run the complete ML pipeline
"""

import subprocess
import sys

scripts = [
    ("preprocess.py", "Data Preprocessing"),
    ("train.py", "Model Training"), 
    ("evaluate.py", "Model Evaluation")
]

print("Starting ML Pipeline...")
print("=" * 50)

for script, description in scripts:
    print(f"\nRunning: {description}")
    print("-" * 30)
    
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"✓ {description} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        break

print("\n" + "=" * 50)
print("Pipeline completed! Start API with: uvicorn api:app --reload")
