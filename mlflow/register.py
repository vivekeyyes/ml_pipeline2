import mlflow
import mlflow.sklearn
import joblib
import os

# Load the trained model
model_path = 'models/trained_model.pkl'
model = joblib.load(model_path)

# Load the accuracy from the test run
with open('models/accuracy.txt', 'r') as f:
    accuracy_info = f.read().strip()

# Start an MLflow run
with mlflow.start_run():
    # Log the accuracy info
    mlflow.log_param("accuracy_info", accuracy_info)

    # Register the model
    mlflow.sklearn.log_model(model, "model")
    print("Model registered in MLflow.")

# You can also include further tracking/logging if needed
