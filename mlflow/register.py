import mlflow
import mlflow.sklearn
import joblib
import os

# Load the trained model
model_path = 'models/trained_model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model = joblib.load(model_path)

# Load the accuracy from the test run
accuracy_file = 'models/accuracy.txt'
if not os.path.exists(accuracy_file):
    raise FileNotFoundError(f"Accuracy file not found at {accuracy_file}")

with open(accuracy_file, 'r') as f:
    accuracy_info = f.read().strip()

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the accuracy info
    mlflow.log_param("accuracy_info", accuracy_info)

    # Log the model to MLflow (artifact of the run)
    mlflow.sklearn.log_model(model, "model")
    print(f"Model logged in MLflow under run ID: {run.info.run_id}")

    # Register the model in the MLflow Model Registry
    model_uri = f"runs:/{run.info.run_id}/model"  # URI of the logged model
    model_name = "MyModel"  # The name to use in the Model Registry

    mlflow.register_model(model_uri=model_uri, name=model_name)
    print(f"Model registered in MLflow Model Registry under name: {model_name}")
