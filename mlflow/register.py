import mlflow
import mlflow.sklearn
import joblib
import os

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow server URI if needed

# Define model and accuracy file paths
model_path = 'models/trained_model.pkl'
accuracy_file = 'models/accuracy.txt'

# Check if the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model
model = joblib.load(model_path)

# Check if the accuracy file exists
if not os.path.exists(accuracy_file):
    raise FileNotFoundError(f"Accuracy file not found at {accuracy_file}")

# Load the accuracy from the test run
with open(accuracy_file, 'r') as f:
    accuracy_info = f.read().strip()

# Start an MLflow run
with mlflow.start_run() as run:
    # Log the accuracy info as a parameter
    mlflow.log_param("accuracy_info", accuracy_info)

    # Log the model as an artifact of the run
    mlflow.sklearn.log_model(model, "model")
    print(f"Model logged in MLflow under run ID: {run.info.run_id}")

    # Register the model in the MLflow Model Registry
    model_uri = f"runs:/{run.info.run_id}/model"  # URI of the logged model
    model_name = "MyModel"  # The name to use in the Model Registry

    # Register the model in the Model Registry
    try:
        mlflow.register_model(model_uri=model_uri, name=model_name)
        print(f"Model registered in MLflow Model Registry under name: {model_name}")
    except Exception as e:
        print(f"Error registering model: {e}")
