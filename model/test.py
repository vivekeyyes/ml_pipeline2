import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
data_path = 'data/data.csv'
data = pd.read_csv(data_path)

# Assuming the data has columns "features" and "target"
X = data.drop(columns='target')  # features
y = data['target']  # target variable

# Load the saved model
model = joblib.load('models/trained_model.pkl')

# Predict and evaluate on the test set (you could also split into train/test here)
predictions = model.predict(X)

# Calculate accuracy (or other metrics)
accuracy = accuracy_score(y, predictions)

print(f"Model accuracy: {accuracy * 100:.2f}%")

# Optionally, save the results
with open('models/accuracy.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")

# Exit with status based on the test result (to gate model registration)
if accuracy < 0.8:  # Example threshold
    raise ValueError("Model accuracy is below acceptable threshold.")
