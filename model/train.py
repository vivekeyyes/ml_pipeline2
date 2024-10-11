import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Load the dataset directly from the file path (data.csv is now committed to Git)
data_path = 'data/data.csv'
data = pd.read_csv(data_path)

# Assuming the data has columns "features" and "target"
X = data.drop(columns='target')  # features
y = data['target']  # target variable

# Split the data into training and testing sets (adjust as necessary)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model (Logistic Regression in this case)
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

os.makedirs('models', exist_ok=True)

# Save the trained model
joblib.dump(model, 'models/trained_model.pkl')
print("Model training complete and saved.")
