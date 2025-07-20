import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv('data/iris.csv')

# Preprocessing
df = df.drop(columns=['Id'])
df['Species'] = df['Species'].str.replace('Iris-', '', regex=False)

# Features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Train–test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=15, random_state=42)
model.fit(X_train, y_train)

# Ensure the 'model' folder exists
os.makedirs('model', exist_ok=True)

# Save the model
joblib.dump(model, 'model/iris_rf_model.pkl')

print("✅ Model trained and saved as 'model/iris_rf_model.pkl'")
