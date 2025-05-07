# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("student_data.csv")  # Replace with your dataset path

# Basic preprocessing
data = data.dropna()
label_enc = LabelEncoder()

# Encode categorical columns
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = label_enc.fit_transform(data[col])

# Feature selection
X = data.drop('G3', axis=1)  # Assuming 'G3' is the target
y = data['G3']
y = y.apply(lambda grade: 1 if grade >= 10 else 0)  # Binary classification

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'student_model.pkl')
