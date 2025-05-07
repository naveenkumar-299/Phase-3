# app.py

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load('student_model.pkl')

st.title("🎓 Student Performance Predictor")

# Input fields
age = st.slider("Age", 15, 22)
studytime = st.selectbox("Weekly Study Time (hours)", [1, 2, 3, 4])
failures = st.selectbox("Past Class Failures", [0, 1, 2, 3])
absences = st.slider("Number of Absences", 0, 100)
internet = st.selectbox("Internet Access", ["yes", "no"])
schoolsup = st.selectbox("School Support", ["yes", "no"])
famsup = st.selectbox("Family Support", ["yes", "no"])

# Preprocess inputs
input_dict = {
    'age': age,
    'studytime': studytime,
    'failures': failures,
    'absences': absences,
    'internet': 1 if internet == 'yes' else 0,
    'schoolsup': 1 if schoolsup == 'yes' else 0,
    'famsup': 1 if famsup == 'yes' else 0
}

input_df = pd.DataFrame([input_dict])

# Prediction
if st.button("Predict Performance"):
    prediction = model.predict(input_df)
    result = "Pass" if prediction[0] == 1 else "Fail"
    st.success(f"Predicted Performance: **{result}**")
