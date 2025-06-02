import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("logistic_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Resignation Prediction", layout="centered")
st.title("💼 Employee Resignation Prediction App")

st.header("🔍 Input Employee Information")

# User input features
years_at_company = st.slider("Years at Company", 0, 40, 5)
performance_score = st.selectbox("Performance Score", [1, 2, 3, 4, 5])
monthly_salary = st.number_input("Monthly Salary", min_value=1000, max_value=30000, step=1000)
remote_freq = st.selectbox("Remote Work Frequency (%)", [0, 25, 50, 75, 100])
training_hours = st.slider("Training Hours", 0, 100, 20)
promotions = st.slider("Number of Promotions", 0, 10, 1)
satisfaction = st.slider("Employee Satisfaction Score", 0.0, 1.0, 0.5)
is_peak = st.selectbox("Peak Season (1=Yes, 0=No)", [0, 1])

# Prediction button
if st.button("Predict Resignation"):
    features = np.array([[years_at_company, performance_score, monthly_salary,
                          remote_freq, training_hours, promotions,
                          satisfaction, is_peak]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0][1]

    st.subheader("📊 Prediction Result")
    st.success("✅ Likely to Stay" if prediction == 0 else "⚠️ Likely to Resign")
    st.metric(label="Resignation Probability", value=f"{prob:.2%}")