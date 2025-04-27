import streamlit as st
import pickle
import numpy as np

# Load the saved model and scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

# App Title
st.title("❤️ Heart Disease Prediction App")

st.write("Enter patient details below to predict the likelihood of heart disease.")

# Input fields
age = st.number_input("Age", 20, 100)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250)
exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.number_input("ST depression", 0.0, 10.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])

# Manual encoding for label-encoded values
sex = 1 if sex == "Male" else 0
cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_map[cp]

fbs = 1 if fbs == "Yes" else 0

restecg_map = {
    "Normal": 0,
    "ST-T Wave Abnormality": 1,
    "Left Ventricular Hypertrophy": 2
}
restecg = restecg_map[restecg]

exang = 1 if exang == "Yes" else 0

slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_map[slope]

# Combine features into array
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope]])

# Scale input
input_scaled = scaler.transform(input_data)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ The patient is likely to have heart disease.")
    else:
        st.success("✅ The patient is unlikely to have heart disease.")
