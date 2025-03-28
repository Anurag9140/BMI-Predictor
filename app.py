import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.markdown("""
    <style>
        
        .main-title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #ff3633;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            width: 100%;
            font-size: 30px;
        }
        
    </style>
""", unsafe_allow_html=True)

st.title("BMI Predictor")
st.write("Enter your details to predict your BMI")


age = st.number_input("Age", min_value=1, max_value=100, value=25)
height = st.number_input("Height (in cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (in kg)", min_value=10, max_value=200, value=70)

if st.button("Predict BMI"):
    input_data = np.array([[age, height, weight]])
    input_scaled = scaler.transform(input_data)
    bmi_prediction = model.predict(input_scaled)[0]
    st.success(f"Predicted BMI: {bmi_prediction:.2f}")
