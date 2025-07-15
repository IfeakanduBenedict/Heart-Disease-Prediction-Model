import warnings
import numpy as np
import base64
import streamlit as st
import pandas as pd
import joblib

# Suppress specific warnings
warnings.filterwarnings("ignore", message="In the future np.bool will be defined as the corresponding NumPy scalar.")
warnings.filterwarnings("ignore", message="The use_column_width parameter has been deprecated.*")
warnings.filterwarnings("ignore", message="Serialization of dataframe to Arrow table was unsuccessful.*")

# Fix for np.bool deprecation
if not hasattr(np, 'bool'):
    np.bool = bool

# Set Streamlit Page Config
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Load model
@st.cache_resource

def load_model():
    return joblib.load("heart_disease_model.pkl")

model = load_model()

# Theme selector
theme = st.sidebar.radio("Select Theme", ["Light", "Dark"])
if theme == "Dark":
    st.markdown("""
    <style>
    .stApp { background-color: #1E1E2F; color: white; }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp { background-color: #F7F7F7; color: black; }
    </style>
    """, unsafe_allow_html=True)

# App title and subtitle
st.title("🫀 Heart Disease Prediction")
st.markdown("""
Enter patient details to assess the likelihood of heart disease.
""")

# UI layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical, 1: Atypical, 2: Non-anginal, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Data formatting
input_data = pd.DataFrame({
    'age': [age],
    'sex': [1 if sex == "Male" else 0],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [1 if fbs == "Yes" else 0],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [1 if exang == "Yes" else 0],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [1 if thal == "Normal" else (2 if thal == "Fixed Defect" else 3)]
})

# Predict
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease. Probability: {probability:.2%}")
