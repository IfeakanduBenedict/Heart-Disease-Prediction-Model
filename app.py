import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('heart_disease_model.pkl')

# Title and description
st.title("Heart Disease Prediction")
st.markdown("""
Enter patient medical details to assess the likelihood of heart disease.
""")

# Input fields organized into columns for better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], help="0: Typical angina, 1: Atypical angina, 2: Non-anginal pain, 3: Asymptomatic")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])

with col2:
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=100)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, step=0.1, value=0.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert categorical data to numerical
data = pd.DataFrame({
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

# Prediction button
if st.button('Predict Heart Disease'):
    prediction = model.predict(data)[0]
    prediction_proba = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.error(f" High Risk of Heart Disease! Probability: {prediction_proba:.2%}")
    else:
        st.success(f" Low Risk of Heart Disease! Probability: {prediction_proba:.2%}")
