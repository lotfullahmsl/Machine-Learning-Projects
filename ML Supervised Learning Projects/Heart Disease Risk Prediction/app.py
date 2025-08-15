import streamlit as st
import joblib
import pandas as pd

# 1. Load your trained model
model = joblib.load('heart_disease_model.pkl')

st.title("Heart Disease Prediction")

# 2. Input fields for the user
BMI = st.number_input("BMI", 10, 50, 25)

Smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
AlcoholDrinking = st.selectbox("Do you drink alcohol?", ["No", "Yes"])
Stroke = st.selectbox("Have you ever had a stroke?", ["No", "Yes"])
PhysicalHealth = st.number_input("Number of physically unhealthy days in past 30 days", 0, 30, 0)
MentalHealth = st.number_input("Number of mentally unhealthy days in past 30 days", 0, 30, 0)
DiffWalking = st.selectbox("Do you have difficulty walking?", ["No", "Yes"])
Sex = st.selectbox("Sex", ["Female", "Male"])
Diabetic = st.selectbox("Diabetic?", ["No", "Yes"])
PhysicalActivity = st.selectbox("Do you do physical activity/exercise?", ["No", "Yes"])
GenHealth = st.selectbox("General health rating", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
SleepTime = st.number_input("Average sleep time (hours per day)", 0, 24, 7)
Asthma = st.selectbox("Do you have asthma?", ["No", "Yes"])
KidneyDisease = st.selectbox("Do you have kidney disease?", ["No", "Yes"])
SkinCancer = st.selectbox("Do you have skin cancer?", ["No", "Yes"])

# 3. Map friendly input to numeric values
GenHealthMapping = {"Poor":1, "Fair":2, "Good":3, "Very Good":4, "Excellent":5}

input_data = pd.DataFrame({
    'BMI':[BMI],
    'Smoking':[1 if Smoking=="Yes" else 0],
    'AlcoholDrinking':[1 if AlcoholDrinking=="Yes" else 0],
    'Stroke':[1 if Stroke=="Yes" else 0],
    'PhysicalHealth':[PhysicalHealth],
    'MentalHealth':[MentalHealth],
    'DiffWalking':[1 if DiffWalking=="Yes" else 0],
    'Sex':[1 if Sex=="Male" else 0],
    'Diabetic':[1 if Diabetic=="Yes" else 0],
    'PhysicalActivity':[1 if PhysicalActivity=="Yes" else 0],
    'GenHealth':[GenHealthMapping[GenHealth]],
    'SleepTime':[SleepTime],
    'Asthma':[1 if Asthma=="Yes" else 0],
    'KidneyDisease':[1 if KidneyDisease=="Yes" else 0],
    'SkinCancer':[1 if SkinCancer=="Yes" else 0]
})

# 4. Make prediction
if st.button("Predict Heart Disease"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High risk of Heart Disease")
    else:
        st.success("✅ Low risk of Heart Disease")
