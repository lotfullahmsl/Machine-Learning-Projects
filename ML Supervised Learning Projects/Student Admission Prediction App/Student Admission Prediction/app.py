import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load scaler and model
scaler = joblib.load('scaler.pkl')
model = joblib.load('knn_model.pkl')

# List of feature names used for training (replace with your actual columns)
feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA', 'Research']

st.title("Graduate Admission Chance Predictor")

# Create empty dict to hold user inputs
user_input = {}

# Dynamically generate inputs for each feature
for feature in feature_names:
    if feature in ['GRE Score']:
        user_input[feature] = st.number_input(feature, min_value=260, max_value=340, value=300)
    elif feature in ['TOEFL Score']:
        user_input[feature] = st.number_input(feature, min_value=0, max_value=120, value=100)
    elif feature in ['University Rating']:
        user_input[feature] = st.selectbox(feature, options=[1, 2, 3, 4, 5])
    elif feature in ['SOP', 'LOR ']:
        user_input[feature] = st.slider(feature, 1.0, 5.0, 3.0)
    elif feature == 'CGPA':
        user_input[feature] = st.number_input(feature, min_value=0.0, max_value=10.0, value=8.0, step=0.01)
    elif feature == 'Research':
        user_input[feature] = st.selectbox(feature, options=[0, 1])

# Convert to DataFrame (1 row)
input_df = pd.DataFrame([user_input])

# Check if number of columns matches scaler
if input_df.shape[1] != scaler.n_features_in_:
    st.error(f"Expected {scaler.n_features_in_} features but got {input_df.shape[1]}")
else:
    # Scale input
    input_scaled = scaler.transform(input_df)

    if st.button('Predict Admission Chance'):
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Admission Chance: {100 * prediction[0]:.2f}%")
