import streamlit as st
import pickle
import pandas as pd

# Load the linear regression model
model_filename = 'linear_regression_model.pkl'
with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

st.title('CPI Prediction for April')

# Input features
st.header('Input Features')
# You can add input fields here for user input (e.g., interest rates, unemployment rate, etc.)

# Make predictions
if st.button('Predict'):
    # Get user inputs
    # input_data = st.text_input('Input Feature 1', default_value)
    # ...
    # Process the input data and make predictions
    # predicted_cpi = model.predict([input_data])

    # For demonstration purposes, let's assume a sample prediction
    predicted_cpi = 150.0

    st.success(f'Predicted CPI for April: {predicted_cpi:.2f}')

