import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved models
model_filenames = {
    'Alcoholic beverages and tobacco': 'Alcoholic beverages and tobacco_model.pkl',
    'Clothing and footwear': 'Clothing and footwear_model.pkl',
    'Communication': 'Communication_model.pkl',
    'Education': 'Education_model.pkl',
    'Food and non-alcoholic beverages': 'Food and non-alcoholic beverages_model.pkl',
    'Headline_CPI': 'Headline_CPI_model.pkl',
    'Health': 'Health_model.pkl',
    'Household contents and services': 'Household contents and services_model.pkl',
    'Housing and utilities': 'Housing and utilities_model.pkl',
    'Miscellaneous goods and services': 'Miscellaneous goods and services_model.pkl',
    'Recreation and culture': 'Recreation and culture_model.pkl',
    'Restaurants and hotels': 'Restaurants and hotels_model.pkl',
    'Transport': 'Transport_model.pkl'
    # Add model filenames for other columns...
}

lr_models = {}
for col, model_filename in model_filenames.items():
    lr_models[col] = joblib.load(model_filename)

# Create the Streamlit app
st.title('CPI Prediction for April')

# Input features
st.header('Input Features')

# Select category
selected_category = st.selectbox('Select Category', list(model_filenames.keys()))

# Select month
selected_month = st.text_input('Enter Month (YYYY-MM)', '2023-03')  # Default value for example

# Define a function to make predictions
@st.cache
def make_prediction(category, month):
    # Assuming you have a function or code to load your dataset and preprocess it
    # Replace this with your actual dataset loading and preprocessing code
    # dataset = load_and_preprocess_dataset()

    # Assuming you have a function or code to extract the lagged feature value
    # Replace this with your actual feature extraction code
    # lagged_feature = extract_lagged_feature(dataset, selected_category, selected_month)

    # Instead of the above lines, you can manually set a default value for the lagged feature
    # Here, we assume a default value of 0.0 for the example
    lagged_feature = 0.0

    # Use the selected model to make predictions
    lr_model = lr_models[category]
    user_inputs_scaled = scaler.transform([[lagged_feature]])
    predicted_cpi = lr_model.predict(user_inputs_scaled)[0]

    return predicted_cpi

# Make predictions
if st.button('Predict'):
    if selected_month and selected_category:
        predicted_cpi = make_prediction(selected_category, selected_month)
        st.success(f'Predicted CPI for {selected_category} in {selected_month}: {predicted_cpi:.2f}')
    else:
        st.warning('Please select a category and enter a valid month.')
