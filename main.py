import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained models
loaded_models = {}
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
               'Communication', 'Education', 'Food and non-alcoholic beverages',
               'Headline_CPI', 'Health', 'Household contents and services',
               'Housing and utilities', 'Miscellaneous goods and services',
               'Recreation and culture', 'Restaurants and hotels ', 'Transport']

for target_col in target_cols:
    loaded_models[target_col] = joblib.load(f"{target_col}_model.pkl")

# Create Streamlit app
st.title("CPI Prediction App")

# User-friendly input fields
selected_category = st.selectbox("Select Category", target_cols)
selected_month = st.slider("Select Month", min_value=1, max_value=12)
selected_year = st.slider("Select Year", min_value= 2022, max_value= 2023)  # Define min_year and max_year

# Define a function to make predictions based on user input
@st.cache  # Caching the function for improved performance
def make_prediction(category, month, year):
    # Extract the corresponding model for the selected category
    lr_model = loaded_models[category]
    
    # You can generate synthetic data based on user input if you don't load the entire dataset
    # Here's a placeholder for data generation:
    user_input_data = pd.DataFrame(data={'Month': [f'{year}-{month:02d}-30'], 'Category': [category]})
    
    # Make predictions (Replace this part with your data preprocessing and prediction code)
    prediction = 0.8
    
    return prediction

# Display the prediction
if st.button("Predict CPI"):
    prediction = make_prediction(selected_category, selected_month, selected_year)
    st.write(f"Predicted CPI for {selected_category} in {selected_year}-{selected_month:02d} is {prediction:.2f}")

# Optionally, display other information or visualizations
