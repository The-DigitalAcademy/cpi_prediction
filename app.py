import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}

# Streamlit UI
st.title("Manoko's CPI Prediction App")

# User input - Select Month
selected_month = st.selectbox("Select Month", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"], index=3)  # Default to April (04)

# User input - Select Year
selected_year = st.selectbox("Select Year", ["2022", "2023"], index=1)  # Default to 2023

# Create a function to preprocess input data for prediction based on the selected month and year
def preprocess_input_data(selected_month, selected_year):
    # Load your input data (similar to what you did during training)
    input_data = pd.read_csv('train.csv')

    # Add the code to calculate lagged features
    feats_to_lag = input_data.columns[1:].to_list()
    for col in feats_to_lag:
        if col != 'year_month':
            for i in range(1, 3):
                input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)

    input_data.drop(0)
    input_data.bfill()

    # Drop columns that are not needed for prediction
    input_data = input_data.drop(columns=target_cols + ['Total_Local Sales', 'Total_Export_Sales'])

    # Filter data for the selected month and year
    selected_date = f"{selected_year}-{selected_month}"
    selected_data = input_data[input_data['year_month'] == selected_date]

    return selected_data

# Add a button to trigger predictions
if st.button("Predict CPI"):
    # Preprocess input data for the selected month and year
    input_data = preprocess_input_data(selected_month, selected_year)

    # Ensure that the selected month and year exist in the data
    if not input_data.empty:
        # Transform the input data using the scaler
        input_scaled = scaler.transform(input_data.drop(columns=['Month', 'year_month']))

        # Use your trained model to make predictions
        lr_model = model_dict[category]
        predicted_cpi = lr_model.predict(input_scaled)

        # Display the predicted CPI value to the user
        st.write(f"Predicted CPI for {category} in {selected_year}-{selected_month}: {predicted_cpi[0]:.2f}")
    else:
        st.write(f"No data available for {selected_year}-{selected_month}. Please select a different month and year.")
