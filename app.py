import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}

# Streamlit UI
st.title("Manoko's CPI Prediction App")

# User input
category = st.selectbox("Select Category", target_cols)

# Allow the user to select the number of months to predict into the future
num_months = st.number_input("Number of Months into the Future", min_value=1, value=1)

# Preprocess the input data (similar to what you did during training)
input_data = pd.read_csv('train.csv')

# Drop the first and second rows
input_data = input_data.drop([0, 1]).reset_index(drop=True)

# Add the code to calculate lagged features
feats_to_lag = input_data.columns[1:].to_list()
for col in feats_to_lag:
    if col != 'year_month':
        for i in range(1, 3):
            input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)

# Get the last available date in your dataset
last_date = pd.to_datetime(input_data['year_month'].iloc[-1])
st.write(last_date)
# Generate future dates starting from the last available date
future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, num_months + 1)]

# Predict for each future date
for future_date in future_dates:
    selected_month = future_date.month
    selected_year = future_date.year
    
    # Prepare input data for the future date (similar to the training data preprocessing)
    input_data_future = input_data.copy()  # Start with a copy of the last available data
    input_data_future['year_month'] = future_date.strftime("%Y-%m")
    
    # Calculate lagged features for the future date
    for col in feats_to_lag:
        if col != 'year_month':
            for i in range(1, 3):
                input_data_future[f"prev_{i}_month_{col}"] = input_data_future[col].shift(i)
    
    # Transform the input data using the scaler
    input_scaled = scaler.transform(input_data_future.drop(columns=['Month', 'year_month']))
    
    # Make predictions for the future date
    lr_model = model_dict[category]
    predicted_cpi = lr_model.predict(input_scaled)
    
    # Display the predicted CPI for the future date
    st.write(f"Predicted CPI for {category} in {selected_month}/{selected_year}: {predicted_cpi[0]:.2f}")
