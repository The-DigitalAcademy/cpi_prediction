import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}

# Streamlit UI
st.title("Manoko's CPI Prediction App")

# User input
category = st.selectbox("Select Category", target_cols)

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

# Allow the user to select a specific month for prediction
selected_month = st.selectbox("Select Month", input_data['year_month'].unique())

# Prepare input data for the selected month (similar to the training data preprocessing)
input_data_selected = input_data[input_data['year_month'] == selected_month]

# Transform the input data using the scaler
input_scaled = scaler.transform(input_data_selected.drop(columns=['Month', 'year_month']))

# Lists to store data for plotting
predicted_cpi_values = []
previous_month_cpi_values = []

# Predict for the selected month
for target_col in target_cols:
    lr_model = model_dict[target_col]
    predicted_cpi = lr_model.predict(input_scaled)
    
    # Append predicted CPI to the list
    predicted_cpi_values.append(predicted_cpi[0])
    
    # Get the CPI value for the previous month
    previous_month = pd.to_datetime(selected_month) - pd.DateOffset(months=1)
    previous_month_data = input_data[input_data['year_month'] == previous_month.strftime("%Y-%m")]
    previous_month_cpi = previous_month_data[target_col].values[0]
    
    # Append previous month's CPI to the list
    previous_month_cpi_values.append(previous_month_cpi)
    
    # Display the predicted CPI for the selected month and category
    st.write(f"Predicted CPI for {target_col} in {selected_month}: {predicted_cpi[0]:.2f}")

# Create a bar graph to compare predicted CPI and previous month's CPI
plt.figure(figsize=(8, 4))
plt.bar([category], predicted_cpi_values, label="Predicted CPI", color='blue')
plt.bar([category], previous_month_cpi_values, label="Previous Month CPI", color='orange', alpha=0.7)
plt.xlabel("Category")
plt.ylabel("CPI Value")
plt.title(f"Comparison of Predicted CPI and Previous Month's CPI for {selected_month}")
plt.legend()
st.pyplot(plt)  # Display the plot in Streamlit
