import streamlit as st
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
import datetime
from sklearn.preprocessing import StandardScaler

# Define the target columns
target_cols = ['Headline_CPI', 'Alcoholic beverages and tobacco', 'Clothing and footwear',
              'Communication', 'Education', 'Food and non-alcoholic beverages',
              'Health', 'Household contents and services',
              'Housing and utilities', 'Miscellaneous goods and services',
              'Recreation and culture', 'Restaurants and hotels ', 'Transport']

# Function to load models for all categories
def load_models():
    loaded_models = {}
    for column in target_cols:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model
    return loaded_models

# Function to create input data for prediction
def create_input_data(selected_categories, previous_cpi_value,Total_Local_Sales, Total_Export_Sales, USD_ZAR, GBP_ZAR, EUR_ZAR):
    input_data = np.zeros((1, len(target_cols)))  # Create an empty array of the correct shape
    for category in selected_categories:
        input_data[0, target_cols.index(category)] = previous_cpi_value
        input_data[0, target_cols.index('Total_Local_Sales')] = Total_Local_Sales
        input_data[0, target_cols.index('Total_Export_Sales')] = Total_Export_Sales
        input_data[0, target_cols.index('USD/ZAR')] = USD_ZAR
        input_data[0, target_cols.index('GBP/ZAR')] = GBP_ZAR
        input_data[0, target_cols.index('EUR/ZAR')] = EUR_ZAR
    
    # Apply StandardScaler to scale the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    return input_data_scaled



def make_prediction(selected_categories, input_data, loaded_models, category_formatted, predictions, reference_date, selected_month):
    for category in selected_categories:
        for i in range(1, 4):
            model_key = f"{category}_month_{i}"
            if model_key in loaded_models:
                loaded_model = loaded_models[model_key]
                y_pred = loaded_model.predict(input_data)
                predictions[f'{category_formatted}_CPI_for_{reference_date.strftime("%B_%Y")}_{selected_month}'] = round(y_pred[0][0], 2)

# Streamlit app
def main():
    # Set the title
    st.title("CPI Prediction Dashboard")

    # Allow the user to select categories for prediction
    selected_categories = st.multiselect("Select categories to predict:", target_cols, default=target_cols[0])

    # Display input fields for previous CPI values
    previous_cpi_value = st.number_input("Enter previous CPI value:", value=0.0)

    # Display input fields for vehicle sales and currency
    st.write("Enter Vehicle Sales and Currency Input:")
    Total_Local_Sales = st.number_input("Total_Local Sales", value=0.0)
    Total_Export_Sales = st.number_input("Total_Export_Sales", value=0.0)
    USD_ZAR = st.number_input("USD/ZAR", value=0.0)
    GBP_ZAR = st.number_input("GBP/ZAR", value=0.0)
    EUR_ZAR = st.number_input("EUR/ZAR", value=0.0)

    # Load saved models
    loaded_models = load_models()

    # Allow the user to select which month they want to predict
    selected_month = st.selectbox("Select a month for prediction:", ["Next Month", "Two Months Later", "Three Months Later"])

    # Add a button to trigger model predictions
    if st.button("Predict CPI"):
        # Dictionary to store predictions
        predictions = {}

        # Calculate the reference date based on the current date
        current_date = datetime.date.today()
        if selected_month == "Next Month":
            reference_date = current_date.replace(month=current_date.month + 1)
        elif selected_month == "Two Months Later":
            reference_date = current_date.replace(month=current_date.month + 2)
        elif selected_month == "Three Months Later":
            reference_date = current_date.replace(month=current_date.month + 3)

        # Make predictions for the selected categories
        make_prediction(selected_categories, create_input_data(selected_categories, previous_cpi_value, Total_Local_Sales, Total_Export_Sales, USD_ZAR, GBP_ZAR, EUR_ZAR), loaded_models, "_".join(selected_categories), predictions, reference_date, selected_month)

        # Display predictions
        st.write(f"Predicted CPI values for {selected_month} for the selected categories:")
        for category in selected_categories:
            category_formatted = category.replace(' ', '_')  # Replace spaces with underscores
            st.write(f"{category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[category_formatted + '_CPI_for_' + reference_date.strftime('%B_%Y') + '_' + selected_month]:.2f}")

if __name__ == "__main__":
    main()

