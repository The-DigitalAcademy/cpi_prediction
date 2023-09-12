import streamlit as st
import pandas as pd
import os
from tensorflow.keras.models import load_model
import datetime

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
def create_input_data(selected_category, previous_cpi_value, vehicle_sales, USD_ZAR, GBP_ZAR, EUR_ZAR):
    input_data = pd.DataFrame(columns=target_cols)  # Create an empty DataFrame
    for category in selected_category:
        input_data.at[0, category] = previous_cpi_value
    input_data.at[[0, 'Vehicle Sales']] = vehicle_sales
    input_data.at[[0, 'USD/ZAR']] = USD_ZAR
    input_data.at[[0, 'GBP/ZAR']] = GBP_ZAR
    input_data.at[[0, 'EUR/ZAR']] = EUR_ZAR
    return input_data

# Function to make predictions for a category
def make_prediction(selected_category, input_data, loaded_models, category_formatted, predictions, reference_date, selected_month):
    for category in selected_category:
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
    selected_categories = st.multiselect("Select categories to predict:", target_cols, default=[target_cols[0]])

    # Display input fields for previous CPI values
    previous_cpi_value = st.number_input("Enter previous CPI value:", value=0.0)

    # Display input fields for vehicle sales and currency
    st.write("Enter Vehicle Sales and Currency Input:")
    vehicle_sales = st.number_input("Vehicle Sales", value=0.0)
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
        make_prediction(selected_categories, create_input_data(selected_categories, previous_cpi_value, vehicle_sales, USD_ZAR, GBP_ZAR, EUR_ZAR), loaded_models, "_".join(selected_categories), predictions, reference_date, selected_month)

        # Display predictions
        st.write(f"Predicted CPI values for {selected_month} for the selected categories:")
        for category in selected_categories:
            category_formatted = category.replace(' ', '_')  # Replace spaces with underscores
            st.write(f"{category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[category_formatted + '_CPI_for_' + reference_date.strftime('%B_%Y') + '_' + selected_month]:.2f}")

if __name__ == "__main__":
    main()
