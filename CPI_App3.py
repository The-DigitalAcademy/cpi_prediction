import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import datetime
from collections import defaultdict

# Define the target columns
target_cols = ['Headline_CPI', 'Alcoholic beverages and tobacco', 'Clothing and footwear',
              'Communication', 'Education', 'Food and non-alcoholic beverages',
              'Health', 'Household contents and services',
              'Housing and utilities', 'Miscellaneous goods and services',
              'Recreation and culture', 'Restaurants and hotels ', 'Transport']

# Create input data for prediction
def create_input_data(selected_category, input_values, vehicle_sales, currency_input):
    input_data = pd.DataFrame(columns=target_cols)  # Create an empty DataFrame
    input_data.at[0, selected_category] = input_values["previous_cpi_value"]
    input_data.at[0, 'Vehicle Sales'] = vehicle_sales
    input_data.at[0, 'Currency Input'] = currency_input

    # Set other features to 0 as placeholders, you can update this as needed
    for col in target_cols:
        if col != selected_category and col not in ['Vehicle Sales', 'Currency Input']:
            input_data.at[0, col] = 0.0  # You can update this with actual input values

    return input_data

# Make predictions for a selected category
def make_prediction(selected_category, input_data, loaded_models, category_formatted, predictions, reference_date):
    for i in range(1, 4):
        model_key = f"{selected_category}_month_{i}"
        if model_key in loaded_models:
            loaded_model = loaded_models[model_key]
            y_pred = loaded_model.predict(input_data)
            predictions[f'{category_formatted}_CPI_for_{reference_date.strftime("%B_%Y")}'] = round(y_pred[0][0], 2)

# Streamlit app
def main():
    # Set the title
    st.title("CPI Prediction Dashboard")

    # Display a dropdown to select the category for prediction
    selected_category = st.selectbox("Select a category to predict:", target_cols)

    # Display input fields for previous CPI values
    st.write(f"Enter previous CPI value for {selected_category}:")
    previous_cpi_value = st.number_input(f"Previous CPI for {selected_category}", value=0.0)

    # Display input fields for vehicle sales and currency
    vehicle_sales = st.number_input("Vehicle Sales", value=0.0)
    currency_input = st.number_input("Currency Input", value=0.0)

    # Store the input values for the selected category
    input_values = {
        "previous_cpi_value": previous_cpi_value,
        "vehicle_sales": vehicle_sales,
        "currency_input": currency_input
    }

    # Add a button to trigger model predictions
    if st.button("Predict CPI"):
        # Load models and make predictions for the selected category
        loaded_models = load_models(target_cols)  # Load your models here

        # Dictionary to store predictions
        predictions = {}

        # Allow the user to select which month they want to predict
        selected_month = st.selectbox("Select a month for prediction:", ["Next Month", "Two Months Later", "Three Months Later"])

        # Calculate the reference date based on the current date
        current_date = datetime.date.today()
        if selected_month == "Next Month":
            reference_date = current_date.replace(month=current_date.month + 1)
        elif selected_month == "Two Months Later":
            reference_date = current_date.replace(month=current_date.month + 2)
        elif selected_month == "Three Months Later":
            reference_date = current_date.replace(month=current_date.month + 3)

        # Make predictions for the selected category
        input_data = create_input_data(selected_category, input_values, vehicle_sales, currency_input)
        category_formatted = selected_category.replace(' ', '_')  # Replace spaces with underscores
        make_prediction(selected_category, input_data, loaded_models, category_formatted, predictions, reference_date)

        # Display predictions
        st.write(f"Predicted CPI values for {selected_month} for {selected_category}:")
        st.write(f"{selected_category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[category_formatted + '_CPI_for_' + reference_date.strftime('%B_%Y')]:.2f}")

if __name__ == "__main__":
    main()
