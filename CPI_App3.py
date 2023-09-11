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

    # Dictionary to store loaded models
    loaded_models = {}
    input_values = {}
    # Iterate over target columns and months
    for column in target_cols:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model

    # Create input data for prediction
    input_data = pd.DataFrame(columns=target_cols)  # Create an empty DataFrame
    input_data.at[0, selected_category] = previous_cpi_value
    input_data.at[0, 'Vehicle Sales'] = vehicle_sales
    input_data.at[0, 'Currency Input'] = currency_input

      # Store the input values for the selected category
    input_values[selected_category] = {
            "previous_cpi_value": previous_cpi_value,
            "vehicle_sales": vehicle_sales,
            "currency_input": currency_input
        }


    # Add a button to trigger model predictions
    if st.button("Predict CPI"):
        # Load models and make predictions for each category
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

        # Iterate over selected categories and make predictions
        for selected_category in selected_categories:
            input_data = create_input_data(selected_category, input_values[selected_category])
            category_formatted = selected_category.replace(' ', '_')  # Replace spaces with underscores
            make_prediction(selected_category, input_data, loaded_models, category_formatted, predictions, reference_date)

        # Display predictions
        st.write(f"Predicted CPI values for {selected_month}:")
        for selected_category in selected_categories:
            category_formatted = selected_category.replace(' ', '_')  # Replace spaces with underscores
            st.write(f"{selected_category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[category_formatted + '_CPI_for_' + reference_date.strftime('%B_%Y')]:.2f}")

if __name__ == "__main__":
    main()
