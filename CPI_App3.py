import streamlit as st
import pandas as pd
import os
from tensorflow.keras.models import load_model

target_cols = ['Headline_CPI', 'Alcoholic beverages and tobacco', 'Clothing and footwear',
              'Communication', 'Education', 'Food and non-alcoholic beverages',
              'Health', 'Household contents and services',
              'Housing and utilities', 'Miscellaneous goods and services',
              'Recreation and culture', 'Restaurants and hotels ', 'Transport']

# Streamlit app
def main():
    # Set the title
    st.title("CPI Prediction Dashboard")

    # Allow the user to select multiple categories for prediction
    selected_categories = st.multiselect("Select categories to predict:", target_cols)

    # Display input fields for previous CPI values for selected categories
    input_values = {}
    for category in selected_categories:
        st.write(f"Enter previous CPI value for {category}:")
        previous_cpi_value = st.number_input(f"Previous CPI for {category}", value=0.0)
        input_values[category] = previous_cpi_value

    # Display input fields for vehicle sales and currency
    vehicle_sales = st.number_input("Vehicle Sales", value=0.0)
    currency_input = st.number_input("Currency Input", value=0.0)

    # Dictionary to store loaded models
    loaded_models = {}

    # Iterate over selected categories and months
    for column in selected_categories:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model

    # Create input data for prediction
    input_data = pd.DataFrame(columns=target_cols)  # Create an empty DataFrame
    for category in selected_categories:
        input_data.at[0, category] = input_values[category]
    input_data.at[0, 'Vehicle Sales'] = vehicle_sales
    input_data.at[0, 'Currency Input'] = currency_input

    # Dictionary to store predictions
    predictions = {}

    # Iterate over selected categories and months
    for column in selected_categories:
        for i in range(1, 4):
            model_key = f"{column}_month_{i}"
            if model_key in loaded_models:
                loaded_model = loaded_models[model_key]
                y_pred = loaded_model.predict(input_data)
                predictions[f'next_{i}_month_{column}'] = round(y_pred[0][0], 2)

    # Display predictions
    st.write("Predicted CPI values for the next 3 months:")
    for i in range(1, 4):
        for category in selected_categories:
            st.write(f"Month {i}: {category} CPI: {predictions[f'next_{i}_month_{category}']:.2f}")

if __name__ == "__main__":
    main()
