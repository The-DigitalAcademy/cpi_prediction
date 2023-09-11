import streamlit as st
import pandas as pd
import os
from tensorflow.keras.models import load_model
from datetime import datetime

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

    # Allow the user to select the prediction month (relative to the system date)
    prediction_month = st.selectbox("Select the prediction month:", ["Next Month", "Two Months Ahead", "Three Months Ahead"])

    # Get the current system date
    current_date = datetime.now()

    # Calculate the reference date based on the user's selection
    if prediction_month == "Next Month":
        reference_date = current_date.replace(day=1) + pd.DateOffset(months=1)
    elif prediction_month == "Two Months Ahead":
        reference_date = current_date.replace(day=1) + pd.DateOffset(months=2)
    elif prediction_month == "Three Months Ahead":
        reference_date = current_date.replace(day=1) + pd.DateOffset(months=3)

    # Add a prediction button
    if st.button("Predict CPI"):
        # Dictionary to store loaded models
        loaded_models = {}

        # Iterate over selected categories
        for column in selected_categories:
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{reference_date.month}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{reference_date.month}"] = loaded_model

        # Create input data for prediction
        input_data = pd.DataFrame(columns=target_cols)  # Create an empty DataFrame
        for category in selected_categories:
            input_data.at[0, category] = input_values[category]
        input_data.at[0, 'Vehicle Sales'] = vehicle_sales
        input_data.at[0, 'Currency Input'] = currency_input

        # Dictionary to store predictions
        predictions = {}

        # Iterate over selected categories
        for column in selected_categories:
            model_key = f"{column}_month_{reference_date.month}"
            if model_key in loaded_models:
                loaded_model = loaded_models[model_key]
                y_pred = loaded_model.predict(input_data)
                predictions[f'{column} CPI for {reference_date.strftime("%B %Y")}'] = round(y_pred[0][0], 2)

        # Display predictions
        st.write("Predicted CPI values:")
        for category in selected_categories:
            st.write(f"{category} CPI for {reference_date.strftime('%B %Y')}: {predictions[f'{category} CPI for {reference_date.strftime('%B %Y')}']:.2f}")

if __name__ == "__main__":
    main()
