import streamlit as st
import pandas as pd
import os
import numpy as np
from tensorflow.keras.models import load_model
import datetime
from sklearn.preprocessing import StandardScaler

# Define the target columns with modified names
target_cols = ['Headline_CPI', 'Alcoholic_beverages_and_tobacco', 'Clothing_and_footwear',
              'Communication', 'Education', 'Food_and_non_alcoholic_beverages',
              'Health', 'Household_contents_and_services',
              'Housing_and_utilities', 'Miscellaneous_goods_and_services',
              'Recreation_and_culture', 'Restaurants_and_hotels', 'Transport']

# Function to load models for all categories
def load_models():
    loaded_models = {}
    for column in target_cols:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep_Neural_Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model
    return loaded_models

def create_input_data(selected_categories, previous_cpi_value, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar):
    input_data = np.zeros((1, len(target_cols)))  # Create an empty array of the correct shape
    for category in selected_categories:
        input_data[0, target_cols.index(category)] = previous_cpi_value
    input_data[0, target_cols.index('Total Local Sales')] = total_local_sales
    input_data[0, target_cols.index('Total Export Sales')] = total_export_sales
    input_data[0, target_cols.index('USD_ZAR')] = usd_zar
    input_data[0, target_cols.index('GBP_ZAR')] = gbp_zar
    input_data[0, target_cols.index('EUR_ZAR')] = eur_zar
    
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
    total_local_sales = st.number_input("Total_Local_Sales", value=0.0)
    total_export_sales = st.number_input("Total_Export_Sales", value=0.0)
    usd_zar = st.number_input("USD_ZAR", value=0.0)
    gbp_zar = st.number_input("GBP_ZAR", value=0.0)
    eur_zar = st.number_input("EUR_ZAR", value=0.0)

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
        make_prediction(selected_categories, create_input_data(selected_categories, previous_cpi_value, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar), loaded_models, "_".join(selected_categories), predictions, reference_date, selected_month)

        # Display predictions
        st.write(f"Predicted CPI values for {selected_month} for the selected categories:")
        for category in selected_categories:
            category_formatted = category.replace(' ', '_')  # Replace spaces with underscores
            st.write(f"{category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[category_formatted + '_CPI_for_' + reference_date.strftime('%B_%Y') + '_' + selected_month]:.2f}")

if __name__ == "__main__":
    main()
