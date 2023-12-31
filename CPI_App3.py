import streamlit as st
import numpy as np
import pandas as pd
import os
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Define the target columns
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
               'Communication', 'Education', 'Food and non-alcoholic beverages',
               'Headline CPI', 'Health', 'Household contents and services',
               'Housing and utilities', 'Miscellaneous goods and services',
               'Recreation and culture', 'Restaurants and hotels ', 'Transport']

def load_models():
    loaded_models = {}
    for column in target_cols:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model
                print(model_path)
            else:
                print(model_path)
    return loaded_models

def create_input_data(selected_category, previous_cpi_value, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar):
    input_data = np.zeros((1, len(target_cols) + 6))  # Create an empty array with additional columns
    input_data[0, target_cols.index(selected_category)] = previous_cpi_value
    
    # Set the values for the non-category columns
    input_data[0, -6] = total_local_sales
    input_data[0, -5] = total_export_sales
    input_data[0, -4] = usd_zar
    input_data[0, -3] = gbp_zar
    input_data[0, -2] = eur_zar
    
    # Apply StandardScaler to scale the input data
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    return input_data_scaled

# Function to make predictions for a category
def make_prediction(selected_category, input_data, loaded_models, category_formatted, predictions, reference_date, selected_month):
    for i in range(1, 4):
        model_key = f"{selected_category}_month_{i}"
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

    # Display input fields for previous CPI values for each selected category
    previous_cpi_values = {}
    for selected_category in selected_categories:
        previous_cpi_values[selected_category] = st.number_input(f"Enter previous CPI value for {selected_category}:", value=0.0)

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
        for selected_category in selected_categories:
            # Create input data excluding the 19th feature (month selection)
            input_data = create_input_data(selected_category, previous_cpi_values[selected_category], total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar)[:, :-1]
    
            make_prediction(selected_category, input_data, loaded_models, selected_category.replace(' ', '_'), predictions, reference_date, selected_month)

        # Display predictions
        st.write(f"Predicted CPI values for {selected_month} for the selected categories:")
        for selected_category in selected_categories:
            category_formatted = selected_category.replace(' ', '_')  # Replace spaces with underscores
            key = f"{category_formatted}_CPI_for_{reference_date.strftime('%B_%Y')}_{selected_month}"
    
            if key in predictions:
                st.write(f"{selected_category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[key]:.2f}")
            else:
                st.write(f"No prediction found for {selected_category} in {selected_month}")

if __name__ == "__main__":
    main()
