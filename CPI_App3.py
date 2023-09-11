import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import datetime
from collections import defaultdict

target_cols = ['Headline_CPI', 'Alcoholic beverages and tobacco', 'Clothing and footwear',
              'Communication', 'Education', 'Food and non-alcoholic beverages',
              'Health', 'Household contents and services',
              'Housing and utilities', 'Miscellaneous goods and services',
              'Recreation and culture', 'Restaurants and hotels ', 'Transport']

# Streamlit app
def main():
    # Set the title
    st.title("CPI Prediction Dashboard")

selected_categories = st.multiselect("Select categories to predict:", target_cols)

# Iterate over selected categories
for selected_category in selected_categories:
    st.write(f"Predictions for {selected_category}:")
    
    # Display input fields for previous CPI values for the selected category
    st.write(f"Enter previous CPI value for {selected_category}:")
    previous_cpi_value = st.number_input(f"Previous CPI for {selected_category}", value=0.0)
    
    # Display input fields for vehicle sales and currency for the selected category
    vehicle_sales = st.number_input(f"{selected_category} Vehicle Sales", value=0.0)
    currency_input = st.number_input(f"{selected_category} Currency Input", value=0.0)

    # Dictionary to store loaded models
    loaded_models = {}

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

    # Display input fields for other X_train columns
    st.write("Enter previous values for other features:")
    for col in target_cols:
        if col != selected_category and col not in ['Vehicle Sales', 'Currency Input']:
            input_value = st.number_input(f"Previous {col}", value=0.0)
            input_data.at[0, col] = input_value

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

    # Dictionary to store predictions
    predictions = {}

    # Iterate over target columns and months
    for category in target_cols:
        for i in range(1, 4):
            model_key = f"{category}_month_{i}"
            if model_key in loaded_models:
                loaded_model = loaded_models[model_key]
                y_pred = loaded_model.predict(input_data)
                category_formatted = category.replace(' ', '_')  # Replace spaces with underscores
                predictions[f'{category_formatted}_CPI_for_{reference_date.strftime("%B_%Y")}'] = round(y_pred[0][0], 2)

    # Display predictions
    st.write(f"Predicted CPI values for {selected_month}:")
    for category in target_cols:
        category_formatted = category.replace(' ', '_')  # Replace spaces with underscores
        st.write(f"{category} CPI for {reference_date.strftime('%B_%Y')}: {predictions[category_formatted + '_CPI_for_' + reference_date.strftime('%B_%Y')]:.2f}")

if __name__ == "__main__":
    main()
