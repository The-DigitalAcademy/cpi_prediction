import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import datetime
from collections import defaultdict


target_cols = ['Headline_CPI','Alcoholic beverages and tobacco', 'Clothing and footwear',
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

    # Load pre-trained models for the selected category
    model_path = os.path.join(save_directory, f"{column}_Deep Neural Network_month_{i}.h5")
    if os.path.exists(model_path):
       loaded_model = load_model(model_path)
       loaded_models[f"{column}_month_{i}"] = loaded_model

        # Create input data for prediction
        input_data = df_merged.tail(1).copy()  # Input for making predictions
        input_data[selected_category] = previous_cpi_value

        # Display input fields for other X_train columns
        st.write("Enter previous values for other features:")
        for col in X_train.columns:
            if col != selected_category:  # Exclude the selected category
                input_value = st.number_input(f"Previous {col}", value=0.0)
                input_data[col] = input_value

        # Make predictions for the next 3 months
        predictions = make_predictions(input_data, loaded_model, datetime_data, selected_category)

        # Display predictions
        st.write("Predicted CPI values for the next 3 months:")
        for i in range(1, 4):
            st.write(f"Month {i}: {selected_category} CPI: {predictions[f'next_{i}_month_{selected_category}']:.2f}")
    else:
        st.error(f"No pre-trained model found for {selected_category}.")

if __name__ == "__main__":
    main()
