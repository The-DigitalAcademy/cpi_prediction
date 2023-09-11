import streamlit as st
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

    # Display a dropdown to select the category for prediction
    selected_category = st.selectbox("Select a category to predict:", target_cols)

    # Display input fields for previous CPI values
    st.write(f"Enter previous CPI value for {selected_category}:")
    previous_cpi_value = st.number_input(f"Previous CPI for {selected_category}", value=0.0)

    # Dictionary to store loaded models
    loaded_models = {}

    # Iterate over target columns and months
    for column in target_cols:
        for i in range(1, 4):
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

    # Dictionary to store predictions
    predictions = {}

    # Iterate over target columns and months
    for column in target_cols:
        for i in range(1, 4):
            model_key = f"{column}_month_{i}"
            if model_key in loaded_models:
                loaded_model = loaded_models[model_key]
                y_pred = loaded_model.predict(input_data)
                predictions[f'next_{i}_month_{column}'] = round(y_pred[0][0], 2)

    # Display predictions
    st.write("Predicted CPI values for the next 3 months:")
    for i in range(1, 4):
        st.write(f"Month {i}: {selected_category} CPI: {predictions[f'next_{i}_month_{selected_category}']:.2f}")

if __name__ == "__main__":
    main()
