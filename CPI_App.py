import streamlit as st
import pdfplumber
import re
import numpy as np
import os
import datetime
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Define the target columns with their corresponding prefixes
target_cols_with_prefixes = {
    'Headline CPI': 'Headline',
    'Food and non-alcoholic beverages': 'Food and non-',
    'Alcoholic beverages and tobacco': 'Alcoholic beverages',
    'Clothing and footwear': 'Clothing and footwear',
    'Housing and utilities': 'Housing and utilities',
    'Household contents and services': 'Household contents',
    'Health': 'Health',
    'Transport': 'Transport',
    'Communication': 'Communication',
    'Recreation and culture': 'Recreation and culture',
    'Education': 'Education',
    'Restaurants and hotels ': 'Restaurants and hotels',
    'Miscellaneous goods and services': 'Miscellaneous goods',
}

def load_models():
    loaded_models = {}
    for column in target_cols_with_prefixes:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model
                print(model_path)
            else:
                print(model_path)
    return loaded_models

# Function to extract text from PDF and process it to get CPI values
def process_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        page7 = pdf.pages[7]  
        page8 = pdf.pages[8]
        text1 = page7.extract_text()
        text2 = page8.extract_text()
    
    # Combine the extracted text from both pages
    text_to_extract = text1 + text2

    # Split the text into lines and initialize a dictionary to store category values
    lines = text_to_extract.split('\n')
    category_values = {}

    # Iterate through the lines starting from the 4th line (skipping headers)
    for line in lines[3:]:
        # Split the line using whitespace
        columns = line.split()
        if len(columns) >= 4:
            # Extract the category and value
            category = ' '.join(columns[:-3])  # Combine columns as the category name
            value = columns[-3]  # Get the value from the 4th column

            # Add the category and its value to the dictionary
            category_values[category] = value

    # Iterate through the category prefixes
    for column, prefix in target_cols_with_prefixes.items():
        category_value = None

        # Iterate through the dictionary items
        for category, value in category_values.items():
            if category.startswith(prefix):
                # Split the value by ":" and get the last part
                category_value = value.split(':')[-1].strip()
                break  # Exit the loop once the category value is found

    return category_values
    
def create_input_data(selected_category, category_values, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar):
    input_data = np.zeros((1, len(target_cols_with_prefixes) + 5))  # Create an empty array with additional columns
    selected_category_adjusted = selected_category.replace(' ', '_')
    
    # Iterate through the target columns and find the index for the selected category
    for index, (category, prefix) in enumerate(target_cols_with_prefixes.items()):
        if category == selected_category_adjusted:
            input_data[0, index] = float(category_values[selected_category])
    
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
def make_predictions(selected_category, category_values, loaded_models, category_formatted, predictions, reference_date, selected_month):
    for i in range(1, 4):
        model_key = f"{selected_category}_month_{i}"
        if model_key in loaded_models:
            loaded_model = loaded_models[model_key]
            input_data = create_input_data(selected_category, category_values, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar)
            y_pred = loaded_model.predict(input_data)
            predictions[f'{category_formatted}_CPI_for_{reference_date.strftime("%B_%Y")}_Month_{i}'] = round(y_pred[0][0], 2)

# Streamlit app
def main():
    # Set the title
    st.title("CPI Vision")

    # Create a sidebar navigation
    menu = st.sidebar.radio("Navigation", ["Home", "Model", "CPI Dashboard"])

    if menu == "Home":
        st.header("Meet the team")

    elif menu == "Model":
        # Display the Model section
        st.header("Model")
        # Allow the user to upload a PDF document
        uploaded_file = st.file_uploader("Upload Current CPI PDF document", type=["pdf"])

        if uploaded_file is not None:
            # Process the uploaded PDF file
            st.text("Processing the uploaded PDF...")
            category_values = process_pdf(uploaded_file)

            # Allow the user to select categories for prediction
            selected_categories = st.multiselect(
                "Select categories to predict:", list(target_cols_with_prefixes.keys()), default=[list(target_cols_with_prefixes.keys())[0]]
            )

            if selected_categories:
                st.text("Extracted CPI values from the PDF:")
                for selected_category in selected_categories:
                    # Initialize the extracted_cpi_value
                    extracted_cpi_value = None

                    # Check if the selected_category is in the category_values dictionary
                    if selected_category in category_values:
                        extracted_cpi_value = category_values[selected_category]

                    if extracted_cpi_value is not None:
                        st.text(f"{selected_category}: {extracted_cpi_value}")
                    else:
                        st.text(f"{selected_category}: Category not found in the extracted data.")
        
            # Display input fields for vehicle sales and currency
            st.write("Enter Vehicle Sales and Currency Input:")
            total_local_sales = st.number_input("Total_Local_Sales", value=0.0)
            total_export_sales = st.number_input("Total_Export_Sales", value=0.0)
            usd_zar = st.number_input("USD_ZAR", value=0.0)
            gbp_zar = st.number_input("GBP_ZAR", value=0.0)
            eur_zar = st.number_input("EUR_ZAR", value=0.0)

            # Initialize an empty predictions dictionary
            predictions = {}

            # Load saved models
            loaded_models = load_models()

            # Create a "Predict CPI" button
            if st.button("Predict CPI"):
                # Create a table to display the predicted CPI values for all three months
                table_data = []

                # Calculate the reference date based on the current date
                current_date = datetime.date.today()

                # Create headers for the table
                headers = ["Category"]
                for i in range(1, 4):
                    reference_date = current_date.replace(month=current_date.month + i)
                    headers.append(f"{reference_date.strftime('%B %Y')}")

                table_data.append(headers)

                # Make predictions for the selected categories
                for selected_category in selected_categories:
                    # Create a row for each category
                    row = [selected_category]

                    # Make predictions for all three months
                    for i in range(1, 4):
                        reference_date = current_date.replace(month=current_date.month + i)
                        make_predictions(selected_category, category_values, loaded_models, selected_category.replace(' ', '_'), predictions, reference_date, f"Month {i}")
                        row.append(predictions[f'{selected_category.replace(" ", "_")}_CPI_for_{reference_date.strftime("%B_%Y")}_Month_{i}'])

                    table_data.append(row)

                # Display the predicted CPI values in a table
                st.text("Predicted CPI values for the next three months for the selected categories:")
                st.table(table_data)

    elif menu == "CPI Dashboard":
        # Display the Dashboard section
        st.header("Dashboard")

if __name__ == "__main__":
    main()
