import streamlit as st
import pdfplumber
import os
import datetime
import numpy as np
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


# Function to load models
def load_models():
    loaded_models = {}
    for column in target_cols_with_prefixes:
        for i in range(1, 4):
            model_path = os.path.join(f"{column}_Deep Neural Network_month_{i}.h5")
            if os.path.exists(model_path):
                loaded_model = load_model(model_path)
                loaded_models[f"{column}_month_{i}"] = loaded_model
    return loaded_models

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

            # Replace '.' with ',' and attempt to convert the value to float
            try:
                value = float(value.replace(',', '.'))
            except ValueError:
                # Handle cases where the value cannot be converted to float
                value = None

            # Add the category and its value to the dictionary
            category_values[category] = value

    return category_values

# Function to create input data
def create_input_data(selected_category, category_value, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar):
    input_data = np.zeros((1, len(target_cols_with_prefixes) + 5))  # Create an empty array with additional columns
    selected_category_adjusted = selected_category.replace(' ', '_')
    
    # Iterate through the target columns and find the index for the selected category
    for index, (category, prefix) in enumerate(target_cols_with_prefixes.items()):
        if category == selected_category_adjusted:
            input_data[0, index] = float(category_value)

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
def make_predictions(selected_category, input_data, loaded_models, category_formatted, category_values, reference_date, selected_month):
    predictions = {}
    for i in range(1, 4):
        model_key = f"{selected_category}_month_{i}"
        if model_key in loaded_models:
            loaded_model = loaded_models[model_key]
            y_pred = loaded_model.predict(input_data)
            predicted_cpi_key = f'{category_formatted}_CPI_for_{reference_date.strftime("%B_%Y")}_Month_{i}'
            percentage_change_key = f'{category_formatted}_Percentage_Change_for_{reference_date.strftime("%B_%Y")}_Month_{i}'

            if predicted_cpi_key in category_values:
                previous_cpi_value = float(category_values[predicted_cpi_key])
                percentage_change = ((y_pred[0][0] - previous_cpi_value) / previous_cpi_value) * 100
                predictions[predicted_cpi_key] = round(y_pred[0][0], 2)
                predictions[percentage_change_key] = round(percentage_change, 2)
            else:
                predictions[predicted_cpi_key] = round(y_pred[0][0], 2)
                predictions[percentage_change_key] = None  # No previous value available
    return predictions

# ...

# ...

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
            selected_category = st.selectbox("Select a category to view extracted CPI value:", list(target_cols_with_prefixes.keys()))

            if selected_category:
                extracted_cpi_value = category_values[selected_category]#target_cols_with_prefixes.items()
                if extracted_cpi_value is not None:
                    st.write(f"Extracted CPI value for {selected_category}: {extracted_cpi_value}")
                else:
                    st.write(f"No CPI value found for {selected_category}")

            # Display input fields for vehicle sales and currency
            st.write("Enter Vehicle Sales and Currency Input:")
            total_local_sales = st.number_input("Total_Local_Sales", value=0.0)
            total_export_sales = st.number_input("Total_Export_Sales", value=0.0)
            usd_zar = st.number_input("USD_ZAR", value=0.0)
            gbp_zar = st.number_input("GBP_ZAR", value=0.0)
            eur_zar = st.number_input("EUR_ZAR", value=0.0)

            # Load saved models
            loaded_models = load_models()

            if st.button("Predict CPI"):
                # Create a table to display the predicted CPI values and percentage changes for all three months
                table_data = []

                # Calculate the reference date based on the current date
                current_date = datetime.date.today()

                # Create headers for the table
                headers = ["Category"]
                for i in range(1, 4):
                    reference_date = current_date.replace(month=current_date.month + i)
                    headers.append(f"{reference_date.strftime('%B %Y')} (CPI)")
                    headers.append(f"{reference_date.strftime('%B %Y')} (Percentage Change)")

                table_data.append(headers)

                # Make predictions for the selected category
                row = [selected_category]

                # Make predictions for all three months
                input_data = create_input_data(selected_category, extracted_cpi_value, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar)
                predictions = make_predictions(selected_category, input_data, loaded_models, selected_category.replace(' ', '_'), category_values, reference_date, f"Month {i}")

                for i in range(1, 4):
                    predicted_cpi_key = f'{selected_category.replace(" ", "_")}_CPI_for_{reference_date.strftime("%B_%Y")}_Month_{i}'
                    percentage_change_key = f'{selected_category.replace(" ", "_")}_Percentage_Change_for_{reference_date.strftime("%B_%Y")}_Month_{i}'
                    row.append(predictions.get(predicted_cpi_key, ""))
                    row.append(predictions.get(percentage_change_key, ""))

                table_data.append(row)

                # Display the predicted CPI values and percentage changes in a table
                st.text("Predicted CPI values and Percentage Changes for the next three months for the selected category:")
                st.table(table_data)

    elif menu == "CPI Dashboard":
        # Display the Dashboard section
        st.header("Dashboard")

if __name__ == "__main__":
    main()
