import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

# Load the linear regression models
model_filenames = {
    'Alcoholic beverages and tobacco': 'model_alcoholic_tobacco.pkl',
    'Clothing and footwear': 'model_clothing_footwear.pkl',
    # Add model filenames for other columns...
}

lr_models = {}
for col, model_filename in model_filenames.items():
    with open(model_filename, 'rb') as model_file:
        lr_models[col] = pickle.load(model_file)

st.title('CPI Prediction for April')

# Input features
st.header('Input Features')

# Select category
selected_category = st.selectbox('Select Category', list(model_filenames.keys()))

# Select month
selected_month = st.text_input('Enter Month (YYYY-MM)', '2023-03')  # Default value for example

# Make predictions
if st.button('Predict'):
    # Load the dataset for the selected month
    dataset_filename = 'UI2.csv'  # Replace with your dataset filename
    dataset = pd.read_csv(dataset_filename)

    # Filter the dataset for the selected month
    selected_row = dataset[dataset['Month'] == selected_month]

    if not selected_row.empty:
        # Extract the lagged feature value
        lagged_feature_value = selected_row[selected_category].values[0]

        # Construct the user_inputs dictionary for making predictions
        user_inputs = {
            'prev_1_month_' + selected_category: lagged_feature_value
        }

        # Scale user inputs
        user_inputs_scaled = scaler.transform([list(user_inputs.values())])

        # Make predictions
        lr_model = lr_models[selected_category]
        predicted_cpi = lr_model.predict(user_inputs_scaled)[0]

        st.success(f'Predicted CPI for {selected_category} in {selected_month}: {predicted_cpi:.2f}')
    else:
        st.error(f'Data not available for {selected_month}. Please select a different month.')
