import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime

# Load your data
# Replace 'your_data.csv' with the actual path to your dataset
cpi_pivot = pd.read_csv('UI2.csv')

# Define functions for data preprocessing and model prediction
def preprocess_data(data):
    # Convert 'Month' column to datetime format
    data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')

    # Add your data preprocessing steps here
    # For example, fill missing values, handle categorical variables, etc.
    # You can also include the lagging method you mentioned in your original code here.

    # Assuming you have already created lag features for each category, e.g., 'prev_1_month_...'
    # Drop unnecessary columns
    cols_to_drop = ['year_month', 'Total_Local Sales', 'Total_Export_Sales']
    data = data.drop(columns=cols_to_drop, axis=1)

    # Fill any remaining missing values
    data = data.fillna(method='bfill')  # Backfill missing values

    return data

# Define the target columns and features as you did before
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
               'Communication', 'Education', 'Food and non-alcoholic beverages',
               'Headline_CPI', 'Health', 'Household contents and services',
               'Housing and utilities', 'Miscellaneous goods and services',
               'Recreation and culture', 'Restaurants and hotels ', 'Transport']

features = [col for col in cpi_pivot.columns if col not in target_cols + ['Month', 'year_month']]

# Initialize models and scaler
lr_models = {col: LinearRegression() for col in target_cols}
#scaler = MinMaxScaler()

# Training
for target_col in target_cols:
    train = cpi_pivot[cpi_pivot['Month'] != '2023-04-30']
    X_train = train[features]
    y_train = train[target_col]
    X_train_scaled = scaler.fit_transform(X_train)
    lr_models[target_col].fit(X_train_scaled, y_train)

# Create a Streamlit app
st.title("CPI Prediction App")

# Add user input sidebar
st.sidebar.header("User Input")

# Input for selecting the category
category = st.sidebar.selectbox("Select Category", target_cols)

# Input for specifying the month and year
month_year_input = st.sidebar.text_input("Enter Month and Year (e.g., '2023-04'):")

# Parse the user input for month and year
try:
    user_month_year = pd.to_datetime(month_year_input, format='%Y-%m')
except ValueError:
    st.sidebar.warning("Please enter a valid Month and Year (YYYY-MM).")
    st.stop()

# Check if the user-selected category is valid
if category not in target_cols:
    st.sidebar.warning("Please select a valid category from the dropdown.")
    st.stop()

# Get user input (you need to implement this based on your UI design)
user_input = cpi_pivot[cpi_pivot['Month'] == user_month_year]

# Preprocess user input
user_input = preprocess_data(user_input)

# Make CPI predictions
X_test = user_input[features]
X_test_scaled = scaler.transform(X_test)
cpi_prediction = lr_models[category].predict(X_test_scaled)

# Display the predictions
st.write(f"Predicted CPI for {category} in {user_month_year}:")
st.write(cpi_prediction[0])  # Display the first prediction

# Note: This code displays only the first prediction for the selected category.
