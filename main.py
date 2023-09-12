import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime

# Load your data
# Replace 'your_data.csv' with the actual path to your dataset
cpi_pivot = pd.read_csv('UI2.csv')

# Define functions for data preprocessing and model prediction
def preprocess_data(data):
    # Add your data preprocessing steps here
    return data

def train_model(data):
    # Split the data into training and validation sets
    train = data[data['Month'] != '2023-04-30']
    validation = data[data['Month'] == '2023-04-30']
    
    # Define target and feature columns
    target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
                   'Communication', 'Education', 'Food and non-alcoholic beverages',
                   'Headline_CPI', 'Health', 'Household contents and services',
                   'Housing and utilities', 'Miscellaneous goods and services',
                   'Recreation and culture', 'Restaurants and hotels ', 'Transport']
    
    features = [col for col in data.columns if col not in target_cols + ['Month', 'year_month', 'Total_Local Sales', 'Total_Export_Sales']]
    
    # Initialize models and scalers
    lr_models = {}
    y_pred = []
    scaler = MinMaxScaler()
    
    # Training
    for target_col in target_cols:
        lr_model = LinearRegression()
        X_train = train[features]
        y_train = train[target_col]
        X_train_scaled = scaler.fit_transform(X_train)
        lr_model.fit(X_train_scaled, y_train)
        lr_models[target_col] = lr_model
    
    # Validation
    for target_col in target_cols:
        lr_model = lr_models[target_col]
        X_val = validation[features]
        y_val = validation[target_col]
        X_val_scaled = scaler.transform(X_val)
        y_pred_col = lr_model.predict(X_val_scaled)
        y_pred.append(y_pred_col)
    
    # Calculate RMSE
    y_pred = np.array(y_pred).T
    df = pd.DataFrame({'y_pred': y_pred.flatten(), 'y_val': y_val.values.flatten()})
    rmse = np.sqrt(mean_squared_error(df['y_pred'], df['y_val']))
    
    return lr_models, scaler, rmse

def predict_cpi(user_input, models, scaler):
    # Preprocess user input
    user_input = preprocess_data(user_input)
    
    # Make CPI predictions
    y_pred_test = []
    for target_col in target_cols:
        lr_model = models[target_col]
        X_test = user_input[features]
        X_test_scaled = scaler.transform(X_test)
        y_pred_col = lr_model.predict(X_test_scaled)
        y_pred_test.append(y_pred_col)
    
    # Combine predictions into a DataFrame
    df_pred = pd.DataFrame({col: y_pred_test[i] for i, col in enumerate(target_cols)})
    
    return df_pred

# Create a Streamlit app
st.title("CPI Prediction App")

# Add user input sidebar
st.sidebar.header("User Input")
# Add UI components for user input (e.g., text inputs, sliders, date pickers)

# Define target columns
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
               'Communication', 'Education', 'Food and non-alcoholic beverages',
               'Headline_CPI', 'Health', 'Household contents and services',
               'Housing and utilities', 'Miscellaneous goods and services',
               'Recreation and culture', 'Restaurants and hotels ', 'Transport']

# Initialize models and scaler
lr_models, scaler, rmse = train_model(cpi_pivot)

# Get user input (you need to implement this based on your UI design)
user_input = get_user_input()

# Make CPI predictions
cpi_predictions = predict_cpi(user_input, lr_models, scaler)

# Display the predictions
st.write("Predicted CPI for April:")
st.write(cpi_predictions)

# Display RMSE
st.write(f"Root Mean Squared Error (RMSE): {rmse}")
