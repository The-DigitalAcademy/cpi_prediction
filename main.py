import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib

# Load your trained models here
ccc = pd.read_csv('ccc.csv')
lr_models = {}  # This should be populated with your models
scaler = MinMaxScaler()  # Use the same scaler used during training
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
       'Communication', 'Education', 'Food and non-alcoholic beverages',
       'Headline_CPI', 'Health', 'Household contents and services',
       'Housing and utilities', 'Miscellaneous goods and services',
       'Recreation and culture', 'Restaurants and hotels ', 'Transport']

dataw= [{"Alcoholic beverages and tobacco":100.0,	"Clothing and footwear":100.2	,"Communication":99.8,	"Education":100.0,	
         "Food and non-alcoholic beverages":100.9	,"Headline_CPI":100.2,	"Health":100.1,	"Household contents and services":100.4,
         "Housing and utilities":100.0, "Miscellaneous goods and services":100.6,	"Recreation and culture":100.2	,
         "Restaurants and hotels ":101.2,	"Transport":98.9,"Total_Local Sales":41382.0,	"Total_Export_Sales":19089.0}]
ccc = ccc.drop(['Month', 'year_month'], axis=1)
cpi_pivot1 = pd.DataFrame(dataw)
cpi_pivot1 = pd.concat([cpi_pivot1,ccc ])

for target_col in target_cols:
    # Load the serialized models, replace 'model_filename.pkl' with your actual filenames
    model = joblib.load(f'models/{target_col}_model.pkl')
    lr_models[target_col] = model

# Function to make predictions
# Function to make predictions
def predict_cpi(data):
    # Create a DataFrame from user input data
    input_df = pd.DataFrame([data])

    # Apply the same data preprocessing as during training
    feats_to_lag = [col for col in input_df.columns if col not in ['Total_Local Sales', 'Total_Export_Sales']]
    for col in feats_to_lag:
        for i in range(1, 8):
            input_df[f"prev_{i}_month_{col}"] = input_df[col].shift(i)
    input_df = input_df.bfill()

    # Make predictions for each category
    y_pred_test = []

    for target_col in target_cols:
        lr_model = lr_models[target_col]
        X_test_scaled = scaler.transform(input_df)  # Scale the test data
        y_pred_col = lr_model.predict(X_test_scaled)  # Make predictions
        y_pred_test.append(y_pred_col)

    # Combine predictions into a DataFrame
    df_pred = pd.DataFrame({col: y_pred_test[i] for i, col in enumerate(target_cols)})
    
    return df_pred

# Streamlit app
def main():
    st.title("CPI Prediction App")

    # User input for April data
    st.write("Enter data for April:")
    data_input = st.text_input("Data (comma-separated values):")
    
    if data_input:
        # Parse user input as a dictionary
        try:
            data = {col: float(value) for col, value in zip(target_cols, data_input.split(","))}
        except ValueError:
            st.error("Invalid input format. Please use comma-separated values.")
            return

        # Make predictions
        predictions = predict_cpi(data)

        # Display predictions
        st.subheader("Predicted CPI Values for April:")
        st.dataframe(predictions)

if __name__ == "__main__":
    main()
