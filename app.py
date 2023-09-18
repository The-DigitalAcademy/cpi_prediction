import streamlit as st
import pandas as pd
import numpy as np
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler


# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}

# Streamlit UI
st.title("Manoko's CPI Prediction App")

# User input
category = st.selectbox("Select Category", target_cols)
selected_date = st.date_input("Select Date")

# Preprocess the input data (similar to what you did during training)
# Extract month and year from selected_date
selected_month = selected_date.month
selected_year = selected_date.year

input_data = pd.read_csv('train.csv')
#st.write(input_data)


# # Add the code to calculate lagged features
feats_to_lag = input_data.columns[1:].to_list()
for col in feats_to_lag:
    if col != 'year_month':
         for i in range(1, 3):
             
             input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)

input_data.drop(0)
input_data.bfill()
st.write(input_data)

# input_data = input_data({'Month': [selected_date.strftime("%Y-%m")],
#                          'year_month':[selected_date].strftime("%Y-%m")})
input_data = input_data.drop(columns= target_cols + ['Total_Local Sales', 'Total_Export_Sales'])


# Add a button to trigger predictions
# Add a button to trigger predictions
if st.button("Predict CPI"):
    
    last_row = input_data[-1:]
    #st.write(last_row)
   #st.write(last_row)
    input_df = last_row
     #last_row = input_data[-1:]
    
    #  # Create a DataFrame from the input features
     #input_df = pd.DataFrame([last_row])
    #st.write(input_df)

    # # Transform the input data using the scaler
    input_scaled = scaler.transform(input_df.drop(columns= ['Month', 'year_month']))
    #st.write(input_scaled)
    y_pred_test = []
    
    #making predictions without the for loop
    lr_model = model_dict[category]
    predicted_cpi = lr_model.predict(input_scaled)
    st.write(f"Predicted CPI for {category} in {selected_month}/{selected_year}: {predicted_cpi[0]:.2f}")
    
    

    # # Use your trained model to make predictions
    # for target_col in target_cols:
    #     lr_model = model_dict[target_col]
    #     #last_row = scaler.transform(last_row.drop(columns= ['Month', 'year_month']))
    #     predicted_cpi = lr_model.predict(input_scaled)
    #     predicted_cpi = np.array(predicted_cpi).T
    #     #st.write(predicted_cpi)
    #     y_pred_test.append(predicted_cpi)
    #     st.write(f"Predicted CPI for {category} in {selected_month}/{selected_year}: {predicted_cpi[0]:.2f}")
    
       
    #    #st.write(predicted_cpi)
       #st.write(target_colomns)
       #Combine predictions into a DataFrame
    #df_pred = pd.DataFrame({col: y_pred_test[i] for i, col in enumerate(target_cols)})
    #st.write(df_pred)
    # input_data = input_data.drop(columns= ['Month', 'year_month'])
    # st.write(input_data)
    # #last_row = input_data[-1:].copy()
    # #st.write(last_row)
    # #scaler = MinMaxScaler()
    # #last_row = scaler.fit(last_row.drop(['Month', 'year_month'], axis=1))
    # #last_row = input_data[-1:].copy()
    
    # #st.write(f"Predicted CPI for {category} in {selected_month}/{selected_year}: {predicted_cpi[0]:.2f}")

    # # # Transform the input data using the scaler
    # #last_row = scaler.transform(last_row)
    # y_pred_test = []
    
    # y_pred_test = []

    # for target_col in target_cols:
        
    #    lr_model = model_dict[target_col]
    #    X_test_scaled = scaler.transform(X)  # Scale the test data
    #    y_pred_col = lr_model.predict(X_test_scaled)  # Make predictions
    #    y_pred_test.append(y_pred_col)

# Combine predictions into a DataFrame
#df_pred = pd.DataFrame({col: y_pred_test[i] for i, col in enumerate(target_cols)})
       #predicted_cpi = model.predict(last_row)
       #predicted_cpi = np.array(predicted_cpi).T
       #y_pred_test.append(predicted_cpi)
       
    
# # # Display the predicted CPI value to the user
#     st.write(f"Predicted CPI for {category} in {selected_month}/{selected_year}: {predicted_cpi[0]:.2f}")
    
#     #st.write(last_row)
# # # Create a DataFrame with the selected month, year, and lagged features
# # # Replace this with your own data preprocessing logic
# # # Here, we create a sample input DataFrame for demonstration purposes
# # input_data = pd.DataFrame({
# #     'Month': [selected_date.strftime("%Y-%m")],
# #     'year_month': [selected_date.strftime("%Y-%m")],  # Assuming 'year_month' follows the same format as 'Month'
# #     # Add lagged features here (replace with your actual feature values)
# #     'prev_1_month_Clothing and footwear': [0.0],  # Replace with actual lagged values
# #     # Add more lagged features...
# # })

# # # Make predictions using the selected model
# # model = model_dict[category]
# # # Pass the preprocessed input data to the model for prediction
# # predicted_cpi = model.predict(input_data.drop(['Month', 'year_month'], axis=1))

# # # Display the predicted CPI value to the user
# # st.write(f"Predicted CPI for {category} in {selected_month}/{selected_year}: {predicted_cpi[0]:.2f}")
