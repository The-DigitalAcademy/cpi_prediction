import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained models
#loaded_models = {}
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
               'Communication', 'Education', 'Food and non-alcoholic beverages',
               'Headline_CPI', 'Health', 'Household contents and services',
               'Housing and utilities', 'Miscellaneous goods and services',
               'Recreation and culture', 'Restaurants and hotels ', 'Transport']


# Load the saved models
model_filenames = {
    'Alcoholic beverages and tobacco': 'Alcoholic beverages and tobacco_model.pkl',
    'Clothing and footwear': 'Clothing and footwear_model.pkl',
    'Communication': 'Communication_model.pkl',
    'Education': 'Education_model.pkl',
    'Food and non-alcoholic beverages': 'Food and non-alcoholic beverages_model.pkl',
    'Headline_CPI': 'Headline_CPI_model.pkl',
    'Health': 'Health_model.pkl',
    'Household contents and services': 'Household contents and services_model.pkl',
    'Housing and utilities': 'Housing and utilities_model.pkl',
    'Miscellaneous goods and services': 'Miscellaneous goods and services_model.pkl',
    'Recreation and culture': 'Recreation and culture_model.pkl',
    'Restaurants and hotels': 'Restaurants and hotels_model.pkl',
    'Transport': 'Transport_model.pkl'
    # Add model filenames for other columns...
}

lr_models = {}
for col, model_filename in model_filenames.items():
    lr_models[col] = joblib.load(model_filename)

st.title('CPI Prediction for April')

# Input features
st.header('Input Features')

# Select category
selected_category = st.selectbox('Select Category', list(model_filenames.keys()))

# Select month
selected_month = st.text_input('Enter Month (YYYY-MM)', '2023-03')  # Default value for example

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
    dataset_filename = 'your_dataset.csv'  # Replace with your dataset filename
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


# Load your dataset here
# dataset_filename = 'your_dataset.csv'
# dataset = pd.read_csv(dataset_filename)

# Assuming you have the dataset loaded and preprocessed, you can extract the lagged feature value as follows
# Replace this part with your actual data preprocessing code
# You may need to modify this code depending on your dataset structure
# if selected_month and selected_category:
#     # Extract the lagged feature value from your dataset
#     # Assuming your dataset has the necessary columns, adapt this part to your dataset structure
#     # You may need to modify this code depending on your dataset structure
#     # selected_row = dataset[(dataset['Month'] == selected_month)]
#     # lagged_feature = selected_row[f'prev_1_month_{selected_category}'].values[0]

#     # Instead of the above lines, you can manually set a default value for the lagged feature
#     # Here, we assume a default value of 0.0 for the example
#     lagged_feature = 0.0

#     # Use the selected model to make predictions
#     lr_model = lr_models[selected_category]
#     user_inputs_scaled = scaler.transform([[lagged_feature]])
#     predicted_cpi = lr_model.predict(user_inputs_scaled)[0]

#     st.success(f'Predicted CPI for {selected_category} in {selected_month}: {predicted_cpi:.2f}')


# for target_col in target_cols:
#     loaded_models[target_col] = joblib.load(f"{target_col}_model.pkl")

# # Load your trained scaler
# scaler = joblib.load("scaler.pkl")

# # Create Streamlit app
# st.title("CPI Prediction App")

# # User-friendly input fields
# selected_category = st.selectbox("Select Category", target_cols)
# selected_month = st.slider("Select Month", min_value=1, max_value=12)
# selected_year = st.slider("Select Year", min_value= 2022, max_value= 2023)  # Define min_year and max_year

# # Define a function to make predictions based on user input
# @st.cache  # Caching the function for improved performance
# def make_prediction(category, month, year):
#     # Extract the corresponding model for the selected category
#     lr_model = loaded_models[category]
    
#     # Prepare user input data
#     user_input = pd.DataFrame({'Month': [f'{year}-{month:02d}-30'], 'Category': [category]})
    
#     # Data preprocessing
#     # Date feature handling
#     user_input['Month'] = pd.to_datetime(user_input['Month'])
#     user_input['Year'] = user_input['Month'].dt.year
#     user_input['Month'] = user_input['Month'].dt.month
    
#     # Categorical encoding (assuming one-hot encoding)
#     user_input = pd.get_dummies(user_input, columns=['Category'], drop_first=True)
    
#     # Scaling (assuming Min-Max scaling with the same scaler used during training)
#     user_input_scaled = scaler.transform(user_input)
    
#     # Make predictions using the loaded model
#     prediction = lr_model.predict(user_input_scaled)
    
#     return prediction[0]  # Return the first element of the prediction (assuming it's a single value)


# # Display the prediction
# if st.button("Predict CPI"):
#     prediction = make_prediction(selected_category, selected_month, selected_year)
#     st.write(f"Predicted CPI for {selected_category} in {selected_year}-{selected_month:02d} is {prediction:.2f}")

# Optionally, display other information or visualizations
