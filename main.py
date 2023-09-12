import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

#Sample training and validation sets (replace with your data)

training_set = train[train['Month'] != "2023-03-31"]
validation_set = train[train['Month'] == "2023-03-31"] 
#training_set = pd.DataFrame()  # Your training data
#validation_set = pd.DataFrame()  # Your validation data

#Define Streamlit app
def main():
    st.title('Multi-Target Linear Regression Model')

    # User inputs for features and target columns
    st.sidebar.header('User Inputs')

    features = st.sidebar.multiselect('Select features:', training_set.columns)
    target_columns = st.sidebar.multiselect('Select target columns:', training_set.columns)

    # Training button
    if st.sidebar.button('Train Models'):
        st.subheader('Training Models')

        # Initialize dictionary to store trained models
        lr_models = {}
        scaler = MinMaxScaler()

        for target_col in target_columns:
            st.write(f"Training model for {target_col}")

#Subset the data based on user-selected features and target
            X_train = training_set[features]
            y_train = training_set[target_col]

            # Scale the features
            X_train_scaled = scaler.fit_transform(X_train)

            # Train a linear regression model
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_models[target_col] = lr_model

            st.write(f"Model for {target_col} trained successfully!")

        st.success('All models trained successfully!')

        # Provide model information (coefficients, etc.)
        st.subheader('Model Information')
        for target_col, lr_model in lr_models.items():
            st.write(f'Model for {target_col}:')
            st.write('Coefficients:', lrmodel.coef)
            # Add more model information as needed

Run Streamlit app
if name == 'main':
    main()
