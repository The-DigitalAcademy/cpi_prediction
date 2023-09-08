import streamlit as st
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import datetime
from collections import defaultdict

target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear',
       'Communication', 'Education', 'Food and non-alcoholic beverages',
       'Headline_CPI', 'Health', 'Household contents and services',
       'Housing and utilities', 'Miscellaneous goods and services',
       'Recreation and culture', 'Restaurants and hotels ', 'Transport']

# Function to preprocess the data
def preprocess_data(cpi_csv, vehicles_csv, currency_csv):
    # Load CSV data
    cpi_hist = pd.read_csv(cpi_csv)
    vehicles = pd.read_csv(vehicles_csv)
    currency_df = pd.read_csv(currency_csv)

    ######################################## HISTORY CPI ############################################

    cpi_hist['Month'] = pd.to_datetime(cpi_hist['Month'])
    cpi_pivot = cpi_hist.pivot(index='Month', columns='Category', values='Value').reset_index()
    cpi_pivot = cpi_pivot.sort_values("Month").reset_index(drop=True)
    cpi_pivot['Month'] = pd.to_datetime(cpi_pivot['Month'], format='%B %Y-%d')

    date_str = '2023-04-30'
    date_obj = pd.to_datetime(date_str)
    new_row = pd.DataFrame({'Month': [date_obj]})
    cpi_pivot = pd.concat([cpi_pivot, new_row]).reset_index(drop=True)
    cpi_pivot['Month'] = pd.to_datetime(cpi_pivot['Month'])

    cpi_pivot['year_month'] = pd.to_datetime(cpi_pivot['Month'], format='%Y-%b').dt.strftime('%Y-%m')
    feats_to_lag = [col for col in cpi_pivot.columns if col not in ['Month', 'year_month']]

    # Create a new column for the rolling average and calculate the average for all 8 features
    for col in feats_to_lag:
        for i in range(1, 8):
            cpi_pivot[f"prev_{i}_month_{col}"] = cpi_pivot[col].shift(i)

    # Calculate the rolling average for the current feature
    for col in feats_to_lag:
        cpi_pivot[f"Average_{col}"] = cpi_pivot[[f"prev_{i}_month_{col}" for i in range(1, 8)]].mean(axis=1)

    # List the columns to keep (those with "prev" in their names)
    columns_to_drop = [col for col in cpi_pivot.columns if 'prev' in col]

    cpi_pivot = cpi_pivot.drop(columns=columns_to_drop)
    # if 0 in cpi_pivot.index:
    #    cpi_pivot = cpi_pivot.drop(index=0)


    ####################################### VEHICLES ##############################################################

    start_date = datetime.datetime.strptime("2020-12-31", "%Y-%m-%d")
    end_date = datetime.datetime.strptime("2023-03-31", "%Y-%m-%d")

        # Concatenate data for the next two months
    for i in range(1, 3):
        date_obj = end_date + datetime.timedelta(days=i)
        new_row = pd.DataFrame({'Month': [date_obj]})
        cpi_pivot = pd.concat([cpi_pivot, new_row]).reset_index(drop=True)

    # difference between each date. M means one month end
    D = 'M'

    date_list = pd.date_range(start_date, end_date, freq=D)[::-1]
    vehicles['Month'] = date_list
    vehicles['Month'] = pd.to_datetime(vehicles['Month'], format='%Y-%b-%d')
    vehicles['year_month'] = pd.to_datetime(vehicles['Month'], format='%Y-%b').dt.strftime('%Y-%m')

    columns_to_drop = [
        'Local_Passenger_Vehicles', 'Export_Sales_Passenger_Vehicles',
        'Local_Light_Commercial_Vehicles', 'Export_Sales_Light_Commercial_Vehicles',
        'Local_Medium_Commercial_Vehicles', 'Export_Sales_Medium_Commercial_Vehicles',
        'Local_Heavy_Commercial_Vehicles', 'Export_Sales_Heavy_Commercial_Vehicles',
        'Local_Extra_Heavy_Commercial_Vehicles', 'Export_Sales_Extra_Heavy_Commercial_Vehicles',
        'Local_Bus_Sales', 'Export_Sales_Buses'
    ]

    vehicles = vehicles.drop(columns=columns_to_drop)

    cpi_vehicles = cpi_pivot.merge(vehicles[['year_month', 'Total_Local Sales', 'Total_Export_Sales']],
                                    on='year_month', how='left')

    #################################################### CURRENCY DATA ###############################################

    currency_df['Date'] = pd.to_datetime(currency_df['Date'])
    currency_df['year_month'] = pd.to_datetime(currency_df['Date'], format='%Y-%b').dt.strftime('%Y-%m')

    df_merged = cpi_vehicles.merge(currency_df, on='year_month', how='left')

    df_merged = df_merged.drop(0)
    df_merged = df_merged.bfill()
    # df_merged = df_merged.drop(['Month'], axis=1)

    df_merged = df_merged.drop(['Date'], axis=1)
    df_merged = df_merged.dropna()


    # Create features for the next 3 months
    for i in range(1, 4):
        df_merged[f"next_{i}_month"] = df_merged['Month'].shift(-i)

    # Adjust target variable to predict next 3 months for float columns
    for i in range(1, 4):
        for col in target_cols:
            df_merged[f"next_{i}_month_{col}"] = df_merged[col].shift(-i)

    # Separate datetime values
    date_cols = ['Month'] + [f"next_{i}_month" for i in range(1, 4)]
    datetime_data = df_merged[date_cols]
    df_merged = df_merged.drop(columns=date_cols)

    if 0 in df_merged.index:
       df_merged = df_merged.drop(index=0)
    
       df_merged = df_merged.bfill()

    return df_merged, datetime_data


# Function to train and save models
def train_and_save_models(df_merged):
    X = df_merged.drop(columns=['year_month'] + target_cols)
    y = df_merged[[f"next_{i}_month_{col}" for i in range(1, 4) for col in target_cols]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Specify which columns to standardize (excluding 'Month')
    columns_to_standardize = [col for col in X_train.columns if col != 'Month']

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Fit and transform the scaler on the selected columns in both training and test sets
    X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
    X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])

    ####### MODEL TRAINING ##########

    # Directory to save the models
    save_directory = "saved_models/"

    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    models = defaultdict(dict)

    model_types = [
        ('Linear Regressor', Sequential([
            Dense(1, input_dim=X_train.shape[1], activation='linear')
        ])),
        ('Deep Neural Network', Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dense(64, activation='relu'),
            Dense(len(target_cols) * 3, activation='linear')  # Adjust output size for 3 months
        ]))
    ]

    for col in target_cols:
        mse_values = []

        for model_name, model_architecture in model_types:
            # Create the model instance
            model = model_architecture

            if isinstance(model, Sequential):
                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Fit the model
                model.fit(X_train, y_train[[f"next_{i}_month_{col}" for i in range(1, 4)]], epochs=100, batch_size=32,
                          verbose=0)

                # Save the model
                model.save(os.path.join(save_directory, f"{col}_{model_name}.h5"))

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate mean squared error for evaluation
                mse = mean_squared_error(y_test[[f"next_{i}_month_{col}" for i in range(1, 4)]], y_pred)
                mse_values.append((model_name, mse))

        if mse_values:
            # Choose the best model based on the lowest MSE
            best_model_name, best_mse = min(mse_values, key=lambda x: x[1])
            print(f"Best Model for {col}: {best_model_name}, Mean Squared Error: {best_mse}")

            # Store the trained best model
            best_model = [model_architecture for model_name, model_architecture in model_types if
                          model_name == best_model_name][0]
            models[col]['model'] = best_model
            models[col]['mse'] = best_mse
        else:
            print(f"No models found for {col}.")

    ############################# END OF TRAINING ###################

    return save_directory, X_test

# Function to make predictions using trained models
def make_predictions(data, models, num_months=1):
    # Directory where models are saved
    save_directory = "saved_models/"

    # Define a dictionary to store the loaded models
    loaded_models = {}

    # Load the models for each target column
    for col in target_cols:
        # Check if a model exists for the column
        model_path = os.path.join(save_directory, f"{col}_Deep Neural Network.h5")
        if os.path.exists(model_path):
            loaded_model = load_model(model_path)
            loaded_models[col] = loaded_model

    # Make predictions for the next 3 months for each category (float columns)
    predictions = {}
    for column, model in loaded_models.items():
        # Predict for the next num_months
        future_X_test = X_test.copy()
        for _ in range(num_months):
            future_X_test = future_X_test.append(future_X_test.tail(1), ignore_index=True)
        y_pred = model.predict(future_X_test)
        predictions[column] = [round(value[0], 2) for value in y_pred]

    return predictions

# Streamlit app
def main():
    # Set the title
    st.title("CPI Dashboard")
    st.sidebar.title("Model")

    # Upload CSV files
    uploaded_cpi = st.sidebar.file_uploader("Upload CPI History CSV", type=["csv"])
    uploaded_vehicles = st.sidebar.file_uploader("Upload Vehicles CSV", type=["csv"])
    uploaded_currency = st.sidebar.file_uploader("Upload Currency CSV", type=["csv"])

    if st.sidebar.button("Submit"):
        # Check if files are uploaded
        if uploaded_cpi is None or uploaded_vehicles is None or uploaded_currency is None:
            st.sidebar.error("Please upload all three CSV files.")
        else:
            # Preprocess data
            df_merged, datetime_data = preprocess_data(uploaded_cpi, uploaded_vehicles, uploaded_currency)

            # Train models
            trained_models = train_and_save_models(df_merged)
            input_for_predictions = df_merged.tail(1)  # Input for making predictions

        # Display predictions for the next 3 months
        st.write("Predicted CPI values for the next 3 months:")
        predictions = make_predictions(trained_models, df_merged, num_months=3)
        st.write(predictions)

if __name__ == "__main__":
    main()
