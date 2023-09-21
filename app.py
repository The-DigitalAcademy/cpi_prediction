import streamlit as st
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # Add this import
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots  # Import make_subplots
import numpy as np

# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}

# Load the dataset
input_data = pd.read_csv('train.csv')

# Add future months and years to the dataset
current_date = datetime.date(2023, 4, 30)  # Starting from April 2023
end_date = datetime.date(2024, 12, 30)  # Extend data up to December 2024

while current_date <= end_date:
    year_month = current_date.strftime('%Y-%m')
    month = current_date.strftime('%Y-%m-%d')
    input_data = input_data.append({'year_month': year_month, 'Month': month}, ignore_index=True)
    current_date = current_date + pd.DateOffset(months=1)

# Streamlit UI
st.title("CPI Vision App")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Predict", "Dashboard"])

# Create a dictionary to store previous month's predictions for each category
previous_month_predictions = {category: None for category in target_cols}

# Page selection
if page == "Predict":
    st.header("Predict CPI")
    
    # User input - Select Category
    category = st.selectbox("Select Category", target_cols)

    # User input - Select Month and Year
    selected_month = st.selectbox("Select Month", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
    selected_year = st.selectbox("Select Year", ["2023", "2024"])

    # Create a function to preprocess input data for prediction based on the selected month and year
    def preprocess_input_data(selected_month, selected_year):
        global input_data  # Declare input_data as a global variable
        
        feats_to_lag = [col for col in input_data.columns if col not in ['Month', 'year_month']]
        for col in feats_to_lag:
            for i in range(1, 12):
                input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)
                    
        # Fill null values in lag columns with random values, sampled from the same column
        lag_columns = [col for col in input_data.columns if col.startswith("prev_")]
        for col in lag_columns:
            null_mask = input_data[col].isnull()
            num_nulls = null_mask.sum()
            if num_nulls > 0:
                random_values = np.random.choice(input_data[col].dropna(), num_nulls)
                input_data.loc[null_mask, col] = random_values

        input_data = input_data.drop(columns=target_cols + ['Total_Local Sales', 'Total_Export_Sales'])
        
        st.write(input_data)
        
        selected_date = f"{selected_year}-{selected_month}"
        selected_data = input_data[input_data['year_month'] == selected_date]

        return selected_data

    # Add a button to trigger predictions
    if st.button("Predict CPI"):
        
        input_data = preprocess_input_data(selected_month, selected_year)

        if not input_data.empty:
            input_scaled = scaler.transform(input_data.drop(columns=['Month', 'year_month']))

            lr_model = model_dict[category]
            predicted_cpi = lr_model.predict(input_scaled)
            
            # Display the predicted CPI value to the user with larger text using HTML styling
            st.markdown(f"<h2>Predicted CPI for {category} in {selected_year}-{selected_month}: {predicted_cpi[0]:.2f}</h2>", unsafe_allow_html=True)

            prev_month_key = category  # Use the category name as the key

            # Fetch previous month's prediction from the actual data
            prev_month_col = f"prev_1_month_{category}"
            prev_month = input_data.loc[input_data['year_month'] == f"{selected_year}-{selected_month}"][prev_month_col].values[0]

            # Update the dictionary with the previous month's prediction
            previous_month_predictions[prev_month_key] = prev_month

            prev_month_data = pd.DataFrame({
                'Month': ['Previous Month', 'Current Month'],
                'CPI Value': [prev_month, predicted_cpi[0]]
            })

            percentage_change = ((predicted_cpi[0] - prev_month) / prev_month) * 100

            if percentage_change > 0:
                change_icon = "📈"
                change_text = f"Increased by {percentage_change:.2f}%"
            elif percentage_change < 0:
                change_icon = "📉"
                change_text = f"Decreased by {abs(percentage_change):.2f}%"
            else:
                change_icon = "📊"
                change_text = "No change"

            # Display the change card/legend with larger text using HTML styling
            st.markdown(f"<h3>Change: {change_icon} {change_text}</h3>", unsafe_allow_html=True)

            fig, ax = plt.subplots()
            ax.bar(prev_month_data['Month'], prev_month_data['CPI Value'], color=['red', 'blue'])
            ax.set_xlabel('Month')
            ax.set_ylabel('CPI Value')
            ax.set_title(f'{category} CPI Comparison')

            for i, v in enumerate(prev_month_data['CPI Value']):
                ax.text(i, v, f'{v:.2f}', va='bottom', ha='center', fontsize=12)

            ax.tick_params(axis='x', which='both', bottom=False)
            ax.set_facecolor('#F5F5F5')

            st.pyplot(fig, use_container_width=True)
        
        else:
            st.write(f"No data available for {selected_year}-{selected_month}. Please select a different month and year.")

# Add the Dashboard part here
# Dashboard page
# Dashboard page

# Dashboard page
elif page == "Dashboard":
    st.header("Dashboard")
    
    # User input - Select Category for the dashboard
    selected_category = st.selectbox("Select Category for the Dashboard", target_cols)
    
    # Filter data for the selected category
    category_data = input_data[['year_month', selected_category]].copy()
    
    # Create a column for percentage change
    category_data['Percentage Change'] = (category_data[selected_category] - category_data[selected_category].shift(1)) / category_data[selected_category].shift(1) * 100
    
    # Display the selected category name
    st.write(f"Dashboard for: {selected_category}")
    
    # Create a subplot with two traces (bar and line)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar trace (CPI Values)
    fig.add_trace(
        go.Bar(x=category_data['year_month'], y=category_data[selected_category], name='CPI Values', marker_color='blue'),
        secondary_y=False,
    )
    
    # Add line trace (Percentage Change)
    fig.add_trace(
        go.Scatter(x=category_data['year_month'], y=category_data['Percentage Change'], mode='lines+markers', name='Percentage Change', line=dict(color='red')),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title=f'{selected_category} CPI and Percentage Change',
        xaxis_title="Month",
        yaxis_title="CPI Values",
        yaxis2_title="Percentage Change",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Set Y-axis limits for the visual under the dashboard page
    fig.update_yaxes(range=[97, 105], secondary_y=False)

    st.plotly_chart(fig, use_container_width=True)

    # Calculate the percentage contribution of each category in the CPI value
    contribution_data = input_data.copy()
    contribution_data['Year'], contribution_data['Month'] = contribution_data['year_month'].str.split('-', 1).str
    contribution_data = contribution_data.groupby(['Year', 'Month'])[target_cols].mean()
    contribution_data = (contribution_data / contribution_data.sum(axis=1, skipna=True).values.reshape(-1, 1)) * 100
    
    # Create a treemap chart for category contributions
    treemap_fig = go.Figure(go.Treemap(
        labels=[f"{year}-{month}" for year, month in contribution_data.index],
        parents=['' for _ in contribution_data.index],
        values=contribution_data[selected_category].values,
    ))

    # Customize the treemap layout
    treemap_fig.update_layout(
        title=f'{selected_category} Contribution to CPI Over Time',
    )

    # Display the treemap chart
    st.plotly_chart(treemap_fig, use_container_width=True)
