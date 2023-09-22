import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import StandardScaler

# Load your scaler and linear regression models for each target column
scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}


# Set page configuration and title
st.set_page_config(page_title=" CPI Vision Prediction App ", layout="wide")
# Add a brief description
# st.write("This app predicts whether a client is likely to default on their loan based on certain features.")


# Sidebar
with st.sidebar:
    from PIL import Image
    #image2 = Image.open('/Users/da_m1_23/Downloads/Manoko-loan-default-prediction/fraudimage.jpeg')
    image2=Image.open("CPI-logo.jpeg")
    st.image(image2, caption='  CPI  Prediction')

    #page_selection = st.selectbox("Navigation", 
                                #   ["Overview", "Step 1: Model", "Step 3: Output", "Contact Us"])
    
    page_selection = option_menu(
            menu_title=None,
            options=["Overview", "Model", "Contact Us"],
            icons=['file-earmark-text', 'graph-up', 'robot', 'file-earmark-spreadsheet', 'envelope'],
            menu_icon='cast',
            default_index=0,
            # orientation='horizontal',
            styles={"container": {'padding': '0!important', 'background_color': 'red'},
                    'icon': {'color': 'red', 'font-size': '18px'},
                    'nav-link': {
                        'font-size': '15px',
                        'text-align': 'left',
                        'margin': '0px',
                        '--hover-color': '#4BAAFF',
                    },
                    'nav-link-selected': {'background-color': '#6187D2'},
                    }
        )

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
input_data = input_data.iloc[3:].reset_index(drop=True)
#st.write(input_data)

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
            for i in range(1, 3):
                input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)
                    
        # Fill null values in lag columns with mean values, excluding target columns
        lag_columns = [col for col in input_data.columns if col.startswith("prev_")]
        input_data[lag_columns] = input_data[lag_columns].fillna(input_data[lag_columns].mean())

        input_data = input_data.drop(columns=target_cols + ['Total_Local Sales', 'Total_Export_Sales'])
        
        #st.write(input_data)
        
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

            prev_month = input_data[f"prev_1_month_{category}"].values[0]
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
