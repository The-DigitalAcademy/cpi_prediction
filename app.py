import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image
import datetime
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pickle
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt 
from streamlit_option_menu import option_menu

scaler = joblib.load("last_scaler.pkl")
target_cols = ['Alcoholic beverages and tobacco', 'Clothing and footwear', 'Communication', 'Education', 'Food and non-alcoholic beverages', 'Headline_CPI', 'Health', 'Household contents and services', 'Housing and utilities', 'Miscellaneous goods and services', 'Recreation and culture', 'Restaurants and hotels ', 'Transport']
model_dict = {target_col: joblib.load(f"{target_col}_model.pkl") for target_col in target_cols}

def resize_image(image_path, size=(150, 150)):
    image = Image.open(image_path)
    image = image.resize(size)
    return image

st.set_page_config(page_title="CPI Vision Prediction", layout="wide")

current_date = datetime.date(2023, 4, 30)
end_date = datetime.date(2025, 12, 30)

data_list = []

while current_date <= end_date:
    year_month = current_date.strftime('%Y-%m')
    month = current_date.strftime('%Y-%m-%d')
    new_row = {'year_month': year_month, 'Month': month}
    data_list.append(new_row)
    current_date = current_date.replace(day=1) + pd.DateOffset(months=1)

input_data = pd.DataFrame(data_list)

def load_data():
    data = pd.read_csv('Book6.csv')
    return data

cpi_data = load_data()

previous_month_predictions = {category: None for category in target_cols}

def meet_the_team():
    st.title("Meet the Team")
    team_members = [
        {"name": "Sibongile Mokoena", "position": "Junior Data Scientist", "image": "Sibongile.jpeg", "description": "Sibongile is a data scientist with expertise in machine learning and data analysis."},
        {"name": "Manoko Langa", "position": "Web Developer", "image": "manoko.jpeg", "description": "Manoko is a web developer responsible for creating the Streamlit app."},
        {"name": "Zandile Mdiniso", "position": "Data Scientist", "image": "zand.jpeg", "description": "Similar to Manoko, Zandile is a data scientist with expertise in data analysis and machine learning."},
        {"name": "Thando Vilakazi", "position": "Business Analyst", "image": "thando.jpeg", "description": "Thando is a business analyst responsible for the valuable insights extracted in this project."},
        {"name": "Zweli Khumalo", "position": "Business Analyst", "image": "zweli.jpeg", "description": "Zweli is a business analyst responsible for the valuable insights extracted in this project."},
    ]

    columns = st.columns(len(team_members))
    for i, member in enumerate(team_members):
        with columns[i]:
            st.image(resize_image(member['image']), caption=member['name'], use_column_width=True)
            st.write(f"**{member['name']}**")
            st.write(f"**Position**: {member['position']}")

if page_selection == "Overview":
    st.title("CPI Vision Application Overview")
    st.write("Welcome to the CPI (Consumer Price Index) Overview page.")
    st.write("This page provides a general overview of CPI data and the purpose of this application.")
    st.header("What is CPI?")
    st.write("The Consumer Price Index (CPI) is a measure of the average change over time in the prices paid by urban consumers for a market basket of consumer goods and services. It is a crucial indicator of inflation and economic stability.")
    st.header("Purpose of the App")
    st.write("The CPI Vision App is designed to assist in predicting future CPI values for various categories based on historical data. Users can select a category, a month, and a year to get predictions.")
    st.write("The data used in this application is sourced from the CPI Nowcast Challenge, and it covers the period from January 2022 to March 2023.")

elif page_selection == "Prediction":
    st.title("CPI Vision App")
    st.header("Predict CPI")
    category = st.selectbox("Select Category", target_cols)
    selected_month = st.selectbox("Select Month", ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"])
    selected_year = st.selectbox("Select Year", ["2023", "2024", "2025"])

    def preprocess_input_data(selected_month, selected_year):
        global input_data
        feats_to_lag = [col for col in input_data.columns if col not in ['Month', 'year_month']]
        for col in feats_to_lag:
            for i in range(1, 12):
                input_data[f"prev_{i}_month_{col}"] = input_data[col].shift(i)
        lag_columns = [col for col in input_data.columns if col.startswith("prev_")]
        for col in lag_columns:
            null_mask = input_data[col].isnull()
            num_nulls = null_mask.sum()
            if num_nulls > 0:
                random_values = np.random.choice(input_data[col].dropna(), num_nulls)
                input_data.loc[null_mask, col] = random_values
        input_data = input_data.drop(columns=target_cols + ['Total_Local Sales', 'Total_Export_Sales'])
        selected_date = f"{selected_year}-{selected_month}"
        selected_data = input_data[input_data['year_month'] == selected_date]
        return selected_data

    if st.button("Predict CPI"):
        input_data = preprocess_input_data(selected_month, selected_year)
        if not input_data.empty:
            input_scaled = scaler.transform(input_data.drop(columns=['Month', 'year_month']))
            lr_model = model_dict[category]
            predicted_cpi = lr_model.predict(input_scaled)
            st.markdown(f"<h2>Predicted CPI for {category} in {selected_year}-{selected_month}: {predicted_cpi[0]:.2f}</h2>", unsafe_allow_html=True)
            prev_month_key = category
            prev_month_col = f"prev_1_month_{category}"
            prev_month = input_data.loc[input_data['year_month'] == f"{selected_year}-{selected_month}"][prev_month_col].values[0]
            previous_month_predictions[prev_month_key] = prev_month
            prev_month_data = pd.DataFrame({
                'Month': ['Previous Month', 'Current Month'],
                'CPI Value': [prev_month, predicted_cpi[0]]
            })
            percentage_change = ((predicted_cpi[0] - prev_month) / prev_month) * 100
            if percentage_change > 0:
                change_icon = "📈"
                change_text = f"Increased by {percentage_change:.2f}%
               elif percentage_change < 0:
                change_icon = "📉"
                change_text = f"Decreased by {abs(percentage_change):.2f}%"
            else:
                change_icon = "📊"
                change_text = "No change"
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

elif page_selection == "Dashboard":
    st.header("Dashboard Insights")
    selected_year = st.slider('Select a Year', min_value=int(cpi_data['Year'].min()), max_value=int(cpi_data['Year'].max()))
    selected_category = st.selectbox('Select a Category', cpi_data['Category'].unique())
    filtered_data = cpi_data[(cpi_data['Year'] == selected_year) & (cpi_data['Category'] == selected_category)]
    filtered_data['MONTH'] = pd.to_datetime(filtered_data['MONTH'], format='%B')
    filtered_data = filtered_data.sort_values(by='MONTH')
    month_names = filtered_data['MONTH'].dt.strftime('%B')
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=month_names, y=filtered_data['Value'], name='Value', marker_color='blue'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=month_names, y=filtered_data['Percentage Change (From Prior Month)'], mode='lines+markers', name='Percentage Change (From Prior Month)', line=dict(color='red')),
        secondary_y=True,
    )
    fig.update_layout(
        title='CPI and Percentage Change',
        xaxis_title="Month",
        yaxis_title="CPI Values",
        yaxis2_title="Percentage Change",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode='x',
    )
    fig.update_yaxes(range=[97, 105], secondary_y=False)
    st.subheader('CPI and Percentage Change Visualization')
    st.plotly_chart(fig, use_container_width=True)

elif page_selection == "Meet the Team":
    meet_the_team()

elif page_selection == "Contact Us":
    st.title('Contact Us!')
    st.markdown("Have a question or want to get in touch with us? Please fill out the form below with your email address, and we'll get back to you as soon as possible. We value your privacy and assure you that your information will be kept confidential.")
    st.markdown("By submitting this form, you consent to receiving email communications from us regarding your inquiry. We may use the email address you provide to respond to your message and provide any necessary assistance or information.")
