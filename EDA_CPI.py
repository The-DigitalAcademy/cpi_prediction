import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Load your dataset
# Replace 'your_data.csv' with the path to your dataset
df = pd.read_csv('cleaned_data.csv')

# Create a temporary dataframe with Month and Year columns
df_temp = df.copy()
df_temp['Year'] = pd.to_datetime(df_temp['Month']).dt.year
df_temp['Month'] = pd.to_datetime(df_temp['Month']).dt.month

# Define the categories to plot
categories_to_plot = df_temp.columns[2:]  # Assuming the first two columns are Month and Year

# Create subplots
num_categories = len(categories_to_plot)
num_rows = math.ceil(num_categories / 2)
num_cols = 2
st.title('Seasonality of Categories')

# Create a dropdown to select the graph
selected_graph = st.selectbox('Select a Graph', ['Seasonality of Categories', 'Percentage Change in Food Prices'])

if selected_graph == 'Seasonality of Categories':
    num_rows = math.ceil(num_categories / num_cols)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
    
    for i, category in enumerate(categories_to_plot):
        row = i // num_cols
        col = i % num_cols
        ax = axs[row, col]
        
        sns.pointplot(x='Month', y=category, hue='Year', data=df_temp, ax=ax)
        ax.set_title(category)
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
    
    # Remove any unused subplots
    for i in range(num_categories, num_rows * num_cols):
        fig.delaxes(axs.flatten()[i])
    
    # Display the plot using Streamlit
    st.pyplot(fig)

elif selected_graph == 'Percentage Change in Food Prices':
    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Filter the data for January to April 2022
    start_date = '2022-01-31'
    end_date = '2023-03-30'
    filtered_df = df_copy[(df_copy['Month'] >= start_date) & (df_copy['Month'] <= end_date)]

    # Calculate percentage change for Food and non-alcoholic beverages
    filtered_df['Food_pct_change'] = ((filtered_df['Food and non-alcoholic beverages'] - filtered_df['Food and non-alcoholic beverages'].iloc[0]) / filtered_df['Food and non-alcoholic beverages'].iloc[0]) * 100

    # Plot the percentage change in Food and non-alcoholic beverages prices
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(filtered_df['Month'], filtered_df['Food_pct_change'], label='Food and Non-Alcoholic Beverages', marker='o', color='b')
    plt.xlabel('Month')
    plt.ylabel('Percentage Change (%)')
    plt.title('Percentage Change in Food and Non-Alcoholic Beverages Prices (Jan to Mar 2023)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the plot using Streamlit
    st.pyplot(fig)

elif selected_graph == 'Percentage Change in Food Prices':
    # Make a copy of the original DataFrame
    df_copy = df.copy()

    # Filter the data for January to April 2022
    start_date = '2022-01-31'
    end_date = '2023-03-30'
    filtered_df = df_copy[(df_copy['Month'] >= start_date) & (df_copy['Month'] <= end_date)]

    # Calculate percentage change for Food and non-alcoholic beverages
    filtered_df['Food_pct_change'] = ((filtered_df['Food and non-alcoholic beverages'] - filtered_df['Food and non-alcoholic beverages'].iloc[0]) / filtered_df['Food and non-alcoholic beverages'].iloc[0]) * 100

    # Plot the percentage change in Food and non-alcoholic beverages prices
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.plot(filtered_df['Month'], filtered_df['Food_pct_change'], label='Food and Non-Alcoholic Beverages', marker='o', color='b')
    plt.xlabel('Month')
    plt.ylabel('Percentage Change (%)')
    plt.title('Percentage Change in Food and Non-Alcoholic Beverages Prices (Jan to Mar 2023)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Display the plot using Streamlit
    st.pyplot(fig)

for i in range(num_categories, num_rows * num_cols):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')

    # Display an empty plot to fill the layout
    st.pyplot(fig)


