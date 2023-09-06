import streamlit as st
import pandas as pd
import numpy as np

col1, col2, col3 = st.columns(3)

with col1:
    st.write('USD/ZAR')

with col2:
    st.write('GBP/ZAR')

with col3:
    st.write('EUR/ZAR')

with st.sidebar:
    select = st.selectbox( '', ('Predictions', 'Analysis'))