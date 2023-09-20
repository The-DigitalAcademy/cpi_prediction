# Streamlit app
def main():
    # Set the title
    st.title("CPI Vision")

    # Create a sidebar navigation
    menu = st.sidebar.radio("Navigation", ["Model", "Dashboard"])

    if menu == "Model":
        # Display the Model section
        st.header("Model")
        st.write("This is the Model section. You can load and manage models here.")

        # Add code for managing models here if needed

    elif menu == "Dashboard":
        # Display the Dashboard section
        st.header("Dashboard")

        # Allow the user to upload a PDF document
        uploaded_file = st.file_uploader("Upload a CPI PDF document", type=["pdf"])

        if uploaded_file is not None:
            # Process the uploaded PDF file
            st.text("Processing the uploaded PDF...")
            category_values = process_pdf(uploaded_file)

            # Allow the user to select categories for prediction
            selected_categories = st.multiselect(
                "Select categories to predict:", list(target_cols_with_prefixes.keys()), default=[list(target_cols_with_prefixes.keys())[0]]
            )

            # Display input fields for vehicle sales and currency
            st.write("Enter Vehicle Sales and Currency Input:")
            total_local_sales = st.number_input("Total_Local_Sales", value=0.0)
            total_export_sales = st.number_input("Total_Export_Sales", value=0.0)
            usd_zar = st.number_input("USD_ZAR", value=0.0)
            gbp_zar = st.number_input("GBP_ZAR", value=0.0)
            eur_zar = st.number_input("EUR_ZAR", value=0.0)

            # Load saved models
            loaded_models = load_models()

            # Allow the user to select which month they want to predict
            selected_month = st.selectbox("Select a month for prediction:", ["Next Month", "Two Months Later", "Three Months Later"])

            if st.button("Predict CPI"):
                # Dictionary to store predictions
                predictions = {}

                # Calculate the reference date based on the current date
                current_date = datetime.date.today()
                if selected_month == "Next Month":
                    reference_date = current_date.replace(month=current_date.month + 1)
                elif selected_month == "Two Months Later":
                    reference_date = current_date.replace(month=current_date.month + 2)
                elif selected_month == "Three Months Later":
                    reference_date = current_date.replace(month=current_date.month + 3)

                # Make predictions for the selected categories
                for selected_category in selected_categories:
                    input_data = create_input_data(selected_category, category_values, total_local_sales, total_export_sales, usd_zar, gbp_zar, eur_zar)
                    make_prediction(selected_category, input_data, loaded_models, selected_category.replace(' ', '_'), predictions, reference_date, selected_month)

                # Display predictions
                st.text(f"Predicted CPI values for {selected_month} for the selected categories:")
                for category, value in predictions.items():
                    st.text(f"{category}: {value}")

if __name__ == "__main__":
    main()
