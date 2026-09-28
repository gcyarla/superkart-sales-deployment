import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="SuperKart Sales Forecaster", layout="wide")

st.title("🏪 SuperKart Sales Revenue Forecasting Dashboard")
st.write("Real-time and batch sales forecasting powered by an optimized Tuned Random Forest Regressor microservice.")

# Set the backend endpoint
BACKEND_URL = "http://backend:7860"

tab1, tab2 = st.tabs(["🎯 Real-Time Prediction", "📊 Batch Forecasting"])

with tab1:
    st.subheader("Single Record Input parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        product_weight = st.number_input("Product Weight", min_value=0.0, max_value=100.0, value=12.66)
        product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        product_allocated_area = st.number_input("Product Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")

    with col2:
        product_mrp = st.number_input("Product MRP", min_value=0.0, value=117.08)
        store_size = st.selectbox("Store Size", ["Medium", "Small", "High"])
        store_location = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])

    with col3:
        store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])
        product_id_char = st.selectbox("Product ID Category Code", ["FD", "DR", "NC"])
        store_age = st.number_input("Store Age (Years)", min_value=0, value=16)

    if st.button("Forecast Sales", key="single_predict"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": product_sugar,
            "Product_Allocated_Area": product_allocated_area,
            "Product_MRP": product_mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": store_location,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age
        }
        try:
            res = requests.post(f"{BACKEND_URL}/v1/predict", json=payload)
            if res.status_code == 200:
                prediction = res.json()["prediction"]
                st.success(f"Predicted Product Store Sales Total: **${prediction:,.2f}**")
            else:
                st.error(f"Error: {res.json().get('error', 'Unknown Error')}")
        except Exception as e:
            st.error(f"Could not reach Backend Service: {e}")

with tab2:
    st.subheader("High-Throughput Batch Predictions")
    uploaded_file = st.file_uploader("Upload raw SuperKart csv dataset", type=["csv"])

    if uploaded_file is not None:
        df_input = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df_input.head())

        if st.button("Run Batch Forecast"):
            try:
                files = {'file': uploaded_file.getvalue()}
                res = requests.post(f"{BACKEND_URL}/v1/predictbatch", files=files)
                if res.status_code == 200:
                    predictions = res.json()
                    df_results = df_input.copy()
                    df_results["Predicted_Sales"] = [predictions[str(i)] for i in range(len(df_results))]
                    st.success("Batch Processing Completed!")
                    st.dataframe(df_results)

                    # Download Option
                    csv_output = df_results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predicted Dataset (CSV)",
                        data=csv_output,
                        file_name="SuperKart_Sales_Predictions.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Error: {res.json().get('error', 'Unknown Error')}")
            except Exception as e:
                st.error(f"Could not reach Backend Service: {e}")
