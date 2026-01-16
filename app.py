import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime, timedelta

# -----------------------------
# Load saved model, scaler, encoders
# -----------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

st.title("📦 Inventory Demand Forecasting System")
st.markdown("Predict optimal stock levels to minimize costs and prevent stockouts")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Parameters")

category = st.sidebar.selectbox(
    "Product Category",
    options=['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty']
)

store_type = st.sidebar.selectbox(
    "Store Type",
    options=['Supermarket', 'Express', 'Hypermarket', 'Warehouse']
)

region = st.sidebar.selectbox(
    "Region",
    options=['North', 'South', 'East', 'West', 'Central']
)

prediction_date = st.sidebar.date_input(
    "Prediction Date",
    value=datetime.today() + timedelta(days=7)
)

is_promotion = st.sidebar.checkbox("Is there a promotion?")

current_stock = st.sidebar.number_input(
    "Current Stock Level",
    min_value=0,
    max_value=1000,
    value=100
)

# -----------------------------
# Prediction Button
# -----------------------------
if st.sidebar.button("🔮 Predict Stock Requirement", type="primary"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Product_Category": [category],
        "Store_Type": [store_type],
        "Region": [region],
        "Date": [prediction_date],
        "Is_Promotion": [int(is_promotion)],
        "Current_Stock_Level": [current_stock]
    })

    # Encode categorical features
    for col in input_data.select_dtypes(include=["object"]).columns:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col])

    # Scale numeric features
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    # -----------------------------
    # Display Results
    # -----------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Units to Stock",
            value=f"{int(prediction)} units",
            delta=f"{int(prediction - current_stock)} from current"
        )

    with col2:
        st.metric(
            label="Current Stock",
            value=f"{current_stock} units"
        )

    with col3:
        status = "Restock Needed" if prediction > current_stock else "Stock Sufficient"
        st.metric(
            label="Status",
            value=status
        )

    if prediction > current_stock:
        st.warning(f"⚠️ Recommended to order {int(prediction - current_stock)} additional units")
    else:
        st.success("✅ Current stock levels are adequate")

# -----------------------------
# Tabs for Dashboard
# -----------------------------
tab1, tab2, tab3 = st.tabs(["Prediction", "Historical Analysis", "Model Performance"])

with tab2:
    st.subheader("📊 Historical Sales Analysis")
    demo_data = pd.DataFrame({
        "Category": ["Electronics", "Clothing", "Groceries", "Home & Kitchen", "Beauty"],
        "Avg_Sales": [120, 90, 150, 80, 60]
    })
    fig = px.bar(demo_data, x="Category", y="Avg_Sales", title="Average Sales by Category")
    st.plotly_chart(fig, width="stretch", key="hist_chart")

with tab3:
    st.subheader("📈 Model Performance Metrics")
    st.metric("RMSE", "5.6741")
    st.metric("MAE", "2.0121")
    st.metric("R² Score", "0.8791")

    perf_data = pd.DataFrame({
        "Model": ["Linear Regression", "Ridge", "Decision Tree", "Random Forest", "Gradient Boosting"],
        "R2_Score": [0.65, 0.70, 0.75, 0.82, 0.8791]
    })
    fig2 = px.bar(perf_data, x="Model", y="R2_Score", title="Model Comparison (R² Score)")
    st.plotly_chart(fig2, width="stretch", key="perf_chart")

    importance_data = pd.DataFrame({
        "Feature": ["Promotion", "Category", "Store_Type", "Region", "Current_Stock_Level"],
        "Importance": [0.25, 0.20, 0.18, 0.15, 0.12]
    })
    fig3 = px.bar(importance_data, x="Importance", y="Feature", orientation="h", title="Feature Importance")
    st.plotly_chart(fig3, width="stretch", key="importance_chart")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("*Model trained on cleaned inventory dataset | Last updated: 2026*")



# Load feature names
feature_names = joblib.load("feature_names.pkl")

# Initialize full feature vector with zeros
input_vector = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)

# Fill in user inputs
input_vector["Product_Category"] = label_encoders["Product_Category"].transform([category])
input_vector["Store_Type"] = label_encoders["Store_Type"].transform([store_type])
input_vector["Region"] = label_encoders["Region"].transform([region])
input_vector["Date"] = pd.to_datetime([prediction_date]).astype(int) / 10**9
input_vector["Is_Promotion"] = int(is_promotion)
input_vector["Current_Stock_Level"] = current_stock

# Scale and predict
input_scaled = scaler.transform(input_vector)
prediction = model.predict(input_scaled)[0]

