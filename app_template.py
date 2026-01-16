"""
Inventory Demand Forecasting - Streamlit App Template
=====================================================
This is a starter template for your Streamlit application.
Complete the TODO sections with your own code.

To run: streamlit run app_template.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
# import joblib  # Uncomment when you have a saved model

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# LOAD MODEL (TODO: Uncomment when ready)
# ============================================
# @st.cache_resource
# def load_model():
#     model = joblib.load('best_model.pkl')
#     scaler = joblib.load('scaler.pkl')
#     return model, scaler
#
# model, scaler = load_model()

# ============================================
# SAMPLE DATA FOR DEMONSTRATION
# ============================================
@st.cache_data
def load_sample_data():
    """Load sample data for visualizations"""
    # Sample historical data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    np.random.seed(42)

    historical_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(1000, 5000, len(dates)) + \
                 np.sin(np.arange(len(dates)) * np.pi / 6) * 500  # Seasonality
    })

    category_data = pd.DataFrame({
        'category': ['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty'],
        'sales': [45000, 38000, 62000, 28000, 22000]
    })

    region_data = pd.DataFrame({
        'region': ['North', 'South', 'East', 'West', 'Central'],
        'sales': [35000, 42000, 38000, 48000, 32000]
    })

    return historical_data, category_data, region_data

historical_data, category_data, region_data = load_sample_data()

# ============================================
# HEADER
# ============================================
st.title("📦 Inventory Demand Forecasting System")
st.markdown("""
    <style>
    .main-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    </style>
    <p class="main-header">
    Predict optimal stock levels to minimize costs and prevent stockouts.
    Powered by Machine Learning.
    </p>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUT PARAMETERS
# ============================================
st.sidebar.header("📊 Input Parameters")
st.sidebar.markdown("---")

# Product Category
category = st.sidebar.selectbox(
    "🏷️ Product Category",
    options=['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty'],
    help="Select the product category for prediction"
)

# Store Type
store_type = st.sidebar.selectbox(
    "🏪 Store Type",
    options=['Supermarket', 'Express', 'Hypermarket', 'Warehouse'],
    help="Select the type of store"
)

# Region
region = st.sidebar.selectbox(
    "🌍 Region",
    options=['North', 'South', 'East', 'West', 'Central'],
    help="Select the geographic region"
)

st.sidebar.markdown("---")

# Date
prediction_date = st.sidebar.date_input(
    "📅 Prediction Date",
    value=datetime.today() + timedelta(days=7),
    min_value=datetime.today(),
    max_value=datetime.today() + timedelta(days=90),
    help="Select the date for demand prediction"
)

# Promotion
is_promotion = st.sidebar.checkbox(
    "🎉 Promotional Period",
    help="Check if there's an active promotion"
)

# Current Stock
current_stock = st.sidebar.number_input(
    "📦 Current Stock Level",
    min_value=0,
    max_value=1000,
    value=100,
    step=10,
    help="Enter current inventory level"
)

st.sidebar.markdown("---")

# ============================================
# MAIN CONTENT - TABS
# ============================================
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📈 Historical Analysis", "🎯 Model Performance"])

# ============================================
# TAB 1: PREDICTION
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Demand Prediction")

        # Predict button
        predict_clicked = st.button("🔮 Generate Prediction", type="primary", use_container_width=True)

        if predict_clicked:
            # ============================================
            # TODO: Replace this with your actual model prediction
            # ============================================

            # Feature engineering (example)
            features = {
                'category': category,
                'store_type': store_type,
                'region': region,
                'month': prediction_date.month,
                'day_of_week': prediction_date.weekday(),
                'is_promotion': 1 if is_promotion else 0,
                'current_stock': current_stock
            }

            # Simulate prediction (replace with actual model)
            base_demand = {
                'Electronics': 80, 'Clothing': 120, 'Groceries': 200,
                'Home & Kitchen': 60, 'Beauty': 90
            }

            store_multiplier = {
                'Supermarket': 1.0, 'Express': 0.6, 'Hypermarket': 1.5, 'Warehouse': 2.0
            }

            promotion_boost = 1.3 if is_promotion else 1.0
            weekend_boost = 1.2 if prediction_date.weekday() >= 5 else 1.0

            prediction = int(
                base_demand[category] *
                store_multiplier[store_type] *
                promotion_boost *
                weekend_boost *
                np.random.uniform(0.9, 1.1)
            )

            # Display results
            st.markdown("---")

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    label="📊 Predicted Demand",
                    value=f"{prediction} units",
                    delta=f"{prediction - current_stock:+d} vs current"
                )

            with metric_col2:
                st.metric(
                    label="📦 Current Stock",
                    value=f"{current_stock} units"
                )

            with metric_col3:
                gap = prediction - current_stock
                if gap > 0:
                    status = "🔴 Restock Needed"
                else:
                    status = "🟢 Sufficient"
                st.metric(label="Status", value=status)

            # Recommendation box
            st.markdown("---")
            if prediction > current_stock:
                st.warning(f"""
                    ⚠️ **Action Required**

                    Based on the prediction, you should order **{prediction - current_stock} additional units**
                    to meet the expected demand and avoid stockouts.

                    - Expected Demand: {prediction} units
                    - Current Stock: {current_stock} units
                    - Shortfall: {prediction - current_stock} units
                """)
            else:
                st.success(f"""
                    ✅ **Stock Levels Adequate**

                    Current inventory is sufficient to meet the predicted demand.

                    - Expected Demand: {prediction} units
                    - Current Stock: {current_stock} units
                    - Surplus: {current_stock - prediction} units
                """)

    with col2:
        st.subheader("Input Summary")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Category | {category} |
        | Store Type | {store_type} |
        | Region | {region} |
        | Date | {prediction_date} |
        | Promotion | {'Yes' if is_promotion else 'No'} |
        | Current Stock | {current_stock} |
        """)

# ============================================
# TAB 2: HISTORICAL ANALYSIS
# ============================================
with tab2:
    st.subheader("Historical Sales Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Sales Trend
        fig_trend = px.line(
            historical_data,
            x='date',
            y='sales',
            title='Monthly Sales Trend (2023-2024)',
            labels={'sales': 'Sales (Units)', 'date': 'Month'}
        )
        fig_trend.update_layout(hovermode='x unified')
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        # Sales by Category
        fig_category = px.bar(
            category_data,
            x='category',
            y='sales',
            title='Sales by Product Category',
            color='category',
            labels={'sales': 'Sales (Units)', 'category': 'Category'}
        )
        st.plotly_chart(fig_category, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Sales by Region
        fig_region = px.pie(
            region_data,
            values='sales',
            names='region',
            title='Sales Distribution by Region',
            hole=0.4
        )
        st.plotly_chart(fig_region, use_container_width=True)

    with col4:
        # Key Statistics
        st.subheader("📊 Key Statistics")
        st.metric("Total Sales (2024)", "195,000 units", "+12% YoY")
        st.metric("Average Daily Sales", "534 units", "+8%")
        st.metric("Stockout Rate", "3.2%", "-1.5%")
        st.metric("Inventory Turnover", "12.4x", "+0.8")

# ============================================
# TAB 3: MODEL PERFORMANCE
# ============================================
with tab3:
    st.subheader("Model Performance Metrics")

    # Model comparison data
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting'],
        'RMSE': [52.3, 48.7, 45.2, 38.4, 35.1],
        'MAE': [41.2, 38.5, 35.8, 29.6, 27.3],
        'R2': [0.72, 0.76, 0.81, 0.87, 0.89]
    })

    col1, col2 = st.columns(2)

    with col1:
        # Performance metrics table
        st.markdown("### Model Comparison")
        st.dataframe(
            model_comparison.style.highlight_max(subset=['R2'], color='lightgreen')
                                   .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
            use_container_width=True
        )

        # Best model highlight
        st.success("🏆 **Best Model: Gradient Boosting** with R² = 0.89")

    with col2:
        # R² Score comparison chart
        fig_r2 = px.bar(
            model_comparison,
            x='Model',
            y='R2',
            title='Model Comparison (R² Score)',
            color='R2',
            color_continuous_scale='greens'
        )
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)

    # Feature Importance
    st.markdown("### Feature Importance")

    feature_importance = pd.DataFrame({
        'Feature': ['Historical Sales (7-day avg)', 'Day of Week', 'Is Promotion',
                   'Product Category', 'Month', 'Store Type', 'Region',
                   'Current Stock', 'Price', 'Season'],
        'Importance': [0.28, 0.18, 0.15, 0.12, 0.08, 0.06, 0.05, 0.04, 0.02, 0.02]
    }).sort_values('Importance', ascending=True)

    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Feature Importance',
        color='Importance',
        color_continuous_scale='blues'
    )
    st.plotly_chart(fig_importance, use_container_width=True)

    # Metric explanations
    with st.expander("📖 Understanding the Metrics"):
        st.markdown("""
        - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of prediction errors.
          Lower is better. Units are the same as the target variable.

        - **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values.
          Lower is better. Less sensitive to outliers than RMSE.

        - **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model.
          Range: 0 to 1. Higher is better. 0.89 means 89% of variance is explained.
        """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📦 Inventory Demand Forecasting System | ML Assignment Project</p>
        <p>Model trained on historical inventory data | Last updated: December 2026</p>
    </div>
""", unsafe_allow_html=True)

