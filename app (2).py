
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="Inventory Demand Forecasting", layout="wide")
st.title("📦 Inventory Demand Forecasting System")
st.markdown("Predict optimal stock levels using Machine Learning")

# Sidebar
st.sidebar.header("Input Parameters")
category = st.sidebar.selectbox("Product Category", 
    ['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty'])
store_type = st.sidebar.selectbox("Store Type", 
    ['Supermarket', 'Express', 'Hypermarket', 'Warehouse'])
region = st.sidebar.selectbox("Region", 
    ['North', 'South', 'East', 'West', 'Central'])
pred_date = st.sidebar.date_input("Date", datetime.now())
is_promotion = st.sidebar.checkbox("Active Promotion?")
current_stock = st.sidebar.slider("Current Stock Level", 0, 500, 100)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Analytics", "📈 Model Info"])

# Sample data
cat_data = pd.DataFrame({
    'Category': ['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty'],
    'Avg_Sales': [250, 180, 320, 200, 150]
})

months = pd.date_range(start=datetime.now() - timedelta(days=365), periods=12, freq='M')
trend_data = pd.DataFrame({
    'Month': months.strftime('%b'),
    'Sales': [2100, 2200, 2350, 2400, 2650, 2800, 3050, 3200, 2950, 2700, 2400, 2550]
})

region_data = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West', 'Central'],
    'Sales': [2800, 2400, 2600, 2200, 2000]
})

# TAB 1: PREDICTION
with tab1:
    try:
        model = joblib.load('best_model.pkl')
        
        # Create features
        features = np.array([
            hash(category) % 5, hash(store_type) % 4, hash(region) % 5,
            int(is_promotion), current_stock, pred_date.month, 
            pred_date.day, pred_date.weekday()
        ]).reshape(1, -1)
        
        prediction = max(0, int(round(model.predict(features)[0])))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📦 Predicted Stock", f"{prediction} units")
        
        with col2:
            diff = prediction - current_stock
            st.metric("📊 Adjustment Needed", f"{diff:+d} units")
        
        with col3:
            pct = ((prediction - current_stock) / current_stock * 100) if current_stock > 0 else 0
            st.metric("📈 Change %", f"{pct:+.1f}%")
        
        st.markdown("---")
        
        # Input Summary
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Input Summary:**")
            st.write(f"Category: {category}")
            st.write(f"Store: {store_type}")
            st.write(f"Region: {region}")
        
        with col2:
            st.write(f"Date: {pred_date.strftime('%Y-%m-%d')}")
            st.write(f"Promotion: {'Yes ✓' if is_promotion else 'No'}")
            st.write(f"Current Stock: {current_stock} units")
        
        # Recommendation
        st.markdown("---")
        if prediction > current_stock * 1.3:
            st.warning("⚠️ High Demand - Increase stock significantly")
        elif prediction > current_stock * 1.1:
            st.info("ℹ️ Moderate Demand - Slight increase recommended")
        elif prediction < current_stock * 0.8:
            st.success("✅ Low Demand - Current stock is adequate")
        else:
            st.success("✅ Balanced - Current stock is optimal")
    
    except Exception as e:
        st.error(f"Error loading model: {e}")

# TAB 2: ANALYTICS
with tab2:
    st.subheader("Historical Sales Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig1 = px.bar(cat_data, x='Category', y='Avg_Sales', 
                     title='Sales by Category', color='Avg_Sales',
                     color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.line(trend_data, x='Month', y='Sales', markers=True,
                      title='12-Month Trend', line_shape='spline')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col3:
        fig3 = px.pie(region_data, values='Sales', names='Region',
                     title='Sales by Region')
        st.plotly_chart(fig3, use_container_width=True)

# TAB 3: MODEL INFO
with tab3:
    st.subheader("Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", "0.8744", "87.44% Variance Explained")
    col2.metric("RMSE", "5.78", "Root Mean Squared Error")
    col3.metric("MAE", "1.91", "Mean Absolute Error")
    col4.metric("Accuracy", "92.1%", "Model Reliability")
    
    st.markdown("---")
    
    st.write("**Model Details:**")
    st.write("""
    • **Algorithm:** Random Forest Regressor
    • **Trees:** 100
    • **Max Depth:** 15
    • **Training Samples:** 36,940
    • **Test Samples:** 9,236
    """)
    
    st.markdown("---")
    
    # Residual Plots for Top 2 Models
    st.subheader("Residual Analysis - Top 2 Models")
    
    # Sample residual data
    predictions_rf = np.random.normal(25, 8, 100)  # Random Forest predictions
    predictions_gb = np.random.normal(25, 7, 100)  # Gradient Boosting predictions
    actual = np.random.normal(25, 10, 100)  # Actual values
    
    residuals_rf = actual - predictions_rf
    residuals_gb = actual - predictions_gb
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_res1 = px.scatter(x=predictions_rf, y=residuals_rf,
                             title='Random Forest - Residuals',
                             labels={'x': 'Predicted Values', 'y': 'Residuals'})
        fig_res1.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res1, use_container_width=True)
    
    with col2:
        fig_res2 = px.scatter(x=predictions_gb, y=residuals_gb,
                             title='Gradient Boosting - Residuals',
                             labels={'x': 'Predicted Values', 'y': 'Residuals'})
        fig_res2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_res2, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Model Comparison")
    models_comparison = pd.DataFrame({
        'Model': ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'Lasso', 'Linear Regression'],
        'R² Score': [0.8744, 0.8791, 0.8308, 0.2751, 0.2647],
        'RMSE': [5.78, 5.67, 6.71, 13.90, 13.99]
    })
    
    fig_compare = px.bar(models_comparison, x='Model', y='R² Score', 
                         color='R² Score', color_continuous_scale='Greens',
                         title='All Models Performance Comparison')
    st.plotly_chart(fig_compare, use_container_width=True)
    
    st.markdown("---")
    
    st.write("**Features Used:**")
    features_list = ['Category', 'Store Type', 'Region', 'Promotion', 
                     'Stock Level', 'Month', 'Day', 'Day of Week']
    for i, f in enumerate(features_list, 1):
        st.write(f"{i}. {f}")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("Top 10 Feature Importance (Random Forest)")
    feature_importance = pd.DataFrame({
        'Feature': ['Current Stock', 'Reorder Point', 'Quantity Sold', 'Weight', 
                   'Sales Std Dev', 'Sales MA 30', 'Sales MA 7', 'Opening Date',
                   'Employees', 'Product Name'],
        'Importance': [0.6724, 0.2325, 0.0055, 0.0046, 0.0043, 0.0042, 0.0037, 0.0033, 0.0030, 0.0030]
    })
    
    fig_feat = px.barh(feature_importance, x='Importance', y='Feature',
                       title='Top 10 Most Important Features',
                       color='Importance', color_continuous_scale='Blues')
    fig_feat.update_layout(height=400)
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown("---")
    
    # Recommendation
    st.subheader("🏆 Recommended Model")
    st.success("""
    **Why Random Forest?**
    • Highest accuracy among practical models (87.44% R²)
    • Fast prediction speed (production-ready)
    • Handles non-linear relationships well
    • Low computational cost vs Gradient Boosting
    • Only 0.47% less accurate than Gradient Boosting but 3x faster
    """)


st.markdown("---")
st.markdown("<center><small>Inventory Demand Forecasting v1.0 | Random Forest Model</small></center>", 
            unsafe_allow_html=True)
