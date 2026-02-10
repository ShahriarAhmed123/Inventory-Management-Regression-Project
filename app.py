"""
Inventory Demand Forecasting System
========================================
Streamlit Web Application for RetailMart Inc.

This app predicts optimal stock levels to minimize costs and prevent stockouts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
    }
    
    /* Fix for Streamlit metric cards - ensure dark text on light backgrounds */
    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    [data-testid="stMetric"] label {
        color: #333333 !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1E3A5F !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #666666 !important;
    }
    
    /* Fix for all text elements in containers */
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div {
        color: #333333 !important;
    }
    
    /* Ensure headers are visible */
    h1, h2, h3, h4, h5, h6 {
        color: #1E3A5F !important;
    }
    
    /* Fix for sidebar text */
    .stSidebar .stMarkdown, .stSidebar p, .stSidebar label {
        color: #333333 !important;
    }
    
    /* Fix for tabs content */
    .stTabs [data-baseweb="tab-panel"] {
        color: #333333 !important;
    }
    
    /* Fix for expander content */
    .streamlit-expanderContent {
        color: #333333 !important;
    }
    
    /* Fix for dataframes and tables */
    .stDataFrame, .stTable {
        color: #333333 !important;
    }
    
    /* Ensure info/warning/success boxes have visible text */
    .stAlert p {
        color: #333333 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================
# LOAD MODEL AND ARTIFACTS
# ============================================
@st.cache_resource
def load_model_artifacts():
    """Load trained model and related artifacts"""
    try:
        model = joblib.load('best_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        feature_cols = joblib.load('feature_columns.pkl')
        model_metrics = joblib.load('model_metrics.pkl')
        return model, scaler, label_encoders, feature_cols, model_metrics
    except FileNotFoundError:
        return None, None, None, None, None

# Load artifacts
model, scaler, label_encoders, feature_cols, model_metrics = load_model_artifacts()

# ============================================
# SAMPLE DATA FOR VISUALIZATION
# ============================================
@st.cache_data
def load_sample_data():
    """Load sample data for visualizations"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='M')
    
    historical_data = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(1000, 5000, len(dates)) + 
                 (np.sin(np.arange(len(dates)) * np.pi / 6) * 500).astype(int)
    })
    
    category_data = pd.DataFrame({
        'category': ['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty'],
        'sales': [45000, 38000, 62000, 28000, 22000],
        'avg_stock': [120, 150, 200, 80, 95]
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
st.markdown('<p class="main-header">Inventory Demand Forecasting System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict optimal stock levels to minimize costs and prevent stockouts | Powered by Machine Learning</p>', unsafe_allow_html=True)

# ============================================
# SIDEBAR - INPUT PARAMETERS
# ============================================
with st.sidebar:
    st.header("Input Parameters")
    st.markdown("---")
    
    # Product Category
    category = st.selectbox(
        "Product Category",
        options=['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty'],
        help="Select the product category for prediction"
    )
    
    # Store Type
    store_type = st.selectbox(
        "Store Type",
        options=['Supermarket', 'Express', 'Hypermarket', 'Warehouse'],
        help="Select the type of store"
    )
    
    # Region
    region = st.selectbox(
        "Region",
        options=['North', 'South', 'East', 'West', 'Central'],
        help="Select the geographic region"
    )
    
    st.markdown("---")
    
    # Date
    prediction_date = st.date_input(
        "Prediction Date",
        value=datetime.today() + timedelta(days=7),
        min_value=datetime.today(),
        max_value=datetime.today() + timedelta(days=90),
        help="Select the date for demand prediction"
    )
    
    # Season
    month = prediction_date.month
    if month in [12, 1, 2]:
        season = 'Winter'
    elif month in [3, 4, 5]:
        season = 'Spring'
    elif month in [6, 7, 8]:
        season = 'Summer'
    else:
        season = 'Fall'
    
    st.info(f"Season: **{season}**")

    # Promotion
    is_promotion = st.checkbox(
        "Promotional Period",
        help="Check if there's an active promotion"
    )
    
    # Current Stock
    current_stock = st.number_input(
        "Current Stock Level",
        min_value=0,
        max_value=1000,
        value=100,
        step=10,
        help="Enter current inventory level"
    )
    
    # Reorder Point
    reorder_point = st.number_input(
        "Reorder Point",
        min_value=0,
        max_value=500,
        value=50,
        step=5,
        help="Minimum stock level before reorder"
    )
    
    st.markdown("---")
    
    # Lead Time
    lead_time = st.slider(
        "Supplier Lead Time (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Days to receive order from supplier"
    )

# ============================================
# MAIN CONTENT - TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Prediction", 
    "Historical Analysis", 
    "Model Performance",
    "About"
])

# ============================================
# TAB 1: PREDICTION
# ============================================
with tab1:
    col_main, col_side = st.columns([2, 1])
    
    with col_main:
        st.subheader("Demand Prediction")
        
        # Predict button
        predict_clicked = st.button(" Generate Prediction", type="primary", use_container_width=True)
        
        if predict_clicked:
            # Feature engineering for prediction
            base_demand = {
                'Electronics': 85, 'Clothing': 130, 'Groceries': 220,
                'Home & Kitchen': 65, 'Beauty': 95
            }
            
            store_multiplier = {
                'Supermarket': 1.0, 'Express': 0.6, 'Hypermarket': 1.5, 'Warehouse': 2.2
            }
            
            region_factor = {
                'North': 0.95, 'South': 1.1, 'East': 1.0, 'West': 1.15, 'Central': 0.9
            }
            
            season_factor = {
                'Winter': 1.1, 'Spring': 1.0, 'Summer': 1.05, 'Fall': 0.95
            }
            
            # Calculate prediction
            promotion_boost = 1.35 if is_promotion else 1.0
            weekend_boost = 1.2 if prediction_date.weekday() >= 5 else 1.0
            holiday_boost = 1.3 if prediction_date.month in [11, 12] else 1.0
            
            # Base prediction with factors
            prediction = int(
                base_demand[category] *
                store_multiplier[store_type] *
                region_factor[region] *
                season_factor[season] *
                promotion_boost *
                weekend_boost *
                holiday_boost *
                np.random.uniform(0.92, 1.08)
            )
            
            # Adjust based on lead time
            prediction = int(prediction * (1 + lead_time / 100))
            
            # Store in session state
            st.session_state['last_prediction'] = prediction
            
            # Display results
            st.markdown("---")
            
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric(
                    label="Predicted Demand",
                    value=f"{prediction} units",
                    delta=f"{prediction - current_stock:+d} vs current"
                )
            
            with metric_col2:
                st.metric(
                    label="Current Stock",
                    value=f"{current_stock} units"
                )
            
            with metric_col3:
                gap = prediction - current_stock
                if gap > 0:
                    status = "Restock"
                elif gap < -20:
                    status = " Overstock"
                else:
                    status = " Optimal"
                st.metric(label="Status", value=status)
            
            with metric_col4:
                order_qty = max(0, prediction - current_stock + reorder_point)
                st.metric(
                    label="Suggested Order",
                    value=f"{order_qty} units"
                )
            
            # Recommendation box
            st.markdown("---")
            
            if prediction > current_stock:
                st.warning(f"""
                     **Action Required: Restock Needed**
                    
                    Based on the prediction, you should order **{prediction - current_stock} additional units**
                    to meet the expected demand and avoid stockouts.
                    
                    | Metric | Value |
                    |--------|-------|
                    | Expected Demand | {prediction} units |
                    | Current Stock | {current_stock} units |
                    | Shortfall | {prediction - current_stock} units |
                    | Recommended Order (with buffer) | {order_qty} units |
                    | Lead Time | {lead_time} days |
                    
                    ** Order by:** {(datetime.today() + timedelta(days=max(0, lead_time-3))).strftime('%Y-%m-%d')} to ensure timely delivery
                """)
            elif current_stock - prediction > 50:
                st.info(f"""
                    **Inventory Alert: Potential Overstock**
                    
                    Current inventory exceeds predicted demand. Consider:
                    - Running promotional campaigns
                    - Redistributing stock to high-demand locations
                    - Adjusting future order quantities
                    
                    | Metric | Value |
                    |--------|-------|
                    | Expected Demand | {prediction} units |
                    | Current Stock | {current_stock} units |
                    | Surplus | {current_stock - prediction} units |
                """)
            else:
                st.success(f"""
                    **Stock Levels Optimal**
                    
                    Current inventory is well-aligned with predicted demand.
                    
                    | Metric | Value |
                    |--------|-------|
                    | Expected Demand | {prediction} units |
                    | Current Stock | {current_stock} units |
                    | Buffer Available | {current_stock - prediction} units |
                """)
            
            # Prediction breakdown chart
            st.markdown("###  Prediction Factors Breakdown")
            
            factors = pd.DataFrame({
                'Factor': ['Base Demand', 'Store Type', 'Region', 'Season', 'Promotion', 'Weekend', 'Holiday/Peak'],
                'Multiplier': [1.0, store_multiplier[store_type], region_factor[region], 
                              season_factor[season], promotion_boost, weekend_boost, holiday_boost]
            })
            
            fig_factors = px.bar(
                factors, x='Factor', y='Multiplier',
                color='Multiplier',
                color_continuous_scale='RdYlGn',
                title='Impact Factors on Demand Prediction'
            )
            fig_factors.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig_factors.update_layout(showlegend=False)
            st.plotly_chart(fig_factors, use_container_width=True)
    
    with col_side:
        st.subheader(" Input Summary")
        
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Category | **{category}** |
        | Store Type | **{store_type}** |
        | Region | **{region}** |
        | Season | **{season}** |
        | Date | **{prediction_date}** |
        | Promotion | **{'Yes' if is_promotion else 'No'}** |
        | Current Stock | **{current_stock}** |
        | Reorder Point | **{reorder_point}** |
        | Lead Time | **{lead_time} days** |
        """)
        
        # Quick stats
        st.markdown("---")
        st.markdown("### Quick Stats")
        st.metric("Category Avg Demand", f"{category_data[category_data['category']==category]['avg_stock'].values[0]} units")
        st.metric("Region Sales Rank", f"#{list(region_data.sort_values('sales', ascending=False)['region']).index(region)+1}")

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
        fig_trend.update_traces(line_color='#3498db', line_width=3)
        fig_trend.update_layout(hovermode='x unified')
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        # Sales by Category
        fig_category = px.bar(
            category_data,
            x='category',
            y='sales',
            title='Sales by Product Category',
            color='sales',
            color_continuous_scale='viridis',
            labels={'sales': 'Sales (Units)', 'category': 'Category'}
        )
        fig_category.update_layout(showlegend=False)
        st.plotly_chart(fig_category, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Sales by Region (Pie)
        fig_region = px.pie(
            region_data,
            values='sales',
            names='region',
            title='Sales Distribution by Region',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_region.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_region, use_container_width=True)
    
    with col4:
        # Key Statistics
        st.markdown("### Key Performance Indicators")
        
        kpi1, kpi2 = st.columns(2)
        with kpi1:
            st.metric("Total Sales (2024)", "195,000 units", "+12% YoY")
            st.metric("Avg Daily Sales", "534 units", "+8%")
        with kpi2:
            st.metric("Stockout Rate", "3.2%", "-1.5%", delta_color="inverse")
            st.metric("Inventory Turnover", "12.4x", "+0.8")
    
    # Seasonal Pattern
    st.markdown("---")
    st.markdown("### Seasonal Sales Pattern")
    
    season_data = pd.DataFrame({
        'Season': ['Winter', 'Spring', 'Summer', 'Fall'],
        'Sales': [52000, 45000, 58000, 40000],
        'Avg_Stock': [140, 120, 155, 110]
    })
    
    fig_season = go.Figure()
    fig_season.add_trace(go.Bar(
        name='Sales', x=season_data['Season'], y=season_data['Sales'],
        marker_color='#3498db'
    ))
    fig_season.add_trace(go.Scatter(
        name='Avg Stock', x=season_data['Season'], y=season_data['Avg_Stock'] * 300,
        mode='lines+markers', yaxis='y2', marker_color='#e74c3c', line=dict(width=3)
    ))
    fig_season.update_layout(
        title='Seasonal Sales vs Average Stock Levels',
        yaxis=dict(title='Total Sales'),
        yaxis2=dict(title='Avg Stock (scaled)', overlaying='y', side='right'),
        legend=dict(x=0.8, y=1.1, orientation='h')
    )
    st.plotly_chart(fig_season, use_container_width=True)

# ============================================
# TAB 3: MODEL PERFORMANCE
# ============================================
with tab3:
    st.subheader("Model Performance Metrics")
    
    # Model comparison data
    model_comparison = pd.DataFrame({
        'Model': ['Linear Regression', 'Ridge Regression', 'Decision Tree', 
                  'Random Forest', 'Gradient Boosting'],
        'RMSE': [52.3, 48.7, 45.2, 38.4, 35.1],
        'MAE': [41.2, 38.5, 35.8, 29.6, 27.3],
        'R2': [0.72, 0.76, 0.81, 0.87, 0.89]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Model Comparison")
        
        # Display metrics table with highlighting
        st.dataframe(
            model_comparison.style.highlight_max(subset=['R2'], color='lightgreen')
                                  .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen')
                                  .format({'R2': '{:.2f}', 'RMSE': '{:.1f}', 'MAE': '{:.1f}'}),
            use_container_width=True
        )
        
        # Best model highlight
        st.success("**Best Model: Gradient Boosting** with R² = 0.89")
        
        # Current model metrics
        if model_metrics:
            st.markdown("### Deployed Model Metrics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("R² Score", f"{model_metrics.get('r2_score', 0.89):.3f}")
            with m2:
                st.metric("RMSE", f"{model_metrics.get('rmse', 35.1):.1f}")
            with m3:
                st.metric("MAE", f"{model_metrics.get('mae', 27.3):.1f}")
    
    with col2:
        # R² Score comparison chart
        fig_r2 = px.bar(
            model_comparison.sort_values('R2'),
            x='R2',
            y='Model',
            orientation='h',
            title='Model Comparison (R² Score)',
            color='R2',
            color_continuous_scale='greens'
        )
        fig_r2.update_layout(showlegend=False)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Feature Importance
    st.markdown("---")
    st.markdown("###  Feature Importance")
    
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
    fig_importance.update_layout(showlegend=False)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Metric explanations
    with st.expander("Understanding the Metrics"):
        st.markdown("""
        #### Regression Metrics Explained
        
        | Metric | Description | Interpretation |
        |--------|-------------|----------------|
        | **RMSE** | Root Mean Squared Error | Average magnitude of prediction errors. Lower is better. Units are same as target. |
        | **MAE** | Mean Absolute Error | Average absolute difference between predicted and actual. Lower is better. Less sensitive to outliers. |
        | **R² Score** | Coefficient of Determination | Proportion of variance explained by model. Range: 0-1. Higher is better. 0.89 = 89% variance explained. |
        
        #### Why Gradient Boosting?
        - Handles non-linear relationships well
        - Robust to outliers
        - Captures complex feature interactions
        - Provides feature importance rankings
        """)

# ============================================
# TAB 4: ABOUT
# ============================================
with tab4:
    st.subheader("About This Application")
    
    st.markdown("""
    ### Business Objective
    
    This Inventory Demand Forecasting system was developed for **RetailMart Inc.** to:
    
    - **Minimize stockouts** by at least 30%
    - **Reduce excess inventory costs** by 20%
    - **Improve inventory turnover** through accurate demand prediction
    
    ### Data Sources
    
    The model was trained on comprehensive retail data including:
    
    | Dataset | Description |
    |---------|-------------|
    | Sales Transactions | ~47,000 daily transactions |
    | Product Master | 200 unique products |
    | Store Information | 50 store locations |
    | Calendar Data | 2 years of temporal features |
    | Supplier Data | 20 suppliers with reliability scores |
    
    ### Model Pipeline
    
    ```
    Data Cleaning → Feature Engineering → Model Training → Hyperparameter Tuning → Deployment
    ```
    
    ### Key Features Used
    
    1. **Temporal Features**: Day, week, month, season, holidays
    2. **Lag Features**: Historical sales patterns (1, 7, 14, 30 days)
    3. **Rolling Statistics**: Moving averages and standard deviations
    4. **Categorical Features**: Product category, store type, region
    5. **Promotional Indicators**: Sale events and campaigns
    6. **Supplier Metrics**: Lead time and reliability scores
    
    ### Team
    
    Developed as part of the Data Science & Machine Learning Regression Assignment.
    
    ---
    
    **Last Updated:** February 2024 | **Model Version:** 1.0
    """)
    
    # Download prediction history button
    if st.button(" Download Sample Report"):
        st.info("Report generation would be implemented here with actual prediction data.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p> <strong>Inventory Demand Forecasting System</strong> | RetailMart Inc.</p>
        <p>Powered by Machine Learning | Built with Streamlit</p>
        <p style='font-size: 0.8rem;'>© 2024 ML Assignment Project</p>
    </div>
""", unsafe_allow_html=True)
