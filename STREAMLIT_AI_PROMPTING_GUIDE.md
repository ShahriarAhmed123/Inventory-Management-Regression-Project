# Streamlit App Development Guide with AI Prompting

## Overview

This guide will help you create a Streamlit web application for your inventory demand prediction model using AI tools like ChatGPT, Claude, or GitHub Copilot.

---

## Step 1: Set Up Your Environment

### Install Required Libraries
```bash
pip install streamlit pandas numpy scikit-learn joblib plotly
```

### Test Streamlit Installation
```bash
streamlit hello
```

---

## Step 2: Save Your Trained Model

Before creating the app, save your best model from the ML notebook:

```python
import joblib

# After training your best model (e.g., Random Forest)
joblib.dump(best_model, 'best_model.pkl')

# Also save the scaler if you used one
joblib.dump(scaler, 'scaler.pkl')

# Save the label encoders for categorical features
joblib.dump(label_encoders, 'label_encoders.pkl')
```

---

## Step 3: AI Prompting Strategies

### Prompt 1: Basic App Structure

**Copy this prompt to your AI assistant:**

```
Create a Streamlit app for inventory demand prediction with the following requirements:

1. Page title: "Inventory Demand Forecasting System"
2. Sidebar with input widgets:
   - Product Category: dropdown with options ['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty']
   - Store Type: dropdown with options ['Supermarket', 'Express', 'Hypermarket', 'Warehouse']
   - Region: dropdown with options ['North', 'South', 'East', 'West', 'Central']
   - Date: date picker
   - Is Promotion: checkbox
   - Current Stock Level: number input (0-500)

3. Main area:
   - Display the predicted units to stock as a large metric
   - Show a confidence message based on the prediction

4. Load a pre-trained model from 'best_model.pkl' using joblib

5. Create features from the inputs to match the model's expected format

Include proper error handling and a clean, professional layout.
```

### Prompt 2: Adding Visualizations

**Follow-up prompt:**

```
Add the following visualizations to the Streamlit app:

1. A bar chart showing historical average sales by product category (use sample data)
2. A line chart showing sales trend over the past 12 months
3. A pie chart showing sales distribution by region

Use Plotly for interactive charts. Create sample data if needed for demonstration purposes.

Also add tabs to organize the content:
- Tab 1: Prediction
- Tab 2: Historical Analysis
- Tab 3: Model Performance
```

### Prompt 3: Model Performance Dashboard

**Additional prompt:**

```
Add a "Model Performance" section to the Streamlit app that displays:

1. Model accuracy metrics in a formatted table:
   - RMSE: 45.2
   - MAE: 32.1
   - R² Score: 0.847

2. A bar chart comparing 5 different models (Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting) with their R² scores

3. Feature importance chart (horizontal bar chart) showing the top 10 most important features

4. A brief explanation of what each metric means

Use st.metric() for displaying key numbers and st.expander() for detailed explanations.
```

### Prompt 4: User Input Validation

**Enhancement prompt:**

```
Add input validation to the Streamlit app:

1. Check if the date is not in the future (beyond 30 days)
2. Validate that current stock level is non-negative
3. Show warning messages using st.warning() for invalid inputs
4. Disable the predict button until all required fields are filled

Also add a "Reset" button to clear all inputs to default values.
```

### Prompt 5: Export and Download Features

**Final enhancement prompt:**

```
Add export functionality to the Streamlit app:

1. A button to download the prediction results as a CSV file
2. A button to download a PDF report (use simple text formatting)
3. Session state to store prediction history
4. A table showing the last 10 predictions made in the current session

Use st.download_button() for the download features and st.session_state for storing history.
```

---

## Step 4: Sample App Template

Here's a starter template you can build upon:

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta

# Page configuration
st.set_page_config(
    page_title="Inventory Demand Forecasting",
    page_icon="📦",
    layout="wide"
)

# Title
st.title("📦 Inventory Demand Forecasting System")
st.markdown("Predict optimal stock levels to minimize costs and prevent stockouts")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Product Category
category = st.sidebar.selectbox(
    "Product Category",
    options=['Electronics', 'Clothing', 'Groceries', 'Home & Kitchen', 'Beauty']
)

# Store Type
store_type = st.sidebar.selectbox(
    "Store Type",
    options=['Supermarket', 'Express', 'Hypermarket', 'Warehouse']
)

# Region
region = st.sidebar.selectbox(
    "Region",
    options=['North', 'South', 'East', 'West', 'Central']
)

# Date
prediction_date = st.sidebar.date_input(
    "Prediction Date",
    value=datetime.today() + timedelta(days=7)
)

# Promotion
is_promotion = st.sidebar.checkbox("Is there a promotion?")

# Current Stock
current_stock = st.sidebar.number_input(
    "Current Stock Level",
    min_value=0,
    max_value=1000,
    value=100
)

# Predict button
if st.sidebar.button("🔮 Predict Stock Requirement", type="primary"):

    # Create feature dictionary
    # TODO: Add your feature engineering here to match your model

    # For demonstration, using a random prediction
    prediction = np.random.randint(50, 200)

    # Display results
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Predicted Units to Stock",
            value=f"{prediction} units",
            delta=f"{prediction - current_stock} from current"
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

    # Recommendation
    if prediction > current_stock:
        st.warning(f"⚠️ Recommended to order {prediction - current_stock} additional units")
    else:
        st.success("✅ Current stock levels are adequate")

# Footer
st.markdown("---")
st.markdown("*Model trained on historical inventory data | Last updated: 2024*")
```

---

## Step 5: Running Your App

```bash
# Navigate to your app directory
cd path/to/your/app

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## Common Issues and Solutions

### Issue 1: Model Loading Error
```python
# Wrong
model = joblib.load('model.pkl')

# Right - use full path or ensure file is in same directory
import os
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
model = joblib.load(model_path)
```

### Issue 2: Feature Mismatch
Make sure your input features match exactly what the model was trained on:
```python
# Check expected features
print(model.feature_names_in_)  # For sklearn models
```

### Issue 3: Categorical Encoding
```python
# If you used Label Encoding during training
category_encoded = label_encoder.transform([category])[0]
```

---

## Grading Criteria for Streamlit App (10 Points)

| Criteria | Points |
|----------|--------|
| App runs without errors | 2 |
| User inputs work correctly | 2 |
| Prediction displays properly | 2 |
| At least one visualization | 2 |
| Clean UI/UX design | 2 |
| **Total** | **10** |

---

## Bonus Features (Extra Credit)

- Real-time data connection
- Multiple model comparison toggle
- Email alert system mockup
- Dark/Light theme toggle
- Multi-language support
- Batch prediction upload (CSV file)

---

## Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cheat Sheet](https://docs.streamlit.io/library/cheatsheet)
- [Plotly + Streamlit](https://docs.streamlit.io/library/api-reference/charts/st.plotly_chart)
- [Streamlit Components Gallery](https://streamlit.io/components)

---

## Document Your AI Prompts

**Important:** In your submission, include a file called `AI_PROMPTS_USED.md` documenting:

1. Which AI tool you used (ChatGPT, Claude, Copilot, etc.)
2. The exact prompts you used
3. What modifications you made to the AI-generated code
4. Any challenges you faced and how you resolved them

This helps us understand your problem-solving process and AI collaboration skills.
