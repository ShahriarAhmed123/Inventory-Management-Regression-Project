# 📦 Inventory Demand Forecasting System

## 🌐 Live Application
Access the deployed app here:  
👉 [Inventory Demand Forecasting App](https://inventory-management-regression-project-v3is6qqhmwppfx4kniaxgy.streamlit.app/)

---
## 🖼️ Tech Stack & Badges ![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python&logoColor=white) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5-orange?logo=scikitlearn&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-Visualization-lightblue?logo=plotly&logoColor=white) ![Joblib](https://img.shields.io/badge/Joblib-Model_Save-green)


## 📖 Overview
The **Inventory Demand Forecasting System** is a **Streamlit web application** designed for **RetailMart Inc.** to predict optimal stock levels.  
It leverages **Gradient Boosting Regression** as the best-performing model, achieving:

- **R² Score:** 0.8840  
- **RMSE:** 5.61  
- **MAE:** 1.89  

This system helps retailers:
- Minimize **stockouts** by anticipating demand  
- Reduce **excess inventory costs**  
- Improve **inventory turnover** through accurate forecasting  

---

## 🚀 Key Features
- **Interactive Prediction**: Input product category, store type, region, season, promotions, lead time, and stock levels to generate demand forecasts.  
- **Actionable Recommendations**: Restock alerts, overstock warnings, and optimal stock insights.  
- **Historical Analysis**: Visualize sales trends, category performance, and regional distribution with Plotly charts.  
- **Model Performance Dashboard**: Compare regression models, view feature importance, and track deployed model metrics.  
- **Business Impact**: Designed to reduce costs, improve efficiency, and enhance decision-making.  

---

## 📂 Project Structure










├── app.py                        # Streamlit application
├── best_model.pkl                # Trained Gradient Boosting model
├── scaler.pkl                    # Scaler object
├── label_encoders.pkl            # Label encoders for categorical features
├── feature_columns.txt           # List of selected features
├── model_metrics.pkl             # Saved performance metrics
├── engineered_inventory_data.csv # Cleaned dataset
└── README.md                     # Project documentation













📊 Model Pipeline
Data Cleaning → Feature Engineering → Model Training → Hyperparameter Tuning → Deployment











🔑 Features Used
Temporal features (day, week, month, season, holidays)

Lag features (historical sales patterns: 1, 7, 14, 30 days)

Rolling statistics (moving averages, standard deviations)

Categorical features (product category, store type, region)

Promotional indicators (sale events, campaigns)

Supplier metrics (lead time, reliability scores)











📈 Model Performance
Model	R²	RMSE	MAE
Linear Regression	0.72	52.3	41.2
Ridge Regression	0.76	48.7	38.5
Decision Tree	0.81	45.2	35.8
Random Forest	0.87	38.4	29.6
Gradient Boosting (Best)	0.8840	5.61	1.89








🎯 Business Objectives
Reduce stockouts by at least 30%

Cut excess inventory costs by 20%

Boost inventory turnover with accurate demand prediction








🛠️ Technologies Used
Python (Pandas, NumPy, Scikit‑Learn)

Streamlit (interactive web app)

Plotly (data visualization)

Joblib (model persistence)











Developed as part of the Data Science & Machine Learning Regression Assignment.
© 2024 RetailMart Inc. | Built with ❤️ using Streamlit & Python.






This project is for educational and demonstration purposes.
All rights reserved © 2024 RetailMart Inc.
