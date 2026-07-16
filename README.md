# 📈 Retail Demand Forecasting System

An end-to-end Machine Learning project that predicts future product demand using historical sales patterns, pricing strategies, inventory levels, promotional activities, and seasonal trends.

The system helps retailers make data-driven decisions regarding inventory management, stock planning, promotions, and demand forecasting.

---

# 🌐 Live Demo

🚀 **Try the application here**

https://retail-demand-forecasting-system-31a1.onrender.com


## 🚀 Project Overview

Retail businesses often face challenges such as:

- Overstocking products
- Stock shortages
- Inefficient inventory planning
- Poor promotional strategies
- Demand fluctuations

This project leverages Machine Learning to forecast product demand and provide actionable insights for better business decision-making.

---

## 🎯 Objectives

- Predict future product demand (Units Sold)
- Analyze the impact of discounts and promotions
- Understand inventory requirements
- Capture seasonality and demand trends
- Build an interactive dashboard for real-time predictions

---

## 📊 Dataset Features

The model uses the following engineered and selected features:

### Historical Demand

- Units Sold Lag 1 Day
- Rolling Average Sales (7 Days)

### Pricing Features

- Discount Percentage
- Effective Price

### Inventory Features

- Inventory Level

### Product Features

- Product Frequency

### Category Features

- Electronics
- Grocery
- Home
- Personal Care
- Clothing (Base Category)

### Calendar Features

- Saturday Indicator
- Sunday Indicator
- Month Sin Encoding
- Month Cos Encoding

### Seasonal Features

- Festival Window Indicator

---

## ⚙️ Machine Learning Pipeline

### Data Preprocessing

- Missing Value Handling
- Date Processing
- One-Hot Encoding
- Feature Engineering

### Feature Engineering

- Lag Features
- Rolling Mean Features
- Effective Price Calculation
- Product Frequency Encoding
- Cyclical Month Encoding

### Feature Selection

Features were selected using:

- Correlation Analysis
- Business Understanding
- Demand Forecasting Logic
- Redundancy Removal

### Scaling

- StandardScaler

### Models Evaluated

- Linear Regression
- Random Forest Regressor

### Best Model

✅ Linear Regression

Performance:

| Metric | Score |
|----------|----------|
| R² Score | 0.999 |
| MAE | 0.13 |

---

## 🖥️ Streamlit Dashboard Features

### Prediction Dashboard

- Demand Prediction
- KPI Cards
- Inventory Status
- Promotion Impact

### Visual Analytics

- Demand Gauge Chart
- Feature Analysis

### Model Insights

- Model Information
- Performance Metrics
- Forecasting Benefits

---

## 📂 Project Structure

```text
Retail-Demand-Forecasting-System/
│
├── Data/
│   └── retail_store_sales_promotions_demand.csv
│
├── Models/
│   ├── Linear_Regression.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
└── Retail_Demand_Forecasting.ipynb
```

---

## 🛠️ Technologies Used

### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Machine Learning

- Scikit-Learn

### Visualization

- Plotly
- Matplotlib
- Seaborn

### Model Persistence

- Joblib

### UI 
- streamlit

### Deployment

- Render

---

## 📈 Business Benefits

### Inventory Optimization

Predict demand before stock shortages occur.

### Promotion Planning

Understand the impact of discounts and promotional campaigns.

### Demand Forecasting

Forecast future sales trends using historical patterns.

### Better Decision Making

Enable data-driven retail planning.

---

## 👨‍💻 Author

Shreya Patel

Machine Learning | Data Science | AI Enthusiast

---

