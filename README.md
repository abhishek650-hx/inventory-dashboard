# 📦 Inventory Optimization & Demand Forecasting System

## 🚀 Overview

This project is a **data-driven Inventory Optimization Dashboard** built using Python, Streamlit, and PostgreSQL. It helps businesses make smarter inventory decisions by combining:

* 📊 Inventory optimization (EOQ, ROP, Safety Stock)
* 📈 Demand forecasting (Prophet)
* ⚡ Real-time data updates
* 📉 Interactive business intelligence dashboard

---

## 🎯 Objective

To build a **scalable and intelligent inventory management system** that:

* Minimizes stockouts and overstock
* Optimizes order quantity and timing
* Provides real-time insights
* Uses forecasting to improve decision-making

---

## 🏗️ System Architecture

```
CSV Upload / Database
        ↓
PostgreSQL (Supabase)
        ↓
Streamlit Application
        ↓
Data Processing + Optimization
        ↓
Dashboard + Forecasting
```

---

## 📊 Features

### 🔹 1. Inventory Optimization

* **EOQ (Economic Order Quantity)**
  Minimizes ordering + holding cost
* **ROP (Reorder Point)**
  Determines when to reorder
* **Safety Stock**
  Handles demand uncertainty
* **Smart Recommendations**
  → “Order X units” or “Stock OK”

---

### 🔹 2. Demand Forecasting

* Built using **Prophet**
* Simulates realistic time-series data
* Includes:

  * Trend
  * Seasonality
  * Noise & irregular spikes
* Forecasts next **7 days demand**

---

### 🔹 3. Interactive Dashboard

#### 📊 KPI Metrics

* Total Products
* Reorders Needed
* Critical Stock Items

#### 📈 Visualizations

* Top Demand Products (highlighting critical items)
* Stock vs ROP comparison
* Demand Forecast (time-series)
* Category Demand Share (Pie chart)
* Demand Distribution (Categorized Histogram)
* Category Performance Heatmap

---

### 🔹 4. Data Variability Engine

* Realistic demand simulation using log-normal distribution
* Category-based scaling
* Product-level randomness

---

### 🔹 5. UI/UX Enhancements

* Horizontal charts for readability
* Semantic color logic:

  * 🔵 Normal items
  * 🔴 Critical items
* Clean layout with KPI hierarchy
* Interactive filters (category-based)

---

## ⚙️ Tech Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Frontend        | Streamlit             |
| Backend         | Python                |
| Database        | PostgreSQL (Supabase) |
| Visualization   | Plotly                |
| Forecasting     | Prophet               |
| Data Processing | Pandas, NumPy         |

---

## 📁 Project Structure

```
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── runtime.txt           # Python version
├── data/                 # Optional datasets
└── README.md             # Project documentation
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/inventory-system.git
cd inventory-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📌 Key Concepts Used

* Inventory Management Theory:

  * EOQ Model
  * Safety Stock
  * Reorder Point

* Data Science:

  * Probability distributions
  * Time-series simulation
  * Forecasting

* Business Intelligence:

  * KPI dashboards
  * Data visualization
  * Decision support systems

---

## ⚠️ Limitations

* Forecasting uses **simulated time-series data**
* No real-time API integration yet
* No authentication system
* Static demand generation (not historical)

---

## 🚀 Future Enhancements

* 🔮 Replace Prophet with ML models (XGBoost / LSTM)
* 📊 ABC Analysis (inventory classification)
* 🚨 Email/SMS alerts for low stock
* 🔗 Real-time API integration (Zepto/Blinkit)
* 👥 Multi-user authentication system
* ⚙️ Microservices architecture

---

## 🧠 One-Line Summary

> A cloud-based inventory optimization system that uses EOQ, ROP, and forecasting to provide real-time insights and smarter inventory decisions.

---


Give it a ⭐ on GitHub and share your feedback!
