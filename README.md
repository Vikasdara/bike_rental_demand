# 🚲 Bike Rental Demand Prediction System

## 📌 Project Overview
This project predicts the **number of bike rentals** based on weather conditions and day-related factors using a **Machine Learning model (XGBoost Regressor)**.  
The trained model is deployed as an **interactive web application using Streamlit**.

The system helps bike rental companies **estimate demand in advance** and optimize resource planning.

---

## 🧠 Machine Learning Model
- **Algorithm:** XGBoost Regressor
- **Problem Type:** Regression
- **Target Variable:** `cnt` (Total bike rentals)
- **Model File:** `bike_demand_xgb_model.pkl`

---

## 📊 Input Features
The model uses the following features for prediction:

- Temperature
- Humidity
- Windspeed
- Season
- Holiday
- Working Day

---

## 🖥️ Web Application
- **Framework:** Streamlit
- **Frontend:** Interactive sliders & dropdowns
- **Backend:** Pickle-loaded trained ML model
- **Output:** Predicted bike rental count

---

## 🗂️ Project Structure
