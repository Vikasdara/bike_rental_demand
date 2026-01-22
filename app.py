import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("bike_demand_xgb_model.pkl")

st.title("🚲 Bike Rental Demand Prediction (Full Feature Input)")
st.write("Enter all feature values used during model training")

st.sidebar.header("Time & Calendar Features")

yr = st.sidebar.number_input("Year (yr)", min_value=2011.0, max_value=2012.0, value=2012.0)
mnth = st.sidebar.number_input("Month (mnth)", min_value=1.0, max_value=12.0, value=7.0)
month = st.sidebar.number_input("Month (engineered)", min_value=1, max_value=12, value=7)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=15)
day_of_week = st.sidebar.number_input("Day of Week (0=Mon)", min_value=0, max_value=6, value=3)
weekday = st.sidebar.number_input("Weekday", min_value=0, max_value=6, value=3)
hr = st.sidebar.number_input("Hour", min_value=0, max_value=23, value=9)

st.sidebar.header("Work & Schedule")

holiday = st.sidebar.selectbox("Holiday", [0, 1])
workingday = st.sidebar.selectbox("Working Day", [0, 1])
is_weekend = st.sidebar.selectbox("Is Weekend", [0, 1])
is_peak_hour = st.sidebar.selectbox("Is Peak Hour", [0, 1])
rush_hour_type = st.sidebar.selectbox(
    "Rush Hour Type",
    options=[0, 1, 2],
    format_func=lambda x: "Non-Peak" if x == 0 else "Morning Peak" if x == 1 else "Evening Peak"
)
workday_peak_interaction = st.sidebar.selectbox("Workday × Peak Interaction", [0, 1])


st.sidebar.header("Weather Features (Manual Precision Input)")

temp = st.sidebar.number_input(
    "Temperature (normalized)",
    min_value=0.0,
    max_value=1.0,
    value=0.2879,
    step=0.0001,
    format="%.4f"
)

atemp = st.sidebar.number_input(
    "Feels Like Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.2879,
    step=0.0001,
    format="%.4f"
)

hum = st.sidebar.number_input(
    "Humidity",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.0001,
    format="%.4f"
)

windspeed = st.sidebar.number_input(
    "Windspeed",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.0001,
    format="%.4f"
)

weather_comfort_index = st.sidebar.number_input(
    "Weather Comfort Index",
    min_value=0.0,
    max_value=1.5,
    value=0.377,
    step=0.001,
    format="%.3f"
)

is_bad_weather = st.sidebar.selectbox("Bad Weather", [0, 1])


st.sidebar.header("Season (One-Hot Encoded)")

season_springer = st.sidebar.selectbox("Season: Springer", [0, 1])
season_summer = st.sidebar.selectbox("Season: Summer", [0, 1])
season_winter = st.sidebar.selectbox("Season: Winter", [0, 1])

season_group_Spring = st.sidebar.selectbox("Season Group: Spring", [0, 1])
season_group_Summer = st.sidebar.selectbox("Season Group: Summer", [0, 1])
season_group_Winter = st.sidebar.selectbox("Season Group: Winter", [0, 1])

st.sidebar.header("Weather Situation (One-Hot Encoded)")

weathersit_Mist = st.sidebar.selectbox("Weather: Mist", [0, 1])
weathersit_Light_Snow = st.sidebar.selectbox("Weather: Light Snow", [0, 1])
weathersit_Heavy_Rain = st.sidebar.selectbox("Weather: Heavy Rain", [0, 1])

# Create input dataframe (EXACT column names)
input_data = pd.DataFrame([{
    'yr': yr,
    'mnth': mnth,
    'hr': hr,
    'holiday': holiday,
    'weekday': weekday,
    'workingday': workingday,
    'temp': temp,
    'atemp': atemp,
    'hum': hum,
    'windspeed': windspeed,
    'year': int(yr),
    'month': month,
    'day': day,
    'day_of_week': day_of_week,
    'is_weekend': is_weekend,
    'is_peak_hour': is_peak_hour,
    'rush_hour_type': rush_hour_type,
    'weather_comfort_index': weather_comfort_index,
    'is_bad_weather': is_bad_weather,
    'workday_peak_interaction': workday_peak_interaction,

    'season_springer': season_springer,
    'season_summer': season_summer,
    'season_winter': season_winter,

    'season_group_Spring': season_group_Spring,
    'season_group_Summer': season_group_Summer,
    'season_group_Winter': season_group_Winter,

    'weathersit_Mist': weathersit_Mist,
    'weathersit_Light Snow': weathersit_Light_Snow,
    'weathersit_Heavy Rain': weathersit_Heavy_Rain
}])

st.write("### Input Data Preview")
st.dataframe(input_data)

if st.button("Predict Bike Rental Demand"):
    try:
        prediction = model.predict(input_data)
        st.success(f"🚲 Predicted Bike Rental Demand: {int(prediction[0])}")
    except Exception as e:
        st.error("Prediction failed. Please check input values.")
        st.error(e)
