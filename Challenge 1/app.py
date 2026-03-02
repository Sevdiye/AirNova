# 🌬️ AirNova — Sofia Air Quality Dashboard
# Real-time PM₂.₅ monitoring + local-time-based short-term forecasting

import os
import math
import numpy as np
import pandas as pd
import requests
import joblib
import pytz
from datetime import datetime
import streamlit as st
import altair as alt


# CONFIGURATION

CITY = "Sofia"
LAT, LON = 42.6977, 23.3219
TIMEZONE = "Europe/Sofia"
BASE_PATH = r"C:\AI\Individual\Challenge 1"

MODEL_FILES = {
    "3h": os.path.join(BASE_PATH, "model_pm25_3h_lightgbm.pkl"),
    "6h": os.path.join(BASE_PATH, "model_pm25_6h_lightgbm.pkl"),
}


# STREAMLIT SETUP

st.set_page_config(page_title="🌬️ AirNova – Sofia Air Quality Dashboard", layout="wide")
st.title("AirNova — Sofia Air Quality Dashboard")
st.caption("Real-time PM₂.₅ monitoring and short-term forecasts for Sofia 🇧🇬")

# Explain PM2.5
st.markdown("""
### 🌫️ What is PM₂.₅?
PM₂.₅ refers to **fine particulate matter** smaller than 2.5 micrometers — about 30 times thinner than a human hair.  
These particles come from **vehicles, heating, and industry** and can **penetrate deep into the lungs**,  
affecting respiratory and cardiovascular health.

| PM₂.₅ (µg/m³) | Air Quality | Health Impact |
|---------------:|:------------|:---------------|
| 0–12 | 🟢 Good | Air quality is satisfactory and poses little or no risk. |
| 12.1–35.4 | 🟡 Moderate | Acceptable; sensitive groups may experience mild effects. |
| 35.5–55.4 | 🟠 Unhealthy (Sensitive) | Children & elderly should limit prolonged outdoor activity. |
| 55.5–150.4 | 🔴 Unhealthy | Everyone may begin to experience adverse health effects. |
| 150.5+ | 🟣 Very Unhealthy | Serious health effects for all populations. |
""")


# FETCH LIVE DATA (REAL-TIME)

@st.cache_data(ttl=900)
def fetch_live_data():
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={LAT}&longitude={LON}&hourly=pm2_5,pm10,carbon_monoxide,"
        f"nitrogen_dioxide,sulphur_dioxide,ozone,temperature_2m,relative_humidity_2m,"
        f"windspeed_10m,winddirection_10m,pressure_msl,precipitation&timezone=UTC"
    )
    r = requests.get(url)
    r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["hourly"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    for col in df.columns:
        if col != "time":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.bfill().ffill()
    return df

with st.spinner("Fetching real-time air-quality data..."):
    live_df = fetch_live_data()


# CURRENT LOCAL TIME

sofia_tz = pytz.timezone(TIMEZONE)
current_time = datetime.now(sofia_tz)
st.markdown(f"**Current local time in Sofia:** {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Convert all timestamps to local time
live_df["time"] = live_df["time"].dt.tz_convert(TIMEZONE)
live_df["data_type"] = np.where(live_df["time"] <= current_time, "Measured", "Forecast")


# LOAD MODELS

@st.cache_resource
def load_models(model_paths):
    models = {}
    errors = []
    for name, path in model_paths.items():
        if os.path.exists(path):
            try:
                models[name] = joblib.load(path)
            except Exception as e:
                errors.append(f"{name}: {e}")
        else:
            errors.append(f"{name}: File not found ({path})")
    return models, errors

models, model_errors = load_models(MODEL_FILES)
if model_errors:
    st.warning("⚠️ Some models failed to load:")
    for err in model_errors:
        st.text(err)

# FEATURE ENGINEERING

def add_features(df):
    df = df.copy()
    df["hour"] = df["time"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    for lag in [1, 2, 3, 6]:
        df[f"pm2_5_lag_{lag}h"] = df["pm2_5"].shift(lag)

    df["pm2_5_roll_mean_3h"] = df["pm2_5"].rolling(3, min_periods=1).mean()
    df["pm2_5_roll_std_3h"] = df["pm2_5"].rolling(3, min_periods=1).std()
    df["pm2_5_roll_std_6h"] = df["pm2_5"].rolling(6, min_periods=1).std()

    df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_temp"] = df["windspeed_10m"] * df["temperature_2m"]

    df = df.bfill().ffill().fillna(0)
    return df

feat_df = add_features(live_df)

def align_to_model(df, model):
    expected = getattr(model, "feature_name_", None)
    X = df.drop(columns=["time", "data_type"], errors="ignore").copy()
    if expected:
        for c in expected:
            if c not in X.columns:
                X[c] = 0
        X = X[expected]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    return X

def safe_predict(model, X):
    try:
        y = model.predict(X)
        return float(y[-1])
    except Exception:
        return float("nan")

# SAFE CURRENT VALUE (LOCAL)

latest_measured = live_df[live_df["time"] <= current_time].tail(1)
current_pm25 = float(latest_measured["pm2_5"].iloc[0])

# MODEL PREDICTIONS (BASED ON LOCAL CONDITIONS)

predictions = {}
for key, model in models.items():
    X = align_to_model(feat_df, model)
    predictions[key] = safe_predict(model, X)

# DISPLAY CURRENT PM₂.₅

st.subheader("📊 Current Air Quality in Sofia")

def categorize_pm25(value):
    if value <= 12: return "🟢 Good"
    elif value <= 35.4: return "🟡 Moderate"
    elif value <= 55.4: return "🟠 Unhealthy (Sensitive)"
    elif value <= 150.4: return "🔴 Unhealthy"
    else: return "🟣 Very Unhealthy"

aqi_label = categorize_pm25(current_pm25)
st.metric("Current PM₂.₅", f"{current_pm25:.1f} µg/m³", aqi_label)

# FORECAST METRICS

st.subheader("Forecasts (Local Time Based)")
col1, col2 = st.columns(2)
col1.metric("+3h Forecast", f"{predictions.get('3h', np.nan):.1f} µg/m³")
col2.metric("+6h Forecast", f"{predictions.get('6h', np.nan):.1f} µg/m³")


# TREND CHART (LAST 24 HOURS)

st.subheader("📈 PM₂.₅ Trend (Last 24 Hours + Forecast)")
start_time = current_time - pd.Timedelta(hours=12)
end_time = current_time + pd.Timedelta(hours=12)
chart_df = live_df[(live_df["time"] >= start_time) & (live_df["time"] <= end_time)][["time", "pm2_5", "data_type"]]
chart = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("time:T", title="Time (Local)"),
        y=alt.Y("pm2_5:Q", title="PM₂.₅ (µg/m³)"),
        color=alt.Color("data_type:N", title="Data Type"),
        tooltip=["time:T", "pm2_5:Q", "data_type:N"]
    )
    .properties(height=300)
)
st.altair_chart(chart, use_container_width=True)

# HEALTH ADVISORY

st.subheader("❤️ Health Advisory")

if current_pm25 <= 12:
    st.success("Air quality is **Good** — safe for everyone.")
elif current_pm25 <= 35.4:
    st.info("Air quality is **Moderate** — acceptable, but sensitive individuals may feel mild effects.")
elif current_pm25 <= 55.4:
    st.warning("Air quality is **Unhealthy for Sensitive Groups** — consider limiting prolonged outdoor activity.")
else:
    st.error("Air quality is **Unhealthy** — everyone may experience adverse health effects.")

# FORECAST TABLE (3H BACK → 6H AHEAD)

st.subheader("Detailed PM₂.₅ Outlook (Local Time)")

start_time = current_time - pd.Timedelta(hours=3)
end_time = current_time + pd.Timedelta(hours=6)
forecast_window = live_df[(live_df["time"] >= start_time) & (live_df["time"] <= end_time)].copy()

forecast_window["Air Quality"] = forecast_window["pm2_5"].apply(categorize_pm25)
forecast_window["Local Time"] = forecast_window["time"].dt.strftime("%Y-%m-%d %H:%M")

forecast_table = forecast_window[["Local Time", "pm2_5", "Air Quality", "data_type"]]
forecast_table = forecast_table.rename(columns={"pm2_5": "PM₂.₅ (µg/m³)", "data_type": "Type"})

st.dataframe(forecast_table.reset_index(drop=True), use_container_width=True)

# OUTLOOK SUMMARY

st.subheader("Short-Term Outlook")

if not math.isnan(predictions.get("6h", np.nan)):
    if predictions["6h"] > current_pm25 + 12:
        st.info("Air quality is **expected to worsen** slightly over the next 6 hours.")
    elif predictions["6h"] < current_pm25 - 12:
        st.info("Air quality is **expected to improve** over the next 6 hours.")
    else:
        st.info("Air quality is **expected to remain stable**.")
else:
    st.warning("Forecast models unavailable — showing only measured data.")

