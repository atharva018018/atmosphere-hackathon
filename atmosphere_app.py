# AtmoSphere: Satellite-Based Air Pollution Monitoring App (Hackathon Project)

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="AtmoSphere - Air Quality Dashboard", layout="wide")
st.title("🌍 AtmoSphere: AI-based Air Pollution Monitoring Dashboard")

# Sidebar for pollutant selection
st.sidebar.header("📊 Filter")
pollutant = st.sidebar.selectbox("Select Pollutant", ["PM2.5", "NO₂", "SO₂"])

# Generate dummy data
np.random.seed(42)
data = pd.DataFrame({
    "Latitude": np.random.uniform(25.0, 28.0, 200),
    "Longitude": np.random.uniform(75.0, 85.0, 200),
    "Temperature": np.random.uniform(15, 40, 200),
    "Humidity": np.random.uniform(20, 90, 200),
    "AOD": np.random.uniform(0.1, 1.5, 200),
    "NO2_column": np.random.uniform(5, 80, 200),
    "SO2_column": np.random.uniform(1, 50, 200),
})

# Generate synthetic pollutant labels
data["PM2.5"] = data["AOD"] * 40 + np.random.normal(0, 5, 200)
data["NO₂"] = data["NO2_column"] + np.random.normal(0, 3, 200)
data["SO₂"] = data["SO2_column"] + np.random.normal(0, 2, 200)

# Train ML model
features = data[["Temperature", "Humidity", "AOD", "NO2_column", "SO2_column"]]
targets = data[["PM2.5", "NO₂", "SO₂"]]
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(features, targets)
data[["PM2.5_pred", "NO₂_pred", "SO₂_pred"]] = model.predict(features)

# Display map with predictions
st.subheader(f"🗺️ Predicted {pollutant} Levels")
m = folium.Map(location=[26.5, 80.5], zoom_start=6)

for _, row in data.iterrows():
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        popup=f"{pollutant}: {round(row[pollutant + '_pred'], 2)}",
        color="blue",
        fill=True,
        fill_color="cyan",
        fill_opacity=0.6
    ).add_to(m)

folium_static(m, width=1200, height=600)

# Show data table
st.subheader("📋 Sample Prediction Table")
st.dataframe(data[[pollutant + "_pred", "Temperature", "Humidity"]].head(10))
# ----------------------------
# 👥 Team Credits (Grouped & Professional Footer)
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size:16px; color: #444;'>
        <b>💡 Developed as a submission to Bharatiya Antariksh Hackathon 2025</b><br><br>
        
        <span style='font-size:15px;'>Team: <b style='color:#2E8B57;'>AtmoSphere</b></span><br><br>
        
        <span style='font-size:14px;'>
            <b>Atharva</b> & <b>Kamakshee</b><br>
            <b>Gaurav</b> & <b>Aditi</b><br>
            <i>Undergraduate Students, Banaras Hindu University (BHU)</i>
        </span><br><br>
        
        <span style='font-size:13px; color:#888;'>Streamlit • AI-Driven • Data Visualization • ML Powered</span>
    </div>
    """,
    unsafe_allow_html=True
)
