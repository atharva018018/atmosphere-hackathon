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

# ----------------------------
# 👥 Team Credits (Clean + Professional)
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding-top: 10px;'>
        <div style='font-size:18px; font-weight:600; color:#1f4e79;'>
            💡 Developed as a submission to <u>Bharatiya Antariksh Hackathon 2025</u>
        </div><br>
        
        <div style='font-size:16px; font-weight:600; color:#2b7a78;'>
            Team: <span style='color:#007F5F; font-weight:bold;'>AtmoSphere</span>
        </div><br>
        
        <div style='font-size:15px;'>
            <span style='color:#444;'><b style='color:#d62828;'>Atharva</b> & <b style='color:#d62828;'>Kamakshee</b></span><br>
            <span style='color:#444;'><b style='color:#003049;'>Gaurav</b> & <b style='color:#003049;'>Aditi</b></span><br><br>
            <i style='color:#555;'>Undergraduate Students, Banaras Hindu University (BHU)</i>
        </div><br>
        
        <div style='font-size:13px; color:#6c757d;'>
            🔍 Powered by Streamlit • AI Models • Data Visualization • ML Integration
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

