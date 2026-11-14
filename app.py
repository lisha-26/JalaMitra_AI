import os
import streamlit as st
import joblib
from member1_data import fetch_pilot_zone_weather
from member2_ml_fao56 import train_and_save_model, calculate_irrigation_recommendation
from member3_validation import generate_sensitivity_table, get_nhp_validation_summary

st.set_page_config(page_title="JalaMitra AI", layout="wide")

# Auto-build dataset and ML model if missing or corrupted (fixes EOFError permanently!)
if not os.path.exists("pilot_zone_weather.csv") or os.path.getsize("pilot_zone_weather.csv") == 0:
    fetch_pilot_zone_weather()

try:
    artifact = joblib.load("jalamitra_model.pkl")
except Exception:
    train_and_save_model()
    artifact = joblib.load("jalamitra_model.pkl")

model = artifact["model"]
# =====================================================================
# 1. FARMER ENTERING DETAILS
# =====================================================================
st.header("1. Farmer Input Interface (Pilot Study Zone A)")
c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Crop & Field Details")
    plot_id = st.text_input("Farmer / Plot Identifier", "Pilot Study Zone A — Plot #104")
    acres = st.number_input("Field Area (Acres)", 0.5, 20.0, 2.0, 0.5)
    crop_age = st.slider("Crop Age (Days After Transplanting)", 1, 120, 45)
    pump_lph = st.selectbox("Pump Flow Rate (Liters/Hour)", [10000, 12000, 15000, 20000], index=1)

with c2:
    st.subheader("Daily Weather (NASA POWER API)")
    tmax = st.slider("Max Temp (°C)", 24.0, 42.0, 33.5, 0.5)
    tmin = st.slider("Min Temp (°C)", 15.0, 28.0, 21.0, 0.5)
    rh = st.slider("Relative Humidity (%)", 25.0, 95.0, 58.0, 1.0)
    ws = st.slider("Wind Speed (m/s)", 0.5, 6.0, 2.3, 0.1)
    solar = st.slider("Solar Radiation (MJ/m²/day)", 8.0, 28.0, 21.5, 0.5)

with c3:
    st.subheader("Soil Moisture & Rainfall Context")
    rain_mm = st.slider("Forecast Rainfall (mm)", 0.0, 40.0, 0.0, 0.5)
    soil_moist = st.slider("Root-Zone Soil Moisture (%) [NASA-USDA]", 18, 90, 42)
    soil_type = st.selectbox("Soil Texture", ["Red Sandy Loam (2.5 mm/day loss)", "Clay Loam (1.5 mm/day loss)"])
    perc_mm = 2.5 if "2.5" in soil_type else 1.5

res = calculate_irrigation_recommendation(
    model, tmax, tmin, rh, ws, solar, crop_age, rain_mm, soil_moist, perc_mm, acres, pump_lph
)

st.divider()

# =====================================================================
# 2. EXPLAIN WHY THE APPLICATION REACHES ITS RESULT
# =====================================================================
st.header("2. Why JalaMitra Reaches This Result (Transparent Physics + ML)")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Step 1: ML Reference ETo", f"{res['eto_mm']} mm/day", "Random Forest Forecast")
k2.metric("Step 2: Crop Demand (ETc)", f"{res['etc_mm']} mm/day", f"Kc = {res['kc']} ({res['stage']})")
k3.metric("Step 3: Rain & Soil Credit", f"-{res['eff_rain_mm'] + res['soil_credit_mm']:.2f} mm", f"Percolation +{res['perc_mm']} mm")
k4.metric("Step 4: Net Irrigation Output", f"{res['net_mm']} mm/day", f"{res['pump_hours']} hrs Pump Runtime")

st.info(
    f"**Decision Trail for {plot_id}:** Based on NASA POWER weather inputs, the Random Forest model predicts "
    f"reference evapotranspiration (**ETo**) of **{res['eto_mm']} mm/day**. Because the paddy crop is at **{crop_age} DAT** "
    f"(**{res['stage']}**, **Kc = {res['kc']}**), crop water demand (**ETc**) is **{res['etc_mm']} mm/day**. "
    f"After adding **{res['perc_mm']} mm** deep percolation and subtracting **{res['eff_rain_mm']} mm** effective rain "
    f"and **{res['soil_credit_mm']} mm** soil moisture credit, the net requirement is **{res['net_mm']} mm** "
    f"(**{res['liters']:,} Liters** = **{res['pump_hours']} hours** of pumping)."
)

# =====================================================================
# 3. DEMONSTRATE HOW THE RESULT CHANGES WITH DIFFERENT INPUTS
# =====================================================================
st.header("3. Sensitivity Analysis: How Results Change With Different Inputs")
sens_df = generate_sensitivity_table(model, tmin, rh, ws, solar, perc_mm, acres, pump_lph)
st.table(sens_df)

# =====================================================================
# 4. VALIDATION RESULTS & CITATIONS
# =====================================================================
st.header("4. Scientific Validation & Regional Benchmarking")
val = get_nhp_validation_summary(artifact)
v1, v2, v3 = st.columns(3)
v1.metric("ETo Prediction RMSE", f"{val['rmse_mm']} mm/day", "5-Year NASA POWER Test Set")
v2.metric("R² Score vs FAO-56 Standard", f"{val['r2_score']}", "High Physical Fidelity")
v3.metric("Seasonal Water Savings", f"{val['water_saved_pct']}%", f"{val['jalamitra_seasonal_mm']} mm vs {val['traditional_flood_mm']} mm Flood")

st.markdown("""
**Core Scientific Citations in Project Report:**
* **FAO Irrigation and Drainage Paper No. 56:** Crop evapotranspiration and crop-water requirement physics.
* **NASA POWER Daily Weather API:** Historical agro-meteorological inputs for Pilot Study Zone A.
* **NASA-USDA Enhanced SMAP Soil Moisture Data:** Regional root-zone soil-moisture context.
* **National Hydrology Project (NHP):** Indian hydrology and regional irrigation decision-support benchmarks.
""")