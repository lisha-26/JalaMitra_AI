import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

st.set_page_config(
    page_title="JalaMitra AI | Minimalist Irrigation",
    page_icon="💧",
    layout="centered"
)

# --- SUBTLE MINIMALIST UI STYLING ---
st.markdown("""
<style>
    .main { background-color: #fafafa; }
    h1 { color: #111111; font-weight: 600; font-size: 28px; text-align: center; margin-bottom: 5px; }
    .subtitle { text-align: center; color: #666666; font-size: 14px; margin-bottom: 30px; }
    .result-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid #eaeaea;
        text-align: center;
        margin-top: 20px;
    }
    .metric-big {
        font-size: 36px;
        font-weight: 700;
        color: #2e7d32;
        margin: 10px 0;
    }
    .metric-sub {
        font-size: 16px;
        color: #555555;
    }
</style>
""", unsafe_allow_html=True)

FEATURES = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"]

def compute_fao56_eto(df, elevation_m=650):
    T = df["T2M"]
    Tmax = df["T2M_MAX"]
    Tmin = df["T2M_MIN"]
    RH = df["RH2M"]
    u2 = df["WS2M"]
    Rs = df["ALLSKY_SFC_SW_DWN"]
    P = 101.3 * (((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26)
    gamma = 0.000665 * P
    es = (0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3)) + 0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))) / 2.0
    ea = es * (RH / 100.0)
    delta = (4098.0 * (0.6108 * np.exp((17.27 * T) / (T + 237.3)))) / ((T + 237.3) ** 2)
    Rn = np.maximum(0.77 * Rs - 2.45, 1.0)
    num = 0.408 * delta * Rn + gamma * (900.0 / (T + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    return np.maximum(num / den, 1.2)

def get_paddy_kc(crop_age_days):
    if crop_age_days <= 25: return 1.05
    elif crop_age_days <= 50: return 1.12
    elif crop_age_days <= 90: return 1.20
    else: return 0.90

@st.cache_resource
def load_or_train_model():
    if os.path.exists("jalamitra_model.pkl"):
        try:
            art = joblib.load("jalamitra_model.pkl")
            return art["model"]
        except Exception:
            pass
    np.random.seed(42)
    size = 200
    df_sim = pd.DataFrame({
        "T2M_MAX": np.random.uniform(28, 38, size),
        "T2M_MIN": np.random.uniform(18, 24, size),
        "T2M": np.random.uniform(23, 31, size),
        "RH2M": np.random.uniform(40, 85, size),
        "WS2M": np.random.uniform(1.5, 4.5, size),
        "ALLSKY_SFC_SW_DWN": np.random.uniform(15, 26, size),
        "PRECTOTCORR": np.random.choice([0, 0, 5, 12], size)
    })
    df_sim["ETo_FAO56"] = compute_fao56_eto(df_sim)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(df_sim[FEATURES], df_sim["ETo_FAO56"])
    joblib.dump({"model": model}, "jalamitra_model.pkl")
    return model

model = load_or_train_model()

# --- MINIMALIST UI ---
st.markdown("<h1>JalaMitra AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Irrigation Calculator</div>", unsafe_allow_html=True)

with st.form("irrigation_form"):
    col1, col2 = st.columns(2)
    with col1:
        crop_age = st.slider("Crop Age (Days)", 1, 120, 45)
        acres = st.number_input("Field Area (Acres)", 0.5, 10.0, 2.0, 0.5)
        tmax = st.slider("Max Temp (°C)", 24.0, 42.0, 33.5, 0.5)
    with col2:
        rain_mm = st.slider("Rainfall Forecast (mm)", 0.0, 30.0, 0.0, 0.5)
        soil_moist = st.slider("Soil Moisture (%)", 20, 80, 42)
        pump_lph = st.selectbox("Pump Flow (Liters/Hour)", [10000, 12000, 15000, 20000], index=1)
    
    submitted = st.form_submit_button("Calculate Irrigation", use_container_width=True)

if submitted:
    # Calculation
    X_in = pd.DataFrame([[tmax, 21.0, 58.0, 2.3, 21.5]], columns=FEATURES)
    eto_mm = float(model.predict(X_in)[0])
    kc = get_paddy_kc(crop_age)
    etc_mm = eto_mm * kc
    eff_rain = rain_mm * 0.80
    soil_credit = max(0.0, (soil_moist - 40.0) * 0.12)
    net_mm = max(0.0, etc_mm + 2.5 - eff_rain - soil_credit)
    total_liters = net_mm * acres * 4046.86
    pump_hours = total_liters / pump_lph

    st.markdown(f"""
    <div class="result-card">
        <div class="metric-sub">Recommended Irrigation Requirement</div>
        <div class="metric-big">{int(total_liters):,} Liters</div>
        <div class="metric-sub">({pump_hours:.1f} Hours of Pump Runtime for {acres} Acres)</div>
    </div>
    """, unsafe_allow_html=True)