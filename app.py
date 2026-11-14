import os
import streamlit as st
import joblib
from member1_data import fetch_pilot_zone_weather
from member2_ml_fao56 import train_and_save_model, calculate_irrigation_recommendation
from member3_validation import generate_sensitivity_table, get_nhp_validation_summary

st.set_page_config(
    page_title="JalaMitra AI | Smart Irrigation",
    page_icon="🌾",
    layout="wide"
)

# --- MODERN NATURAL UI STYLING ---
st.markdown("""
<style>
    .main {
        background-color: #f4f7f6;
    }
    h1 {
        color: #1b5e20;
        font-weight: 700;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #2e7d32;
        font-weight: 600;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #4caf50;
    }
    .stAlert {
        border-radius: 10px;
        background-color: #e8f5e9;
        color: #1b5e20;
        border: 1px solid #c8e6c9;
    }
    div[data-testid="stTable"] {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)