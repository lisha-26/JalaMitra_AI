import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

FEATURES = ["T2M_MAX", "T2M_MIN", "RH2M", "WS2M", "ALLSKY_SFC_SW_DWN"]

def compute_fao56_eto(df, elevation_m=650):
    """Computes FAO-56 Penman-Monteith reference evapotranspiration (mm/day)."""
    T = df["T2M"]
    Tmax = df["T2M_MAX"]
    Tmin = df["T2M_MIN"]
    RH = df["RH2M"]
    u2 = df["WS2M"]
    Rs = df["ALLSKY_SFC_SW_DWN"]

    P = 101.3 * (((293.0 - 0.0065 * elevation_m) / 293.0) ** 5.26)
    gamma = 0.000665 * P

    es = (0.6108 * np.exp((17.27 * Tmax) / (Tmax + 237.3)) +
          0.6108 * np.exp((17.27 * Tmin) / (Tmin + 237.3))) / 2.0
    ea = es * (RH / 100.0)
    delta = (4098.0 * (0.6108 * np.exp((17.27 * T) / (T + 237.3)))) / ((T + 237.3) ** 2)

    Rn = np.maximum(0.77 * Rs - 2.45, 1.0)
    num = 0.408 * delta * Rn + gamma * (900.0 / (T + 273.0)) * u2 * (es - ea)
    den = delta + gamma * (1.0 + 0.34 * u2)
    return np.maximum(num / den, 1.2)

def get_paddy_kc(crop_age_days):
    """Returns FAO-56 Crop Coefficient (Kc) and growth stage name for Kharif Paddy."""
    if crop_age_days <= 25:
        return 1.05, "Initial Stage"
    elif crop_age_days <= 50:
        return 1.12, "Development Stage"
    elif crop_age_days <= 90:
        return 1.20, "Mid-Season Stage"
    else:
        return 0.90, "Late Season Stage"

def calculate_irrigation_recommendation(model, tmax, tmin, rh, ws, solar, crop_age_days,
                                        rain_mm, soil_moisture_pct, perc_mm, acres, pump_lph):
    """Explainable AI + FAO-56 Water Balance calculation."""
    X_in = pd.DataFrame([[tmax, tmin, rh, ws, solar]], columns=FEATURES)
    eto_mm = float(model.predict(X_in)[0])
    kc, stage = get_paddy_kc(crop_age_days)
    etc_mm = eto_mm * kc

    eff_rain_mm = rain_mm * 0.80
    soil_credit_mm = max(0.0, (soil_moisture_pct - 40.0) * 0.12)
    net_irrigation_mm = max(0.0, etc_mm + perc_mm - eff_rain_mm - soil_credit_mm)

    # 1 mm over 1 acre = 4,046.86 Liters
    total_liters = net_irrigation_mm * acres * 4046.86
    pump_hours = total_liters / pump_lph

    return {
        "eto_mm": round(eto_mm, 2),
        "kc": kc,
        "stage": stage,
        "etc_mm": round(etc_mm, 2),
        "eff_rain_mm": round(eff_rain_mm, 2),
        "soil_credit_mm": round(soil_credit_mm, 2),
        "perc_mm": perc_mm,
        "net_mm": round(net_irrigation_mm, 2),
        "liters": int(total_liters),
        "pump_hours": round(pump_hours, 1)
    }

def train_and_save_model():
    df = pd.read_csv("pilot_zone_weather.csv")
    df["ETo_FAO56"] = compute_fao56_eto(df)

    X = df[FEATURES]
    y = df["ETo_FAO56"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    joblib.dump({"model": model, "rmse": rmse, "r2": r2, "features": FEATURES}, "jalamitra_model.pkl")
    print(f"[Member 2] Model trained! RMSE = {rmse:.3f} mm/day | R² = {r2:.4f}")

if __name__ == "__main__":
    train_and_save_model()