import requests
import pandas as pd

import numpy as np

def fetch_pilot_zone_weather(lat=12.50, lon=76.90, start="20210101", end="20251231"):
    """Fetches daily meteorological inputs from NASA POWER API for Pilot Study Zone A."""
    url = (
        f"https://power.larc.nasa.gov/api/temporal/daily/point?"
        f"parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,ALLSKY_SFC_SW_DWN,PRECTOTCORR"
        f"&community=AG&longitude={lon}&latitude={lat}&start={start}&end={end}&format=JSON"
    )
    print("[Member 1] Fetching NASA POWER satellite data for Pilot Study Zone A...")
    response = requests.get(url, timeout=30).json()
    records = response["properties"]["parameter"]

    df = pd.DataFrame(records)
    df.index.name = "date"
    df = df.replace(-999.0, np.nan).dropna()

    # Add synthetic regional root-zone soil moisture context (NASA-USDA SMAP proxy in %)
    np.random.seed(42)
    df["SMAP_SOIL_MOISTURE_PCT"] = np.clip(
        42.0 + 0.6 * df["PRECTOTCORR"] - 0.35 * (df["T2M_MAX"] - 28.0) + np.random.normal(0, 3, len(df)),
        18.0, 90.0
    ).round(1)

    df.to_csv("pilot_zone_weather.csv")
    print(f"[Member 1] Saved 'pilot_zone_weather.csv' with {len(df)} daily records.")
    return df

if __name__ == "__main__": fetch_pilot_zone_weather()