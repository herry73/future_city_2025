import streamlit as st
import pandas as pd
import requests
from datetime import datetime

#Visual Crossing API key
API_KEY = "5ZPZXNSBTGBXJ67SZS3MDHWRR"


def get_weather(lat, lon, date):
    """
    Fetches Visual Crossing weather data for a given date.
    Extracts only the 12:00 (noon) hourly record.
    """

    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/{date}?unitGroup=metric&include=hours&key={API_KEY}&contentType=json"
    )

    response = requests.get(url)
    data = response.json()

    # Validate API response
    if "days" not in data:
        return None

    hourly = data["days"][0]["hours"]

    # Find the hour 12:00
    record = None
    for h in hourly:
        if h["datetime"].startswith("12:"):
            record = h
            break

    if record is None:
        return None

    # Return data in the same format as your WeatherData.csv
    return {
        "timestamp": f"{date} 12:00",
        "latitude": lat,
        "longitude": lon,
        "temperature_degC": record.get("temp"),
        "humidity_percent": record.get("humidity"),
        "wind_speed_ms": record.get("windspeed"),
        "rain_mm": record.get("precip"),
        "solar_radiation_wm2": record.get("solarradiation"),
        "is_anomaly": 0
    }


# -------------------------------
# STREAMLIT UI
# -------------------------------

st.title("Weather Data Downloader (12:00 daily) — Visual Crossing API")

st.write("Enter coordinates and date range:")

lat = st.number_input("Latitude", value=49.1738)
lon = st.number_input("Longitude", value=9.2120)

start_date = st.date_input("Start date")
end_date = st.date_input("End date")

if st.button("Download Weather Data"):
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    rows = []
    progress = st.progress(0)

    for i, d in enumerate(dates):
        row = get_weather(lat, lon, d.strftime("%Y-%m-%d"))
        if row:
            rows.append(row)

        progress.progress((i + 1) / len(dates))

    df = pd.DataFrame(rows)

    st.subheader("Weather Data (12:00)")
    st.dataframe(df)

    st.download_button(
        label="Download CSV",
        data=df.to_csv(index=False),
        file_name="WeatherData.csv",
        mime="text/csv"
    )
