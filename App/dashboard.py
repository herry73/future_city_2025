
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from smart_alerts import smart_alerts_page
from features import analyze_sensor_relationships
from plots import ts_plot, corr_heatmap, hourly_pattern_bar


@st.cache_data
def load_air_data():
    df = pd.read_csv("data/processed/air_quality.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


@st.cache_data
def load_weather_data():
    df = pd.read_csv("data/processed/weather.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    df = df.rename(columns={
        "temperature_degC": "temperature",
        "humidity_percent": "humidity"
    })

    return df




@st.cache_data
def load_noise_data():
    df = pd.read_csv("data/processed/noise.csv", parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    df = df.rename(columns={
        "noise_db": "noise"
    })

    return df



def date_range_selector(df):
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    start, end = st.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask]

def air_quality_page():
    st.header("Air Quality & Traffic Alerts")

    df = load_air_data()
    df_range = date_range_selector(df)

    pollutant_map = {
        "NO₂ (µg/m³)": "no2_ugm3",
        "PM10 (µg/m³)": "pm10_ugm3",
        "PM2.5 (µg/m³)": "pm2_5_ugm3",
        "O₃ (µg/m³)": "o3_ugm3",
    }

    label = st.selectbox("Choose pollutant", list(pollutant_map.keys()))
    col = pollutant_map[label]

    # Threshold slider (80th percentile default)
    default_thr = float(df_range[col].quantile(0.8))
    threshold = st.slider(
        f"{label} alert threshold",
        float(df_range[col].min()),
        float(df_range[col].max()),
        value=default_thr,
    )

    above = df_range[df_range[col] > threshold]


    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Average", f"{df_range[col].mean():.1f}")
    c2.metric("Max", f"{df_range[col].max():.1f}")
    c3.metric("Min", f"{df_range[col].min():.1f}")
    c4.metric("Days above threshold", above.shape[0])

    fig, ax = plt.subplots(figsize=(10, 4))
    df_range[col].plot(ax=ax, label="value")

    if "is_anomaly" in df_range.columns:
        anoms = df_range[df_range["is_anomaly"] == 1]
        if not anoms.empty:
            ax.scatter(anoms.index, anoms[col], marker="o")

    ax.axhline(threshold, linestyle="--")
    ax.set_title(f"{label} over time (dashed = alert threshold)")
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    fig.tight_layout()
    st.pyplot(fig)




def heatwave_page():
    st.header("Heatwave Early Warning")

    df = load_weather_data()
    df_range = date_range_selector(df)

    sensor = "temperature"

    col1, col2, col3 = st.columns(3)
    col1.metric("Average °C", f"{df_range[sensor].mean():.1f}")
    col2.metric("Max °C", f"{df_range[sensor].max():.1f}")
    col3.metric("Min °C", f"{df_range[sensor].min():.1f}")

    fig = ts_plot(df_range, sensor, "Temperature over time")
    st.pyplot(fig)




def noise_page():
    st.header("Event & Noise Monitoring")

    df = load_noise_data()
    df_range = date_range_selector(df)

    sensor = "noise"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg dB", f"{df_range[sensor].mean():.1f}")
    col2.metric("Max dB", f"{df_range[sensor].max():.1f}")
    col3.metric("Min dB", f"{df_range[sensor].min():.1f}")
    col4.metric("Anomalies", int(df_range["is_anomaly"].sum()))

    fig = ts_plot(df_range, sensor, "Noise level over time")
    st.pyplot(fig)




def insights_page():
    st.header("Overall Insights")

    # Load datasets
    air = load_air_data()
    weather = load_weather_data()
    noise = load_noise_data()

    # Rename pollutant columns (required by relationship analyzer)
    air = air.rename(columns={
        "no2_ugm3": "NO2",
        "o3_ugm3": "O3"
    })

    # ⚠️ FIX: remove duplicate columns before joining
    for df in (air, weather, noise):
        for col in ["latitude", "longitude", "is_anomaly"]:
            df.drop(columns=[col], errors="ignore", inplace=True)

    # Merge everything
    df = air.join(weather, how="outer").join(noise, how="outer")

    # Choose columns for heatmap
    possible = ["NO2", "O3", "pm10_ugm3", "temperature", "humidity", "noise"]
    cols = [c for c in possible if c in df.columns]

    if len(cols) < 2:
        st.info("Not enough variables yet to show a correlation heatmap.")
        return

    # Correlation matrix
    fig = corr_heatmap(df, cols, "Correlation between sensors")
    st.pyplot(fig)

    # --- Sensor relationships ---
    stats = analyze_sensor_relationships(df)

    st.subheader("Sensor relationships")

    col1, col2 = st.columns(2)

    col1.metric("NO₂ vs Traffic (noise) correlation", f"{stats['corr_no2_traffic']:.2f}")
    col1.metric("NO₂ high traffic / low traffic", f"{stats['no2_traffic_factor']:.2f}×")

    col2.metric("O₃ vs Sunlight (temperature) correlation", f"{stats['corr_o3_sunlight']:.2f}")
    col2.metric("O₃ high sun / low sun", f"{stats['o3_sunlight_factor']:.2f}×")





def main():
    st.set_page_config(page_title="Future City Dashboard", layout="wide")

    st.sidebar.title("Smart City Use Cases")
    page = st.sidebar.radio(
    "Select view",
    (
        "Air Quality",
        "Heatwave",
        "Noise & Events",
        "Insights",
        "Smart Alerts"
    )
)

    if page == "Air Quality":
        air_quality_page()
    elif page == "Heatwave":
        heatwave_page()
    elif page == "Noise & Events":
        noise_page()
    elif page == "Insights":
        insights_page()
    elif page == "Smart Alerts":
        smart_alerts_page(load_air_data, load_weather_data, load_noise_data)



if __name__ == "__main__":
    main()
