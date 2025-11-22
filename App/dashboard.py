import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from plots import ts_plot, corr_heatmap, hourly_pattern_bar



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

    # st.subheader("Average noise by hour of day")
    # fig2 = hourly_pattern_bar(df_range, sensor, "Noise hourly pattern")
    # st.pyplot(fig2)



def insights_page():
    st.header("Cross-Dataset Insights")

    weather = load_weather_data()
    noise = load_noise_data()

    # Remove conflicting columns before joining
    noise = noise.drop(columns=["latitude", "longitude"], errors="ignore")
    weather = weather.drop(columns=["latitude", "longitude"], errors="ignore")

    df = weather.join(noise, how="inner")

    possible = ["temperature", "humidity", "noise"]
    cols = [c for c in possible if c in df.columns]

    if len(cols) < 2:
        st.info("More data needed for correlation matrix.")
        return

    fig = corr_heatmap(df, cols, "Correlation between weather & noise")
    st.pyplot(fig)




def main():
    st.set_page_config(page_title="Future City Dashboard", layout="wide")

    st.sidebar.title("Smart City Use Cases")
    page = st.sidebar.radio(
        "Select view",
        (
            "Heatwave",
            "Noise & Events",
            "Insights"
        )
    )

    if page == "Heatwave":
        heatwave_page()
    elif page == "Noise & Events":
        noise_page()
    elif page == "Insights":
        insights_page()


if __name__ == "__main__":
    main()
