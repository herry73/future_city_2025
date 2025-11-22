import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from plots import ts_plot, corr_heatmap
from features import analyze_sensor_relationships

@st.cache_data
def load_data():
    """
    Try to load real cleaned data; if it doesn't exist yet,
    fall back to the dummy data so the app still works.
    """
    try:
        df = pd.read_csv("data/processed/clean_data.csv", parse_dates=["timestamp"])
    except FileNotFoundError:
        df = pd.read_csv("data/processed/clean_data_dummy.csv", parse_dates=["timestamp"])

    df = df.set_index("timestamp").sort_index()
    return df


def date_range_selector(df):
    """Widget to pick a date range and return the filtered dataframe."""
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


def air_quality_page(df):
    st.header("Air Quality & Traffic Alerts")

    df_range = date_range_selector(df)

    sensor = st.selectbox("Choose pollutant", ["NO2", "O3", "PM10"])

    # Summary KPIs
    col1, col2, col3 = st.columns(3)
    col1.metric("Average", f"{df_range[sensor].mean():.1f}")
    col2.metric("Max", f"{df_range[sensor].max():.1f}")
    col3.metric("Min", f"{df_range[sensor].min():.1f}")

    fig = ts_plot(df_range, sensor, f"{sensor} over time")
    st.pyplot(fig)


def heatwave_page(df):
    st.header("Heatwave Early Warning")

    df_range = date_range_selector(df)
    sensor = "temperature"

    col1, col2, col3 = st.columns(3)
    col1.metric("Average °C", f"{df_range[sensor].mean():.1f}")
    col2.metric("Max °C", f"{df_range[sensor].max():.1f}")
    col3.metric("Min °C", f"{df_range[sensor].min():.1f}")

    fig = ts_plot(df_range, sensor, "Temperature over time")
    st.pyplot(fig)


def noise_page(df):
    st.header("Event & Noise Monitoring")

    df_range = date_range_selector(df)
    sensor = "noise"

    col1, col2, col3 = st.columns(3)
    col1.metric("Average dB", f"{df_range[sensor].mean():.1f}")
    col2.metric("Max dB", f"{df_range[sensor].max():.1f}")
    col3.metric("Min dB", f"{df_range[sensor].min():.1f}")

    fig = ts_plot(df_range, sensor, "Noise level over time")
    st.pyplot(fig)


def insights_page(df):
    st.header("Overall Insights")

    possible = ["NO2", "O3", "PM10", "temperature", "humidity", "noise"]
    cols = [c for c in possible if c in df.columns]

    if len(cols) < 2:
        st.info("Not enough variables yet to show a correlation heatmap.")
        return

    st.markdown("Correlation between pollutants and weather variables.")
    fig = corr_heatmap(df, cols, "Correlation between sensors")
    st.pyplot(fig)

    # 🔍 Extra: relationships NO2↔traffic (noise), O3↔sunlight (temperature)
    stats = analyze_sensor_relationships(df)

    st.subheader("Sensor relationships")

    col1, col2 = st.columns(2)

    col1.metric(
        "NO₂ vs Traffic (noise) correlation",
        f"{stats['corr_no2_traffic']:.2f}",
    )
    col1.metric(
        "NO₂ high traffic / low traffic",
        f"{stats['no2_traffic_factor']:.2f}×",
    )

    col2.metric(
        "O₃ vs Sunlight (temperature) correlation",
        f"{stats['corr_o3_sunlight']:.2f}",
    )
    col2.metric(
        "O₃ high sun / low sun",
        f"{stats['o3_sunlight_factor']:.2f}×",
    )



def main():
    st.set_page_config(page_title="Future City Dashboard", layout="wide")

    df = load_data()

    st.sidebar.title("Smart City Use Cases")
    page = st.sidebar.radio(
        "Select view",
        ("Air Quality", "Heatwave", "Noise & Events", "Insights")
    )

    if page == "Air Quality":
        air_quality_page(df)
    elif page == "Heatwave":
        heatwave_page(df)
    elif page == "Noise & Events":
        noise_page(df)
    elif page == "Insights":
        insights_page(df)


if __name__ == "__main__":
    main()
    