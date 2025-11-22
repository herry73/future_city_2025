import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from features import (
    add_health_score,
    detect_air_risk_alerts,
    detect_night_noise_events,
    compute_tree_priority
)

def plot_health_score(df):
    fig = px.line(
        df,
        x=df.index,
        y="health_score",
        title="Air Quality Health Score",
        line_shape="spline",
        template="plotly_dark",
    )

    fig.update_layout(
        height=400,
        margin=dict(l=30, r=30, t=50, b=10),
        xaxis_title="",
        yaxis_title="Health Score",
        hovermode="x unified",
    )

    fig.update_traces(line=dict(width=3, color="#4CC9F0"))

    st.plotly_chart(fig, use_container_width=True)


# We will import these from dashboard.py
def smart_alerts_page(load_air_data, load_weather_data, load_noise_data):
    st.header("Smart City Controller – Alerts & Recommendations")

    # Load datasets
    air = load_air_data()
    weather = load_weather_data()
    noise = load_noise_data()

    # Remove duplicate columns from all datasets BEFORE merge
    for df_src in (air, noise, weather):
        df_src.drop(columns=["latitude", "longitude", "is_anomaly"], errors="ignore", inplace=True)

    # Merge datasets
    df = air.join(noise, how="outer").join(weather, how="outer")
    df = df.sort_index()

    # ----- Health Score -----
    try:
        df = add_health_score(df)
    except Exception as e:
        st.error(f"Health Score Error: {e}")
        return

    st.subheader("Air Quality Health Score")
    plot_health_score(df)


    # ---------------------------
    # 1. AIR RISK
    # ---------------------------
    st.subheader("🔥 Air Risk Alerts (Bad Air + High Noise)")

    df_alerts, episodes = detect_air_risk_alerts(
        df,
        health_threshold=40,
        noise_threshold=70,
        min_hours=3
    )

    if episodes.empty:
        st.success("No Air Risk episodes detected ✔")
    else:
        st.warning(f"⚠ {len(episodes)} Air Risk episodes detected")
        st.dataframe(episodes)

    fig, ax = plt.subplots(figsize=(10, 4))
    df_alerts["air_risk_alert"].astype(int).plot(ax=ax)
    ax.set_title("Air Risk Alerts (1 = alert)")
    st.pyplot(fig)

    # ---------------------------
    # 2. NIGHT NOISE
    # ---------------------------
    st.subheader("🌙 Night Noise Disturbances")

    df_noise_events, night_events = detect_night_noise_events(
        df,
        noise_threshold=60,
        min_duration_minutes=60
    )

    if night_events.empty:
        st.success("No night disturbances detected ✔")
    else:
        st.warning(f"⚠ {len(night_events)} night disturbances found")
        st.dataframe(night_events)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    df_noise_events["noise_disturbance"].astype(int).plot(ax=ax2)
    ax2.set_title("Night Noise Disturbances (1 = disturbance)")
    st.pyplot(fig2)

    # ---------------------------
    # 3. TREE PRIORITY
    # ---------------------------
    st.subheader("🌳 Tree Priority Score (Urban Green Planning)")

    chronic_air, chronic_noise, score = compute_tree_priority(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Chronic Bad Air", f"{chronic_air*100:.1f}%")
    c2.metric("Chronic Noise", f"{chronic_noise*100:.1f}%")
    c3.metric("Tree Priority Score", f"{score:.1f}")
