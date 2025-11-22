# =====================================================================
#  PART 1 — IMPORTS + THEMES + UTILITIES + UNIFIED DATA LOADER
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # kept in case you still use it somewhere
import pydeck as pdk
from datetime import datetime, date

import plotly.express as px
import plotly.graph_objects as go

# Import backend analysis functions
from features import (
    add_health_score,
    detect_air_risk_alerts,
    detect_night_noise_events,
    compute_tree_priority,
    analyze_sensor_relationships,
)

# ---------------------------------------------------------
#  GLOBAL STYLE PALETTES
# ---------------------------------------------------------

PASTEL = {
    "bg": "#F2F6FF",        # soft blue pastel, readable
    "text": "#1F2A44",      # deep navy for contrast
    "card": "#FFFFFF",      # white cards
    "accent": "#6BA8FF",    # soft blue accents
    "primary": "#4C84E0",   # stronger blue for graphs
}

DARK_NEON = {
    "bg": "#0A0F1F",
    "text": "#F5F7FA",
    "card": "#131A2B",
    "accent": "#FF3B8D",
    "primary": "#40C9FF",
    "danger": "#FF5577",
}

PLANNER = {
    "bg": "#E8F5E9",        # medium soft green
    "text": "#1B5E20",      # deep green for contrast
    "card": "#FFFFFF",      # clean white cards
    "accent": "#66BB6A",    # medium green accents
    "primary": "#388E3C",   # darker forest green
}

# ---------------------------------------------------------
#  ROLE-BASED THEMES (Inject CSS)
# ---------------------------------------------------------

def apply_resident_theme():
    st.markdown(f"""
    <style>

    /* ACTIVE ROLE BADGE — RESIDENT (Blue Pastel) */
    [data-testid="stSidebar"] code {{
        background-color: #4C84E0 !important;
        color: white !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }}

    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {{
        background-color: {PASTEL['bg']} !important;
        color: {PASTEL['text']} !important;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: #F7FAFF !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {PASTEL['text']} !important;
    }}

    /* HEADINGS */
    h1, h2, h3, h4, h5, h6 {{
        color: {PASTEL['text']} !important;
        font-weight: 700 !important;
    }}

    /* INPUTS */
    .stTextInput input, 
    .stDateInput input, 
    .stSelectbox div[role='combobox'] {{
        background-color: white !important;
        color: {PASTEL['text']} !important;
        border-radius: 8px;
        border: 1px solid #BFD7FF !important;
    }}

    /* TABLES */
    .stDataFrame table, .stDataFrame th, .stDataFrame td {{
        color: {PASTEL['text']} !important;
        background-color: white !important;
    }}

    /* BUTTONS */
    .stButton>button {{
        background-color: {PASTEL['accent']} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }}

    /* METRIC CARDS */
    [data-testid="stMetric"] {{
        background-color: white !important;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #D7E6FF;
    }}
    [data-testid="stMetric"] * {{
        color: {PASTEL['text']} !important;
    }}

    </style>
    """, unsafe_allow_html=True)



def apply_controller_theme():
    st.markdown(f"""
    <style>

    /* ACTIVE ROLE BADGE — CONTROLLER (Neon Pink) */
    [data-testid="stSidebar"] code {{
        background-color: #FF3B8D !important;
        color: white !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {DARK_NEON['bg']} !important;
        color: {DARK_NEON['text']} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: #111624 !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {DARK_NEON['text']} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {DARK_NEON['text']} !important;
    }}

    .stTextInput input, 
    .stDateInput input, 
    .stSelectbox div[role='combobox'] {{
        background-color: #1A2238 !important;
        color: {DARK_NEON['text']} !important;
        border-radius: 8px;
        border: 1px solid #33415C !important;
    }}

    .stButton>button {{
        background-color: {DARK_NEON['accent']} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetric"] {{
        background-color: #1A2238 !important;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #33415C;
    }}
    [data-testid="stMetric"] * {{
        color: {DARK_NEON['text']} !important;
    }}

    </style>
    """, unsafe_allow_html=True)


def apply_planner_theme():
    st.markdown(f"""
    <style>

    /* ACTIVE ROLE BADGE — PLANNER (Dark Green) */
    [data-testid="stSidebar"] code {{
        background-color: #1B5E20 !important;
        color: white !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {PLANNER['bg']} !important;
        color: {PLANNER['text']} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: #F1F8F4 !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {PLANNER['text']} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {PLANNER['text']} !important;
        font-weight: 700 !important;
    }}

    .stTextInput input, 
    .stDateInput input, 
    .stSelectbox div[role='combobox'] {{
        background-color: white !important;
        color: {PLANNER['text']} !important;
        border-radius: 8px;
        border: 1px solid #A5D6A7 !important;
    }}

    .stDataFrame table, .stDataFrame th, .stDataFrame td {{
        color: {PLANNER['text']} !important;
        background-color: white !important;
    }}

    .stButton>button {{
        background-color: {PLANNER['accent']} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }}

    [data-testid="stMetric"] {{
        background-color: white !important;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #C8E6C9;
    }}
    [data-testid="stMetric"] * {{
        color: {PLANNER['text']} !important;
    }}

    </style>
    """, unsafe_allow_html=True)



def fix_label_colors():
    st.markdown("""
    <style>
    label, .stDateInput label, .stTextInput label, .stSelectbox label {
        color: #1A1A1A !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
#  UTILITY — DROPDOWN DATE SELECTORS
# ---------------------------------------------------------

def select_date_range(df):
    """User-friendly date input widgets."""
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    col1, col2 = st.columns(2)

    start = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask]

# ---------------------------------------------------------
#  UTILITY — ADD THRESHOLD BANDS (used in some text)
# ---------------------------------------------------------

def draw_threshold(ax, threshold, label, color="red"):
    # kept for any remaining matplotlib use
    ax.axhline(threshold, linestyle="--", color=color, linewidth=1.5, label=label)
    ax.legend()

# ---------------------------------------------------------
#  UTILITY — HIGHLIGHT ANOMALIES ON PLOT (for matplotlib – no longer used)
# ---------------------------------------------------------

def highlight_anomalies(ax, df, column):
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        if len(anomalies):
            ax.scatter(anomalies.index, anomalies[column], color="red", s=30, label="Anomaly")

# ---------------------------------------------------------
#  SHARED COMPONENT — METRIC CARD
# ---------------------------------------------------------

def metric_card(title, value, subtitle="", color="#FFFFFF"):
    st.markdown(
        f"""
        <div style="
            background-color:{color};
            padding: 15px;
            border-radius: 12px;
            margin-bottom: 10px;
        ">
            <h3 style="margin:0;padding:0;">{title}</h3>
            <h2 style="margin:0;padding:0;">{value}</h2>
            <p style="margin:0;padding:0;font-size:13px;opacity:0.8;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
#  UTILITY — PYDECK MAP CONSTRUCTION
# ---------------------------------------------------------

def map_layer(df, lat_col="latitude", lon_col="longitude", value_col=None, color=[255, 0, 0]):
    """Generic map layer for any metric."""
    df_map = df[[lat_col, lon_col, value_col]].dropna() if value_col else df[[lat_col, lon_col]].dropna()

    layer = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        get_position=[lon_col, lat_col],
        get_radius=30,
        get_fill_color=color,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=df_map[lat_col].mean(),
        longitude=df_map[lon_col].mean(),
        zoom=11,
        pitch=40,
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state)

# ---------------------------------------------------------
#  UNIFIED DATA LOADER
# ---------------------------------------------------------

@st.cache_data
def load_data():
    """Loads and merges all CSVs from data/processed."""

    air = pd.read_csv("data/processed/air_quality.csv", parse_dates=["timestamp"])
    weather = pd.read_csv("data/processed/weather.csv", parse_dates=["timestamp"])
    noise = pd.read_csv("data/processed/noise.csv", parse_dates=["timestamp"])

    # Rename weather columns
    weather = weather.rename(columns={
        "temperature_degC": "temperature",
        "humidity_percent": "humidity"
    })

    # Rename noise column
    noise = noise.rename(columns={"noise_db": "noise"})

    df = (
        air.merge(weather, on="timestamp", how="outer", suffixes=("", "_wx"))
           .merge(noise, on="timestamp", how="outer", suffixes=("", "_nx"))
           .sort_values("timestamp")
    )

    df = df.loc[:, ~df.columns.duplicated()]
    df = df.set_index("timestamp")

    # Ensure numeric
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# =====================================================================
#  PART 2 — LANDING PAGE + ROLE SELECTION
# =====================================================================

def init_session_state():
    """Ensure required session_state keys exist."""
    if "role" not in st.session_state:
        st.session_state.role = None
    if "resident_page" not in st.session_state:
        st.session_state.resident_page = "Today"
    if "controller_page" not in st.session_state:
        st.session_state.controller_page = "Dashboard"
    if "planner_page" not in st.session_state:
        st.session_state.planner_page = "Overview"

def role_card(title, emoji, description, role_key, bg_color, text_color):
    """Clickable card used on landing page."""

    if st.button(f"{emoji}  {title}", key=f"role_btn_{role_key}", use_container_width=True):
        st.session_state.role = role_key
        st.rerun()

    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            color:{text_color};
            padding:10px 14px;
            border-radius:10px;
            font-size:13px;
            margin-top:4px;
            min-height:60px;
        ">
            {description}
        </div>
        """,
        unsafe_allow_html=True
    )

def landing_page():
    """Landing screen where the user chooses their role."""
    init_session_state()

    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🌆 Future City Intelligence</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; font-size:18px; margin-top:4px;'>"
        "Choose your perspective to explore Heilbronn's environment."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        role_card(
            title="Resident",
            emoji="👤",
            description="Simple, friendly insights about today's air, heat and noise.",
            role_key="resident",
            bg_color=PASTEL['card'],
            text_color=PASTEL['text'],
        )

    with col2:
        role_card(
            title="Smart City Controller",
            emoji="🚨",
            description="Live alerts, air-risk episodes and noise disturbances.",
            role_key="controller",
            bg_color=DARK_NEON['card'],
            text_color=DARK_NEON['text'],
        )

    with col3:
        role_card(
            title="City Planner",
            emoji="🏙️",
            description="Long-term trends, correlations and tree priority hotspots.",
            role_key="planner",
            bg_color=PLANNER['card'],
            text_color=PLANNER['text'],
        )

    st.markdown("<br><hr>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; font-size:13px; opacity:0.7;'>"
        "You can always return here using the sidebar."
        "</p>",
        unsafe_allow_html=True,
    )

# =====================================================================
#  PART 3 — RESIDENT DASHBOARD (pastel theme)
# =====================================================================

def resident_city_mood(health_score):
    """Return emoji + message based on average health score."""
    if health_score > 80:
        return "😄 Excellent — Enjoy the fresh air!"
    elif health_score > 60:
        return "🙂 Good — Conditions are mostly fine."
    elif health_score > 40:
        return "😐 Mixed — Some pollution may be noticeable."
    elif health_score > 20:
        return "😟 Poor — Sensitive groups take care."
    return "😡 Very Bad — Avoid outdoor activities."

def resident_air_panel(df):
    st.subheader("🌫 Air Quality Overview")

    pollutants = ["no2_ugm3", "pm2_5_ugm3", "pm10_ugm3", "o3_ugm3"]
    labels = ["NO₂ (µg/m³)", "PM2.5 (µg/m³)", "PM10 (µg/m³)", "O₃ (µg/m³)"]

    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for col, label, pol in zip(cols, labels, pollutants):
        if pol in df.columns:
            col.metric(label, f"{df[pol].mean():.1f}", help="Daily average concentration")
        else:
            col.metric(label, "—", help="Missing in data")

    st.write("**PM2.5 is the most relevant pollutant for health.**")

    if "pm2_5_ugm3" in df.columns:
        df_reset = df.reset_index()
        thr = df["pm2_5_ugm3"].quantile(0.90)

        fig = px.line(
            df_reset,
            x="timestamp",
            y="pm2_5_ugm3",
            title="PM2.5 over time",
        )
        fig.update_traces(line_color=PASTEL["primary"], name="PM2.5")

        fig.add_hline(
            y=thr,
            line_dash="dash",
            line_color="red",
            annotation_text="Upper health limit",
            annotation_position="top left"
        )

        if "is_anomaly" in df.columns:
            anomalies = df[df["is_anomaly"] == 1].reset_index()
            if not anomalies.empty:
                fig.add_scatter(
                    x=anomalies["timestamp"],
                    y=anomalies["pm2_5_ugm3"],
                    mode="markers",
                    marker=dict(color="red", size=6),
                    name="Anomaly",
                )

        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("PM2.5 data not available in this dataset.")

    st.write("**Air Quality Hotspots Map**")
    try:
        deck = map_layer(df, value_col="pm2_5_ugm3", color=[200, 0, 80])
        st.pydeck_chart(deck)
    except Exception:
        st.warning("Map view unavailable — missing coordinates.")

def resident_heat_panel(df):
    st.subheader("🔥 Heat Comfort")

    if "temperature" not in df.columns:
        st.info("Temperature data not available.")
        return

    avg_temp = df["temperature"].mean()
    hot_hours = (df["temperature"] > 30).sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Temperature (°C)", f"{avg_temp:.1f}")
    col2.metric("Hot Hours (>30°C)", f"{hot_hours}")

    df_reset = df.reset_index()
    fig = px.line(
        df_reset,
        x="timestamp",
        y="temperature",
        title="Temperature over time",
    )
    fig.update_traces(line_color="#FF7F50", name="Temperature")

    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="red",
        annotation_text="Heat discomfort threshold",
        annotation_position="top left",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

def resident_noise_panel(df):
    st.subheader("🔊 Noise Comfort")

    if "noise" not in df.columns:
        st.info("Noise data not available.")
        return

    avg_noise = df["noise"].mean()
    loud_hours = (df["noise"] > 70).sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Noise (dB)", f"{avg_noise:.1f}")
    col2.metric("Loud Hours (>70 dB)", f"{loud_hours}")

    df_reset = df.reset_index()
    fig = px.line(
        df_reset,
        x="timestamp",
        y="noise",
        title="Noise over time",
    )
    fig.update_traces(line_color="#8A2BE2", name="Noise (dB)")

    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Noise discomfort threshold",
        annotation_position="top left",
    )

    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1].reset_index()
        if not anomalies.empty:
            fig.add_scatter(
                x=anomalies["timestamp"],
                y=anomalies["noise"],
                mode="markers",
                marker=dict(color="red", size=6),
                name="Anomaly",
            )

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("**Noise Hotspots Map**")
    try:
        deck = map_layer(df, value_col="noise", color=[0, 80, 255])
        st.pydeck_chart(deck)
    except Exception:
        st.warning("Map not available — missing coordinates.")

def resident_dashboard(df):
    apply_resident_theme()
    fix_label_colors()

    st.title("👤 Resident View — How is the city today?")

    df_range = select_date_range(df)
    df_health = add_health_score(df_range.copy())

    avg_health = df_health["health_score"].mean()

    st.subheader("🌈 City Mood")
    st.markdown(f"<h2>{resident_city_mood(avg_health)}</h2>", unsafe_allow_html=True)

    metric_card(
        title="Wellbeing Score",
        value=f"{avg_health:.1f}/100",
        subtitle="Based on clean air & comfort levels today",
        color="#FFFFFF"
    )

    st.subheader("💓 City Health Score Over Time")

    df_h_reset = df_health.reset_index()
    fig = px.line(
        df_h_reset,
        x="timestamp",
        y="health_score",
        title="City Health Score Over Time",
    )
    fig.update_traces(line_color=PASTEL["primary"], name="Health Score")
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    resident_air_panel(df_range)

    st.markdown("<hr>", unsafe_allow_html=True)
    resident_heat_panel(df_range)

    st.markdown("<hr>", unsafe_allow_html=True)
    resident_noise_panel(df_range)

# =====================================================================
#  PART 4 — SMART CITY CONTROLLER DASHBOARD (dark neon, alerts)
# =====================================================================

def controller_dashboard(df):
    apply_controller_theme()
    fix_label_colors()

    st.title("🚨 Smart City Control Hub")

    page = st.sidebar.radio(
        "Controller Views",
        ["Dashboard", "Air-Risk Episodes", "Night Noise Disturbances", "Incident Map"],
        key="controller_nav"
    )

    df_health = add_health_score(df.copy())

    if page == "Dashboard":
        controller_main_dashboard(df, df_health)
    elif page == "Air-Risk Episodes":
        controller_air_risk_page(df_health)
    elif page == "Night Noise Disturbances":
        controller_night_noise_page(df)
    elif page == "Incident Map":
        controller_incident_map_page(df_health)

def controller_main_dashboard(df, df_health):
    st.subheader("System Health Overview")

    df_range = select_date_range(df)
    df_h_range = df_health.loc[df_range.index.intersection(df_health.index)]

    avg_health = df_h_range["health_score"].mean()
    pollution_peaks = (df_h_range["health_score"] < 40).sum()
    avg_noise = df_range["noise"].mean() if "noise" in df_range.columns else np.nan
    loud_hours = (df_range["noise"] > 70).sum() if "noise" in df_range.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Health Score", f"{avg_health:.1f}")
    c2.metric("Hours with bad air (score<40)", f"{pollution_peaks}")
    c3.metric("Avg Noise (dB)", f"{avg_noise:.1f}" if not np.isnan(avg_noise) else "—")
    c4.metric("Loud Hours (>70 dB)", f"{loud_hours}")

    st.markdown("---")

    st.subheader("Health Score Timeline (with risk band)")
    df_h_reset = df_h_range.reset_index()

    fig = px.line(
        df_h_reset,
        x="timestamp",
        y="health_score",
        title="Health Score Timeline",
    )
    fig.update_traces(line_color=DARK_NEON["primary"], name="Health Score")

    fig.add_hline(
        y=40,
        line_dash="dash",
        line_color=DARK_NEON["danger"],
        annotation_text="Air-risk threshold (score<40)",
        annotation_position="top left",
    )

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Pollutant Levels (95th percentile bands)")
    pollutants = ["no2_ugm3", "pm10_ugm3", "pm2_5_ugm3", "o3_ugm3"]
    rows = []
    for pol in pollutants:
        if pol in df_range.columns:
            series = df_range[pol]
            thr = series.quantile(0.95)
            rows.append({
                "Pollutant": pol,
                "Mean": series.mean(),
                "95th percentile threshold": thr,
            })

    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Pollutant").round(2))
    else:
        st.info("No pollutant columns found for this period.")

def controller_air_risk_page(df_health):
    st.subheader("Air-Risk Episodes (Health Score < 40 & Noise > 70 dB)")

    df_alert, episodes = detect_air_risk_alerts(df_health)

    if episodes.empty:
        st.success("✅ No multi-hour air-risk episodes detected for this period.")
        return

    st.info("Episodes are periods where health score < 40 and noise > 70 dB for ≥ 3 hours.")
    st.dataframe(episodes)

    st.subheader("Timeline with Air-Risk Episodes Shaded")

    df_alert_reset = df_alert.reset_index()

    fig = px.line(
        df_alert_reset,
        x="timestamp",
        y="health_score",
        title="Health Score with Air-Risk Episodes",
    )
    fig.update_traces(line_color=DARK_NEON["primary"], name="Health Score")

    fig.add_hline(
        y=40,
        line_dash="dash",
        line_color=DARK_NEON["danger"],
        annotation_text="Health threshold (40)",
        annotation_position="top left",
    )

    # Shade air_risk_alert True segments (using vrects)
    in_episode = False
    start = None
    for idx, row in df_alert.iterrows():
        if row.get("air_risk_alert", False) and not in_episode:
            in_episode = True
            start = idx
        elif not row.get("air_risk_alert", False) and in_episode:
            in_episode = False
            fig.add_vrect(x0=start, x1=idx, fillcolor="red", opacity=0.15, line_width=0)
    if in_episode:
        fig.add_vrect(
            x0=start,
            x1=df_alert.index[-1],
            fillcolor="red",
            opacity=0.15,
            line_width=0,
        )

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

def controller_night_noise_page(df):
    st.subheader("Night-Time Noise Disturbances")

    st.caption("Night defined as 22:00–06:00. Event = noise > 60 dB for ≥ 60 minutes.")

    df_events, night_events = detect_night_noise_events(df)

    if night_events.empty:
        st.success("✅ No night-time disturbance episodes detected for this period.")
    else:
        st.dataframe(night_events)

    st.subheader("Noise Timeline with Night Disturbances Highlighted")

    df_range = select_date_range(df)

    if "noise" not in df_range.columns:
        st.info("Noise data not available.")
        return

    df_reset = df_range.reset_index()
    fig = px.line(
        df_reset,
        x="timestamp",
        y="noise",
        title="Noise Timeline with Night Periods",
    )
    fig.update_traces(line_color="#58A6FF", name="Noise (dB)")

    fig.add_hline(
        y=60,
        line_dash="dash",
        line_color="#F0883E",
        annotation_text="Night disturbance threshold (60 dB)",
        annotation_position="top left",
    )

    # Highlight night-time periods with vrects
    hours = df_range.index.hour
    is_night = (hours >= 22) | (hours < 6)
    in_block = False
    start = None
    for ts, night in zip(df_range.index, is_night):
        if night and not in_block:
            in_block = True
            start = ts
        elif not night and in_block:
            in_block = False
            fig.add_vrect(
                x0=start,
                x1=ts,
                fillcolor="grey",
                opacity=0.08,
                line_width=0,
            )
    if in_block:
        fig.add_vrect(
            x0=start,
            x1=df_range.index[-1],
            fillcolor="grey",
            opacity=0.08,
            line_width=0,
        )

    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

def controller_incident_map_page(df_health):
    st.subheader("Geospatial View of Air-Risk Hotspots")

    df_alert, episodes = detect_air_risk_alerts(df_health)
    df_alert = df_alert[df_alert.get("air_risk_alert", False)]

    if df_alert.empty or "latitude" not in df_alert.columns or "longitude" not in df_alert.columns:
        st.warning("No geo-located air-risk alerts available.")
        return

    try:
        deck = map_layer(df_alert, value_col="health_score", color=[255, 80, 80])
        st.pydeck_chart(deck)
    except Exception as e:
        st.warning(f"Could not render map: {e}")

# =====================================================================
#  PART 5 — CITY PLANNER DASHBOARD (eco theme, long-term planning)
# =====================================================================

def planner_noise_time_travel_page():
    st.subheader("⏳ Temporal Noise Map (Time Travel)")

    st.markdown("Use the slider to see how Heilbronn's soundscape changes throughout the day.")

    time_options = {
        "09:00": 9,
        "12:00": 12,
        "15:00": 15,
        "00:00 (Midnight)": 0
    }

    selected_label = st.select_slider(
        "Select Time of Day",
        options=list(time_options.keys()),
        value="12:00"
    )

    selected_hour = time_options[selected_label]

    map_data, scenario_desc = simulate_hourly_noise_data(selected_hour, n_points=500)

    avg_noise = map_data["noise_db"].mean()
    col1, col2 = st.columns(2)
    col1.metric("Scenario", scenario_desc)
    col2.metric("City Average Noise", f"{avg_noise:.1f} dB")

    def get_noise_color(db):
        if db < 50:
            return [0, 255, 128, 140]  # Quiet (Green)
        elif db < 70:
            return [255, 200, 0, 160]  # Moderate (Yellow)
        else:
            return [255, 0, 0, 200]  # Loud (Red)

    map_data["color"] = map_data["noise_db"].apply(get_noise_color)
    map_data = map_data.round({"latitude": 4, "longitude": 4, "noise_db": 1})

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["longitude", "latitude"],
        get_radius=100,
        get_fill_color="color",
        pickable=True,
        opacity=0.8,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        filled=True,
    )

    view_state = pdk.ViewState(
        latitude=49.1427,
        longitude=9.2109,
        zoom=13,
        pitch=45
    )

    tooltip = {
        "html": "<b>🔊 Noise Level: {noise_db} dB</b><br/>"
                "🕒 Time: " + selected_label + "<br/>"
                "<hr style='margin: 5px 0; border: 0; border-top: 1px solid #666;'/>"
                "🌐 Lat: {latitude}<br/>"
                "🌐 Lon: {longitude}",
        "style": {"backgroundColor": "#1f2937", "color": "white", "fontSize": "12px", "padding": "10px",
                  "zIndex": "9999"}
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )

    st.pydeck_chart(r)

def planner_dashboard(df):
    apply_planner_theme()
    fix_label_colors()

    st.title("🏙 City Planner Dashboard")

    page = st.sidebar.radio(
        "Planner Views",
        [
            "Overview",
            "Correlation Analysis",
            "Tree Priority & Chronic Stress",
            "Sensor Relationship Explorer",
            "Tree Priority Map",
            "Noise Time-Travel Map"
        ],
        key="planner_nav"
    )

    if page == "Overview":
        planner_overview_page(df)
    elif page == "Correlation Analysis":
        planner_correlation_page(df)
    elif page == "Tree Priority & Chronic Stress":
        planner_tree_priority_page(df)
    elif page == "Sensor Relationship Explorer":
        planner_sensor_relationship_page(df)
    elif page == "Tree Priority Map":
        planner_tree_map_page(df)
    elif page == "Noise Time-Travel Map":
        planner_noise_time_travel_page()

# ---------------------------------------------------------
#  PLANNER: Overview Page
# ---------------------------------------------------------

def planner_overview_page(df):
    st.subheader("📊 Long-Term Environmental Overview")

    df_range = select_date_range(df)
    df_health = add_health_score(df_range.copy())

    avg_health = df_health["health_score"].mean()
    chronic_air = (df_health["health_score"] < 40).mean()
    chronic_noise = (df_range["noise"] > 70).mean() if "noise" in df_range.columns else 0.0

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Health Score", f"{avg_health:.1f}")
    col2.metric("Chronic Air Stress (%)", f"{chronic_air*100:.1f}")
    col3.metric("Chronic Noise Stress (%)", f"{chronic_noise*100:.1f}")

    st.markdown("### Health Score Over Time")

    df_h_reset = df_health.reset_index()
    fig = px.line(
        df_h_reset,
        x="timestamp",
        y="health_score",
        title="Health Score Over Time",
    )
    fig.update_traces(line_color=PLANNER["primary"], name="Health Score")
    fig.add_hline(
        y=40,
        line_dash="dash",
        line_color="red",
        annotation_text="Air-risk threshold",
        annotation_position="top left",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Key Pollutant Trends")
    pollutants = ["no2_ugm3", "pm10_ugm3", "pm2_5_ugm3", "o3_ugm3"]
    for pol in pollutants:
        if pol not in df_range.columns:
            continue
        st.markdown(f"#### {pol}")
        df_pol = df_range[[pol]].reset_index()
        fig = px.line(
            df_pol,
            x="timestamp",
            y=pol,
            title=None,
        )
        fig.update_traces(line_color=PLANNER["accent"], name=pol)
        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
#  PLANNER: Correlation Analysis
# ---------------------------------------------------------

def planner_correlation_page(df):
    st.subheader("📈 Correlation Matrix Between Environmental Factors")

    df_range = select_date_range(df)

    cols = [
        "no2_ugm3", "pm10_ugm3", "pm2_5_ugm3", "o3_ugm3",
        "temperature", "humidity", "noise"
    ]
    cols = [c for c in cols if c in df_range.columns]

    if len(cols) < 2:
        st.warning("Not enough variables for correlation analysis.")
        return

    corr = df_range[cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="Greens",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Green = positive correlation, dark green = strong positive, white ≈ zero, brown = negative.")

# ---------------------------------------------------------
#  PLANNER: Tree Priority & Chronic Stress
# ---------------------------------------------------------

def planner_tree_priority_page(df):
    st.subheader("🌳 Tree Priority Index & Chronic Stress Analysis")

    df_range = select_date_range(df)
    df_health = add_health_score(df_range.copy())

    chronic_air, chronic_noise, priority = compute_tree_priority(df_health)

    col1, col2, col3 = st.columns(3)
    col1.metric("Chronic Air Stress (%)", f"{chronic_air*100:.1f}")
    col2.metric("Chronic Noise Stress (%)", f"{chronic_noise*100:.1f}")
    col3.metric("Tree Priority Score", f"{priority:.1f}")

    st.write("""
    **Interpretation:**  
    - Chronic Air Stress → % of time health score < 40  
    - Chronic Noise Stress → % of time noise > 70 dB  
    - Tree Priority Score → Weighted combination (0–100) guiding where green coverage is needed  
    """)

    st.markdown("---")

    st.subheader("Tree Priority Timeline (Health Score)")
    df_h_reset = df_health.reset_index()
    fig = px.line(
        df_h_reset,
        x="timestamp",
        y="health_score",
        title=None,
    )
    fig.update_traces(line_color=PLANNER["primary"], name="Health Score")
    fig.add_hline(
        y=40,
        line_dash="dash",
        line_color="red",
        annotation_text="Air-risk threshold",
        annotation_position="top left",
    )
    fig.update_layout(
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
#  PLANNER: Sensor Relationship Explorer
# ---------------------------------------------------------

def planner_sensor_relationship_page(df):
    st.subheader("🔬 Sensor Relationship Explorer")

    df_range = select_date_range(df)
    stats = analyze_sensor_relationships(df_range)

    st.write("These metrics highlight causal relationships between variables (e.g., traffic → NO₂).")
    st.dataframe(pd.DataFrame([stats]))

# ---------------------------------------------------------
#  HELPER: SIMULATE HEILBRONN CITY GRID
# ---------------------------------------------------------

def simulate_city_grid(n_points=200):
    """
    Generates synthetic sensor points specifically around Heilbronn City Center.
    Heilbronn Center Coordinates: ~49.1427° N, 9.2109° E
    """
    CENTER_LAT = 49.1427
    CENTER_LON = 9.2109

    lat_offsets = np.random.normal(0, 0.015, n_points)
    lon_offsets = np.random.normal(0, 0.025, n_points)

    lats = CENTER_LAT + lat_offsets
    lons = CENTER_LON + lon_offsets

    dist_from_center = np.sqrt(lat_offsets ** 2 + lon_offsets ** 2)
    urban_factor = 1 - (dist_from_center / dist_from_center.max())

    heat = np.clip(urban_factor * 10 + np.random.normal(0, 2, n_points), 1, 10)
    noise = np.clip(urban_factor * 10 + np.random.normal(0, 3, n_points), 1, 10)
    air = np.clip(urban_factor * 10 + np.random.normal(0, 1.5, n_points), 1, 10)

    priority = (heat * 0.4) + (air * 0.4) + (noise * 0.2)
    priority = np.clip(priority * 10, 0, 100)

    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "heat_level": heat,
        "noise_level": noise,
        "air_quality_gap": air,
        "tree_priority": priority
    })

# ---------------------------------------------------------
#  HELPER: SIMULATE NOISE BY HOUR (TEMPORAL LOGIC)
# ---------------------------------------------------------

def simulate_hourly_noise_data(hour, n_points=400):
    """
    Generates noise data for Heilbronn based on the time of day.
    Logic:
    - 09:00: High traffic noise on main roads (Morning Rush).
    - 12:00: High activity in city center (Lunch/Shops).
    - 15:00: Moderate traffic + School run.
    - 00:00: Generally quiet, but some hotspots (Bars/Clubs).
    """
    CENTER_LAT = 49.1427
    CENTER_LON = 9.2109

    lat_offsets = np.random.normal(0, 0.015, n_points)
    lon_offsets = np.random.normal(0, 0.025, n_points)
    lats = CENTER_LAT + lat_offsets
    lons = CENTER_LON + lon_offsets

    dists = np.sqrt(lat_offsets ** 2 + lon_offsets ** 2)
    urban_density = 1 - (dists / dists.max())  # 1 = Center, 0 = Outskirts

    base_noise = np.clip(urban_density * 60 + np.random.normal(0, 5, n_points), 30, 70)

    final_noise = base_noise.copy()

    if hour == 9:
        final_noise += 15
        scenario = "🚗 Morning Rush Hour"
    elif hour == 12:
        final_noise += (urban_density * 20)
        scenario = "🍽️ City Center Bustle"
    elif hour == 15:
        final_noise += 10
        scenario = "🚌 Afternoon Activity"
    elif hour == 0 or hour == 24:
        final_noise -= 20
        nightlife_indices = np.random.choice(n_points, size=int(n_points * 0.05), replace=False)
        final_noise[nightlife_indices] = 85
        scenario = "🌙 Midnight (Mostly Quiet)"
    else:
        scenario = "Normal"

    final_noise = np.clip(final_noise, 30, 95)

    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "noise_db": final_noise
    }), scenario

# ---------------------------------------------------------
#  PLANNER: Tree Priority Map (Fixed Number Formatting)
# ---------------------------------------------------------

def planner_tree_map_page(df):
    st.subheader("🗺️ Heilbronn Green Intervention Map")

    map_data = simulate_city_grid(n_points=300)

    map_data = map_data.round({
        "latitude": 4,
        "longitude": 4,
        "heat_level": 1,
        "air_quality_gap": 1,
        "noise_level": 1,
        "tree_priority": 0
    })

    def get_color(score):
        if score < 50:
            return [0, 255, 128, 160]
        elif score < 75:
            return [255, 200, 0, 180]
        else:
            return [255, 0, 0, 200]

    map_data["color"] = map_data["tree_priority"].apply(get_color)

    critical_zones = len(map_data[map_data["tree_priority"] > 75])

    col1, col2 = st.columns(2)
    col1.metric("Total Monitored Zones", len(map_data))
    col2.metric("🔥 Critical Heat/Air Zones", critical_zones, delta="High Priority", delta_color="inverse")

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["longitude", "latitude"],
        get_radius=120,
        get_fill_color="color",
        pickable=True,
        opacity=0.9,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        filled=True,
    )

    view_state = pdk.ViewState(
        latitude=49.1427,
        longitude=9.2109,
        zoom=13.5,
        pitch=45,
        bearing=0
    )

    tooltip = {
        "html": "<b>📍 Zone Priority: {tree_priority}</b><br/>"
                "🌡 Heat Stress: {heat_level}<br/>"
                "🌫 Air Quality: {air_quality_gap}<br/>"
                "📢 Noise Level: {noise_level}<br/>"
                "<hr style='margin: 5px 0; border: 0; border-top: 1px solid #666;'/>"
                "🌐 Lat: {latitude}<br/>"
                "🌐 Lon: {longitude}",
        "style": {
            "backgroundColor": "#1f2937",
            "color": "white",
            "fontSize": "12px",
            "padding": "10px",
            "zIndex": "9999"
        }
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )

    st.pydeck_chart(r)

    st.success("💡 **Map Updated:** Coordinates and metrics are now displayed correctly.")

# =====================================================================
#  PART 6 — MAIN APP ASSEMBLY
# =====================================================================

def main():
    st.set_page_config(
        page_title="Future City Intelligence Dashboard",
        layout="wide",
        page_icon="🌆",
    )

    init_session_state()

    df = load_data()

    if st.session_state.role is None:
        landing_page()
        return

    st.sidebar.title("Future City Intelligence")
    # st.sidebar.markdown(f"**Active Role:** `{st.session_state.role.capitalize()}`")

    if st.sidebar.button("⬅ Back to Role Selection"):
        st.session_state.role = None
        st.rerun()

    st.sidebar.markdown("---")

    role = st.session_state.role

    if role == "resident":
        resident_dashboard(df)
    elif role == "controller":
        controller_dashboard(df)
    elif role == "planner":
        planner_dashboard(df)

# =====================================================================
if __name__ == "__main__":
    main()
