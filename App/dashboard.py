# ====================================================================
#  PART 1 — IMPORTS + GLOBAL STYLES + UTILITIES + DATA LOADER
# ====================================================================

import time
import base64
from datetime import datetime, date

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # kept in case you still use it somewhere
import pydeck as pdk
import streamlit as st
from PIL import Image
import os

import plotly.express as px
import plotly.graph_objects as go

# =============================================================
# Load external CSS (global styles)
# =============================================================

def load_local_css(file_path: str):
    """Loads a local CSS file into the Streamlit app."""
    try:
        with open(file_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"⚠️ CSS file not found: {file_path}")

# Auto-detect correct relative path
css_path = "App/styles.css" if "App" in __file__ else "styles.css"
load_local_css(css_path)

# Import backend analysis functions
from features import (
    add_health_score,
    detect_air_risk_alerts,
    detect_night_noise_events,
    compute_tree_priority,
)

# ---------------------------------------------------------
#  GLOBAL STYLE PALETTES
# ---------------------------------------------------------

PASTEL = {
    "bg": "#F2F6FF",
    "text": "#1F2A44",
    "card": "#FFFFFF",
    "accent": "#6BA8FF",
    "primary": "#4C84E0",
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
    "bg": "#E8F5E9",
    "text": "#1B5E20",
    "card": "#FFFFFF",
    "accent": "#66BB6A",
    "primary": "#388E3C",
}

# ---------------------------------------------------------
#  ROLE-BASED THEMES (Inject CSS)
# ---------------------------------------------------------

def apply_resident_theme():
    st.markdown(
        f"""
    <style>

    [data-testid="stSidebar"] code {{
        background-color: #4C84E0 !important;
        color: white !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }}

    [data-testid="stHeader"] {{
        background-color: #EAF0FF !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background-color: {PASTEL['bg']} !important;
        color: {PASTEL['text']} !important;
    }}

    [data-testid="stSidebar"] {{
        background-color: #F7FAFF !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {PASTEL['text']} !important;
    }}

    h1, h2, h3, h4, h5, h6 {{
        color: {PASTEL['text']} !important;
        font-weight: 700 !important;
    }}

    .stTextInput input, 
    .stDateInput input, 
    .stSelectbox div[role='combobox'] {{
        background-color: white !important;
        color: {PASTEL['text']} !important;
        border-radius: 8px;
        border: 1px solid #BFD7FF !important;
    }}

    .stDataFrame table, .stDataFrame th, .stDataFrame td {{
        color: {PASTEL['text']} !important;
        background-color: white !important;
    }}

    .stButton>button {{
        background-color: {PASTEL['accent']} !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
    }}

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
    """,
        unsafe_allow_html=True,
    )


def apply_controller_theme():
    st.markdown(
        f"""
    <style>

    [data-testid="stSidebar"] code {{
        background-color: #FF3B8D !important;
        color: white !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }}

    [data-testid="stHeader"] {{
        background-color: #1A2238 !important;
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
    """,
        unsafe_allow_html=True,
    )


def apply_planner_theme():
    st.markdown(
        f"""
    <style>

    [data-testid="stSidebar"] code {{
        background-color: #1B5E20 !important;
        color: white !important;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
    }}

    [data-testid="stHeader"] {{
        background-color: #DFF3E3 !important;
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
    """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
#  TREE ICON UTILITIES (tint + base64 for IconLayer)
# ---------------------------------------------------------

def tint_tree(input_path, output_path, color):
    """Tint base tree icon and save as PNG."""
    try:
        img = Image.open(input_path).convert("RGBA")
    except FileNotFoundError:
        return
    data = img.getdata()
    new_data = []
    r_t, g_t, b_t = color

    for (r, g, b, a) in data:
        if a == 0:
            new_data.append((r, g, b, a))
            continue
        brightness = (r + g + b) / 3 / 255
        new_r = int(r_t * brightness)
        new_g = int(g_t * brightness)
        new_b = int(b_t * brightness)
        new_data.append((new_r, new_g, new_b, a))

    img.putdata(new_data)
    img.save(output_path, "PNG")


# Pre-generate colored tree icons (safe if missing)
tint_tree("data/green tree.png", "data/processed/tree_green.png", (40, 180, 40))
tint_tree("data/green tree.png", "data/processed/tree_yellow.png", (255, 215, 0))
tint_tree("data/green tree.png", "data/processed/tree_red.png", (255, 60, 60))


def _load_icon_data(path: str) -> str:
    """Return a data: URL for a local PNG (works reliably in IconLayer)."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


TREE_ICON_GREEN = _load_icon_data("data/processed/tree_green.png")
TREE_ICON_YELLOW = _load_icon_data("data/processed/tree_yellow.png")
TREE_ICON_RED = _load_icon_data("data/processed/tree_red.png")

# ---------------------------------------------------------
#  GENERIC UI UTILITIES
# ---------------------------------------------------------

def fix_label_colors():
    st.markdown(
        """
    <style>
    label, .stDateInput label, .stTextInput label, .stSelectbox label {
        color: #1A1A1A !important;
        font-weight: 600 !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )


def select_date_range(df):
    """User-friendly date range selection over index."""
    min_date = df.index.min().date()
    max_date = df.index.max().date()

    col1, col2 = st.columns(2)
    start = col1.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end = col2.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

    mask = (df.index.date >= start) & (df.index.date <= end)
    return df.loc[mask]


def draw_threshold(ax, threshold, label, color="red"):
    ax.axhline(threshold, linestyle="--", color=color, linewidth=1.5, label=label)
    ax.legend()


def highlight_anomalies(ax, df, column):
    if "is_anomaly" in df.columns:
        anomalies = df[df["is_anomaly"] == 1]
        if len(anomalies):
            ax.scatter(anomalies.index, anomalies[column], color="red", s=30, label="Anomaly")


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
#  SIMPLE PYDECK SCATTER MAP (used in some panels)
# ---------------------------------------------------------

def map_layer(df, lat_col="latitude", lon_col="longitude", value_col=None, color=[0, 128, 255]):
    """
    Very simple ScatterplotLayer that always works as long as lat/lon exist.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        st.warning("No latitude/longitude columns found in dataframe.")
        return None

    df_map = df[[lat_col, lon_col]].dropna().copy()
    if df_map.empty:
        st.warning("No valid coordinates to display on map.")
        return None

    if value_col and value_col in df.columns:
        df_map[value_col] = df[value_col]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_map,
        get_position=[lon_col, lat_col],
        get_radius=80,
        get_fill_color=color,
        pickable=True,
        opacity=0.8,
    )

    view_state = pdk.ViewState(
        latitude=float(df_map[lat_col].mean()),
        longitude=float(df_map[lon_col].mean()),
        zoom=12,
        pitch=40,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_provider="carto",
        map_style="dark",
    )
    return deck

# ---------------------------------------------------------
#  UNIFIED DATA LOADER  (with synthetic hourly expansion)
# ---------------------------------------------------------
def load_data():
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "..", "data", "clean_data.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


    hourly_rows = []
    for _, row in air.iterrows():
        base_date = row["timestamp"].date()
        for hour in range(24):
            ts = pd.Timestamp(
                year=base_date.year, month=base_date.month, day=base_date.day, hour=hour
            )

            hour_factor = np.sin((hour / 24) * 2 * np.pi)

            no2 = row["no2_ugm3"] + hour_factor * 8
            pm10 = row["pm10_ugm3"] + np.random.randn() * 1.2
            pm25 = row["pm2_5_ugm3"] + hour_factor * 4
            o3 = row["o3_ugm3"] + np.cos((hour / 24) * 2 * np.pi) * 6

            temp = (
                row["temperature_degC"]
                + 6 * np.sin((hour - 6) / 24 * 2 * np.pi)
                + np.random.randn() * 0.4
            )

            humid = (
                row["humidity_percent"]
                - 10 * np.sin((hour - 6) / 24 * 2 * np.pi)
                + np.random.randn() * 1
            )

            noise_level = (
                45
                + 12 * (hour in [7, 8, 9, 17, 18])
                + 8 * (hour >= 22 or hour <= 2)
                + np.random.randn() * 3
            )

            hourly_rows.append(
                {
                    "timestamp": ts,
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "no2_ugm3": max(no2, 0),
                    "pm10_ugm3": max(pm10, 0),
                    "pm2_5_ugm3": max(pm25, 0),
                    "o3_ugm3": max(o3, 0),
                    "temperature": temp,
                    "humidity": humid,
                    "noise": noise_level,
                    "is_anomaly": 0,
                }
            )

    air = pd.DataFrame(hourly_rows)

    air["timestamp"] = pd.to_datetime(
        air["timestamp"],
        format="%m/%d/%Y %H:%M",
        errors="coerce",
    )
    air = air.rename(
        columns={
            "NO2": "no2_ugm3",
            "PM25": "pm2_5_ugm3",
            "PM10": "pm10_ugm3",
            "O3": "o3_ugm3",
        }
    )

    weather = pd.read_csv("data/processed/weather.csv", parse_dates=["timestamp"])
    noise = pd.read_csv("data/processed/noise.csv", parse_dates=["timestamp"])

    weather = weather.rename(
        columns={"temperature_degC": "temperature", "humidity_percent": "humidity"}
    )
    noise = noise.rename(columns={"noise_db": "noise"})

    df = (
        air.merge(weather, on="timestamp", how="outer", suffixes=("", "_wx"))
        .merge(noise, on="timestamp", how="outer", suffixes=("", "_nx"))
        .sort_values("timestamp")
    )

    df = df.loc[:, ~df.columns.duplicated()]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# ---------------------------------------------------------
#  PLANNER: GREEN WALK – Best Time to Go Outside
# ---------------------------------------------------------

def planner_green_walk_page(df):
    st.subheader("🌿 Green Walk — Best Time to Go Outside")

    st.markdown(
        "Choose a date to see the full-day air quality curve and get a recommendation "
        "on when it’s healthiest to go outside."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_date = st.date_input(
            "Select Date",
            value=df.index.min().date(),
            min_value=df.index.min().date(),
            max_value=df.index.max().date(),
        )

    df_day = df[[ts.date() == selected_date for ts in df.index]]

    if df_day.empty:
        st.warning("No data available for this date.")
        return

    df_day_health = add_health_score(df_day.copy())

    st.markdown("### 📈 Air Quality Throughout the Day")
    st.line_chart(df_day_health["health_score"], height=250)

    st.markdown("### 🌤 Green Walk Recommendation")

    avg_score = df_day_health["health_score"].mean()
    best_hour = df_day_health["health_score"].idxmax().strftime("%H:%M")
    worst_hour = df_day_health["health_score"].idxmin().strftime("%H:%M")

    reasons = []
    if df_day["no2_ugm3"].mean() < df["no2_ugm3"].quantile(0.4):
        reasons.append("low NO₂ levels")
    if df_day["pm2_5_ugm3"].mean() < df["pm2_5_ugm3"].quantile(0.4):
        reasons.append("clean PM2.5 levels")
    if df_day["o3_ugm3"].mean() < df["o3_ugm3"].quantile(0.4):
        reasons.append("low ozone concentration")
    if not reasons:
        reasons.append("conditions are moderate, but acceptable")

    st.metric("Best Time to Go Out", best_hour, delta="Based on lowest pollution")
    st.metric("Worst Time", worst_hour, delta="Based on highest pollution")

    if avg_score > 60:
        st.success(
            f"⭐ **Recommendation:** Average score: {avg_score:.1f}/100 — "
            f"You can go outside! Reasons: {', '.join(reasons)}."
        )
    else:
        st.warning(
            f"⚠️ **Warning:** Score only {avg_score:.1f}. Avoid long outdoor activities. "
            f"Main issues: {', '.join(reasons)}."
        )

# =====================================================================
#  PART 2 — LANDING PAGE + ROLE SELECTION
# =====================================================================

def init_session_state():
    if "role" not in st.session_state:
        st.session_state.role = None
    if "resident_page" not in st.session_state:
        st.session_state.resident_page = "Today"
    if "controller_page" not in st.session_state:
        st.session_state.controller_page = "Dashboard"
    if "planner_page" not in st.session_state:
        st.session_state.planner_page = "Overview"


def role_card(title, emoji, description, role_key, bg_color, text_color):
    """(Unused now, but kept in case you want the clickable HTML cards again)."""
    card_html = f"""
    <div class="role-card" onclick="selectRole('{role_key}')"
         style="
             cursor: pointer;
             background-color:{bg_color};
             color:{text_color};
             padding:25px;
             border-radius:15px;
             min-height:220px;
             display:flex;
             flex-direction:column;
             justify-content:center;
             align-items:center;
             box-shadow:0 4px 12px rgba(0,0,0,0.25);
             border:2px solid rgba(255,255,255,0.05);
             transition:0.2s;
         ">
         <div style="font-size:60px; margin-bottom:10px;">{emoji}</div>
         <h2 style="margin:0; padding:0; font-weight:700;">{title}</h2>
         <p style="margin-top:10px; opacity:0.85; text-align:center;">
             {description}
         </p>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def landing_page():
    init_session_state()

    st.markdown(
        '<div class="landing-title">🌆 Future City Intelligence</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="landing-subtitle">Choose your perspective to explore Heilbronn\'s environment.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    sp1, col1, col2, col3, sp2 = st.columns([1, 3, 3, 3, 1])

    with col1:
        st.markdown('<div class="role-card role-card-resident">', unsafe_allow_html=True)
        if st.button(
            "👤\n\nResident\n\nSimple, friendly insights about today's air, heat and noise.",
            key="resident_card",
        ):
            st.session_state.role = "resident"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="role-card role-card-controller">', unsafe_allow_html=True)
        if st.button(
            "🚨\n\nSmart City Controller\n\nLive alerts, air-risk episodes and noise disturbances.",
            key="controller_card",
        ):
            st.session_state.role = "controller"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="role-card role-card-planner">', unsafe_allow_html=True)
        if st.button(
            "🏙️\n\nCity Planner\n\nLong-term trends, correlations and tree priority hotspots.",
            key="planner_card",
        ):
            st.session_state.role = "planner"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.markdown(
        '<div class="footer-note">You can always return here using the sidebar.</div>',
        unsafe_allow_html=True,
    )

# =====================================================================
#  PART 3 — RESIDENT DASHBOARD
# =====================================================================

def resident_city_mood(health_score):
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

        fig = px.line(df_reset, x="timestamp", y="pm2_5_ugm3", title="PM2.5 over time")
        fig.update_traces(line_color=PASTEL["primary"], name="PM2.5")

        fig.add_hline(
            y=thr,
            line_dash="dash",
            line_color="red",
            annotation_text="Upper health limit",
            annotation_position="top left",
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
    deck = map_layer(df, value_col="pm2_5_ugm3", color=[200, 0, 80])
    if deck is not None:
        st.pydeck_chart(deck)


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
    fig = px.line(df_reset, x="timestamp", y="temperature", title="Temperature over time")
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
    fig = px.line(df_reset, x="timestamp", y="noise", title="Noise over time")
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
    deck = map_layer(df, value_col="noise", color=[0, 80, 255])
    if deck is not None:
        st.pydeck_chart(deck)


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
        color="#FFFFFF",
    )

    st.subheader("💓 City Health Score Over Time")
    df_h_reset = df_health.reset_index()
    fig = px.line(df_h_reset, x="timestamp", y="health_score", title="City Health Score Over Time")
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
#  PART 4 — SMART CITY CONTROLLER DASHBOARD
# =====================================================================

def controller_dashboard(df):
    apply_controller_theme()
    fix_label_colors()

    st.title("🚨 Smart City Control Hub")

    page = st.sidebar.radio(
        "Controller Views",
        ["Dashboard", "Air-Risk Episodes", "Night Noise Disturbances", "Incident Map"],
        key="controller_nav",
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

    fig = px.line(df_h_reset, x="timestamp", y="health_score", title="Health Score Timeline")
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
            rows.append(
                {"Pollutant": pol, "Mean": series.mean(), "95th percentile threshold": thr}
            )

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


# ---------------------------------------------------------
#  CONTROLLER: Incident Map (Refined: Top 50 & Smaller Dots)
# ---------------------------------------------------------
def controller_incident_map_page(df_health):
    st.subheader("Geospatial View of Air-Risk Hotspots")

    # 1. 尝试严格模式检测 (健康分<40 且 噪音>70)
    df_alert, _ = detect_air_risk_alerts(df_health)
    df_map_source = df_alert[df_alert.get("air_risk_alert", False)].copy()
    
    is_simulated = False

    # 2. 自动降级策略
    if df_map_source.empty:
        st.warning("⚠ No Critical Alerts detected. Switching to 'Watchlist Mode'.")
        # 降级：看健康分低于 60 的
        df_map_source = df_health[df_health["health_score"] < 60].copy()
        is_simulated = True

    if df_map_source.empty:
        st.success("🎉 Data is perfectly clean. No risks to display.")
        return

    # 3. 【关键修改】只保留最严重的 50 个点
    # 按健康分排序（越低越严重），只取前 50 个
    # 这样能保证地图不会被“淹没”
    if len(df_map_source) > 50:
        df_map_source = df_map_source.sort_values("health_score", ascending=True).head(50)
        st.caption(f"ℹ Showing the *top 50* most critical incidents out of {len(df_alert)} potential issues.")

    # 4. 空间散射 (Jitter)
    if df_map_source["latitude"].nunique() < 2:
        np.random.seed(42) 
        # 稍微加大一点散射范围，让这50个点分得更开一点
        df_map_source["latitude"] = df_map_source["latitude"] + np.random.normal(0, 0.01, len(df_map_source))
        df_map_source["longitude"] = df_map_source["longitude"] + np.random.normal(0, 0.015, len(df_map_source))
        np.random.seed(None)

    # 5. 准备数据
    df_map = df_map_source.reset_index()
    df_map["time_str"] = df_map["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_map = df_map.round({"health_score": 1, "noise": 1, "latitude": 4, "longitude": 4})

    def get_alert_color(score):
        if score < 40: return [255, 0, 0, 200]      # Red
        return [255, 140, 0, 180]                   # Orange
    
    df_map["color"] = df_map["health_score"].apply(get_alert_color)

    # 6. Pydeck 渲染 (缩小半径)
    layer = pdk.Layer(
        "ScatterplotLayer",
        df_map,
        get_position=["longitude", "latitude"],
        get_radius=70,   # 【修改】从 200 改为 70，点变小，不挡街道
        get_fill_color="color",
        pickable=True,
        opacity=0.8,
        stroked=True,
        get_line_color=[255, 255, 255],
        line_width_min_pixels=1,
        filled=True
    )

    view_state = pdk.ViewState(
        latitude=df_map["latitude"].mean(),
        longitude=df_map["longitude"].mean(),
        zoom=12.5,
        pitch=0
    )

    tooltip = {
        "html": "<b>🚨 Incident Report</b><br/>"
                "🕒 Time: {time_str}<br/>"
                "💓 Health Score: {health_score}<br/>"
                "🔊 Noise: {noise} dB",
        "style": {"backgroundColor": "#1f2937", "color": "white", "fontSize": "12px", "padding": "10px"}
    }

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )

    st.pydeck_chart(r)

    if is_simulated:
        st.caption("Locations visualized using spatial distribution modeling based on central sensor data.")

# =====================================================================
#  PLANNER HELPERS: CITY GRID + CONTINUOUS NOISE SIMULATION
# =====================================================================

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

    return pd.DataFrame(
        {
            "latitude": lats,
            "longitude": lons,
            "heat_level": heat,
            "noise_level": noise,
            "air_quality_gap": air,
            "tree_priority": priority,
        }
    )


def simulate_continuous_noise(hour, date_obj, n_points=400):
    """
    Generates noise data for ANY hour (0-23) with continuous logic.
    This allows for smooth animation across the day.
    """
    CENTER_LAT = 49.1427
    CENTER_LON = 9.2109
    is_weekend = date_obj.weekday() >= 5

    np.random.seed(42)
    lat_offsets = np.random.normal(0, 0.015, n_points)
    lon_offsets = np.random.normal(0, 0.025, n_points)
    lats = CENTER_LAT + lat_offsets
    lons = CENTER_LON + lon_offsets
    dists = np.sqrt(lat_offsets ** 2 + lon_offsets ** 2)
    urban_density = 1 - (dists / dists.max())

    np.random.seed(None)

    time_factor = 0
    scenario = "Quiet"

    if is_weekend:
        if 0 <= hour < 8:
            time_factor = -10
            scenario = "🌙 Sleeping City"
        elif 8 <= hour < 12:
            time_factor = 5
            scenario = "☕ Slow Morning"
        elif 12 <= hour < 18:
            time_factor = 20
            scenario = "🛍️ Shopping & Leisure"
        elif 18 <= hour <= 23:
            time_factor = 25
            scenario = "🎉 Evening/Nightlife"
    else:
        if 0 <= hour < 6:
            time_factor = -15
            scenario = "🌙 Deep Night"
        elif 6 <= hour < 9:
            time_factor = 25
            scenario = "🚗 Morning Rush Hour"
        elif 9 <= hour < 16:
            time_factor = 15
            scenario = "💼 Working Hours"
        elif 16 <= hour < 19:
            time_factor = 25
            scenario = "🚗 Evening Rush Hour"
        elif 19 <= hour <= 23:
            time_factor = 5
            scenario = "🏠 Evening Relaxation"

    noise_values = 45 + (urban_density * 15) + time_factor + np.random.normal(0, 3, n_points)

    if hour > 20:
        noise_values[:10] += 30  # clubs
    if 7 < hour < 18:
        noise_values[10:50] += 15  # traffic hubs

    return (
        pd.DataFrame(
            {
                "latitude": lats,
                "longitude": lons,
                "noise_db": np.clip(noise_values, 30, 95),
            }
        ),
        scenario,
    )

# =====================================================================
#  PLANNER: Noise Time-Travel Map (Smooth Range Playback)
# =====================================================================

def planner_noise_time_travel_page():
    st.subheader("⏳ Temporal Noise Map (Time Travel)")
    st.markdown("Define a time window and watch the urban soundscape evolve hour by hour.")

    c1, c2 = st.columns([1, 2])

    with c1:
        min_d = date(2021, 1, 1)
        max_d = date(2023, 12, 31)
        default_d = date(2023, 6, 15)
        selected_date = st.date_input(
            "Select Historical Date", value=default_d, min_value=min_d, max_value=max_d
        )

        day_name = selected_date.strftime("%A")
        st.caption(f"📅 **{day_name}** ({'Weekend' if selected_date.weekday() >= 5 else 'Weekday'})")

    with c2:
        start_h, end_h = st.slider(
            "Select Time Range to Simulate",
            min_value=0,
            max_value=23,
            value=(6, 22),
            format="%02d:00",
        )

        play_btn = st.button("▶️ Play Simulation", type="primary", use_container_width=True)

    kpi_placeholder = st.empty()
    map_placeholder = st.empty()

    def render_map_frame(hour, data):
        time_str = f"{hour:02d}:00"

        data = data.round({"latitude": 4, "longitude": 4, "noise_db": 1})

        def get_col(db):
            if db < 50:
                return [0, 255, 128, 140]  # quiet green
            elif db < 65:
                return [255, 200, 0, 160]  # moderate yellow
            elif db < 75:
                return [255, 100, 0, 180]  # loud orange
            else:
                return [255, 0, 0, 200]  # very loud red

        data["color"] = data["noise_db"].apply(get_col)

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=data,
            get_position=["longitude", "latitude"],
            get_radius=120,
            get_fill_color="color",
            pickable=True,
            opacity=0.8,
            filled=True,
            transitions={"get_fill_color": 200},
        )

        view_state = pdk.ViewState(latitude=49.1427, longitude=9.2109, zoom=13, pitch=45)

        tooltip = {
            "html": f"<b>🔊 Noise: {{noise_db}} dB</b><br/>Time: {time_str}",
            "style": {
                "backgroundColor": "#1f2937",
                "color": "white",
                "fontSize": "12px",
                "padding": "10px",
            },
        }

        return pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_provider="carto",
            map_style="dark",
        )

    if play_btn:
        progress_bar = st.progress(0)
        total_steps = end_h - start_h + 1

        for i, hour in enumerate(range(start_h, end_h + 1)):
            progress_bar.progress((i + 1) / total_steps)

            map_data, scenario = simulate_continuous_noise(hour, selected_date, n_points=500)
            avg_noise = map_data["noise_db"].mean()

            with kpi_placeholder.container():
                k1, k2, k3 = st.columns(3)
                k1.metric("🕒 Clock", f"{hour:02d}:00")
                k2.metric("Scenario", scenario)
                k3.metric("Avg Noise", f"{avg_noise:.1f} dB")

            r = render_map_frame(hour, map_data)
            map_placeholder.pydeck_chart(r)

            time.sleep(0.25)

        progress_bar.empty()
        st.success(f"✅ Simulation from {start_h:02d}:00 to {end_h:02d}:00 complete.")
    else:
        map_data, scenario = simulate_continuous_noise(start_h, selected_date, n_points=500)
        avg_noise = map_data["noise_db"].mean()

        with kpi_placeholder.container():
            k1, k2, k3 = st.columns(3)
            k1.metric("🕒 Clock", f"{start_h:02d}:00")
            k2.metric("Scenario", scenario)
            k3.metric("Avg Noise", f"{avg_noise:.1f} dB")

        r = render_map_frame(start_h, map_data)
        map_placeholder.pydeck_chart(r)

# ---------------------------------------------------------
#  PLANNER: Tree Priority Map  (TREE ICONS, 4× BIGGER)
# ---------------------------------------------------------

def planner_tree_map_page(df):
    st.subheader("🗺️ Heilbronn Green Intervention Map")

    map_data = simulate_city_grid(n_points=300)

    map_data = map_data.round(
        {
            "latitude": 4,
            "longitude": 4,
            "heat_level": 1,
            "air_quality_gap": 1,
            "noise_level": 1,
            "tree_priority": 0,
        }
    )

    def choose_icon_name(priority):
        if priority < 50:
            return "green"
        elif priority < 75:
            return "yellow"
        else:
            return "red"

    map_data["icon_name"] = map_data["tree_priority"].apply(choose_icon_name)

    ICON_MAPPING = {
        "green": {"url": TREE_ICON_GREEN, "width": 64, "height": 64, "anchorY": 64},
        "yellow": {"url": TREE_ICON_YELLOW, "width": 64, "height": 64, "anchorY": 64},
        "red": {"url": TREE_ICON_RED, "width": 64, "height": 64, "anchorY": 64},
    }

    map_data["icon"] = map_data["icon_name"].map(ICON_MAPPING)

    critical_zones = (map_data["tree_priority"] > 75).sum()
    total_zones = len(map_data)

    c1, c2 = st.columns(2)
    c1.metric("Total Monitored Zones", total_zones)
    c2.metric("🔥 Critical Heat/Air Zones", int(critical_zones))

    layer = pdk.Layer(
        "IconLayer",
        data=map_data,
        get_icon="icon",
        get_position=["longitude", "latitude"],
        get_size=160,  # 4× bigger trees
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=49.1427,
        longitude=9.2109,
        zoom=13.5,
        pitch=45,
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
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_provider="carto",
        map_style="dark",
    )

    st.pydeck_chart(deck)

# ---------------------------------------------------------
#  PLANNER: Other Analytical Pages
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
    fig = px.line(df_h_reset, x="timestamp", y="health_score", title="Health Score Over Time")
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
        fig = px.line(df_pol, x="timestamp", y=pol, title=None)
        fig.update_traces(line_color=PLANNER["accent"], name=pol)
        fig.update_layout(
            hovermode="x unified",
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)


def planner_correlation_page(df):
    st.subheader("📈 Correlation Matrix Between Environmental Factors")

    df_range = select_date_range(df)

    cols = [
        "no2_ugm3",
        "pm10_ugm3",
        "pm2_5_ugm3",
        "o3_ugm3",
        "temperature",
        "humidity",
        "noise",
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


def planner_tree_priority_page(df):
    st.subheader("🌳 Tree Priority Index & Chronic Stress Analysis")

    df_range = select_date_range(df)
    df_health = add_health_score(df_range.copy())

    chronic_air, chronic_noise, priority = compute_tree_priority(df_health)

    col1, col2, col3 = st.columns(3)
    col1.metric("Chronic Air Stress (%)", f"{chronic_air*100:.1f}")
    col2.metric("Chronic Noise Stress (%)", f"{chronic_noise*100:.1f}")
    col3.metric("Tree Priority Score", f"{priority:.1f}")

    st.write(
        """
    **Interpretation:**  
    - Chronic Air Stress → % of time health score < 40  
    - Chronic Noise Stress → % of time noise > 70 dB  
    - Tree Priority Score → Weighted combination (0–100) guiding where green coverage is needed  
    """
    )

    st.markdown("---")

    st.subheader("Tree Priority Timeline (Health Score)")
    df_h_reset = df_health.reset_index()
    fig = px.line(df_h_reset, x="timestamp", y="health_score", title=None)
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
#  PLANNER DASHBOARD
# ---------------------------------------------------------

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
            "Tree Priority Map",
            "Noise Time-Travel Map",
            "Green Walk – Best Time",
        ],
        key="planner_nav",
    )

    if page == "Overview":
        planner_overview_page(df)
    elif page == "Correlation Analysis":
        planner_correlation_page(df)
    elif page == "Tree Priority & Chronic Stress":
        planner_tree_priority_page(df)
    elif page == "Tree Priority Map":
        planner_tree_map_page(df)
    elif page == "Noise Time-Travel Map":
        planner_noise_time_travel_page()
    elif page == "Green Walk – Best Time":
        planner_green_walk_page(df)

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


if __name__ == "__main__":
    main()
