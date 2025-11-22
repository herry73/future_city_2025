

# =====================================================================
#  PART 1 — IMPORTS + THEMES + UTILITIES + UNIFIED DATA LOADER
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydeck as pdk
from datetime import datetime, date

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
        body {{
            background-color: {PASTEL['bg']} !important;
            color: {PASTEL['text']} !important;
        }}
        .stMetricValue, .stMetricLabel {{
            color: {PASTEL['text']} !important;
        }}
        .stButton>button {{
            background-color: {PASTEL['accent']} !important;
            color: white !important;
            border-radius: 10px;
            border: none;
        }}
        .stTab {{
            color: {PASTEL['text']} !important;
        }}
    </style>
    """, unsafe_allow_html=True)



def apply_controller_theme():
    st.markdown(f"""
    <style>
        body {{
            background-color: {DARK_NEON['bg']} !important;
            color: {DARK_NEON['text']} !important;
        }}
        .stMetricValue, .stMetricLabel {{
            color: {DARK_NEON['text']} !important;
        }}
        .stButton>button {{
            background-color: {DARK_NEON['accent']} !important;
            color: white !important;
            border-radius: 10px;
            border: none;
        }}
    </style>
    """, unsafe_allow_html=True)



def apply_planner_theme():
    st.markdown(f"""
    <style>
        body {{
            background-color: {PLANNER['bg']} !important;
            color: {PLANNER['text']} !important;
        }}
        .stMetricValue, .stMetricLabel {{
            color: {PLANNER['text']} !important;
        }}
        .stButton>button {{
            background-color: {PLANNER['accent']} !important;
            color: white !important;
            border-radius: 10px;
            border: none;
        }}
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
#  UTILITY — ADD THRESHOLD BANDS
# ---------------------------------------------------------

def draw_threshold(ax, threshold, label, color="red"):
    ax.axhline(threshold, linestyle="--", color=color, linewidth=1.5, label=label)
    ax.legend()


# ---------------------------------------------------------
#  UTILITY — HIGHLIGHT ANOMALIES ON PLOT
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
#  PART 2 — LANDING PAGE + ROLE SELECTION (patched for rerun)
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


# ---------- PATCH: proper rerun-safe callbacks ----------

def set_role(role):
    """Instantly update role and rerun app."""
    st.session_state.role = role
    st.rerun()


def reset_role():
    """Reset the role and rerun to landing page."""
    st.session_state.role = None
    st.rerun()


# --------------------------------------------------------

def role_card(title, emoji, description, role_key, bg_color, text_color):
    """Clickable card used on landing page."""

    # Button (patched to use callback instead of state check)
    st.button(
        f"{emoji}  {title}",
        key=f"role_btn_{role_key}",
        use_container_width=True,
        on_click=lambda r=role_key: set_role(r)
    )

    # Description box below each button
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
#  PART 3 — RESIDENT DASHBOARD (experimental, pastel theme)
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
        col.metric(label, f"{df[pol].mean():.1f}", help="Daily average concentration")

    # Plot main pollutant for residents: PM2.5
    st.write("**PM2.5 is the most relevant pollutant for health.**")
    fig, ax = plt.subplots(figsize=(8, 3))
    df["pm2_5_ugm3"].plot(ax=ax, label="PM2.5")
    draw_threshold(ax, df["pm2_5_ugm3"].quantile(0.90), "Upper health limit")
    highlight_anomalies(ax, df, "pm2_5_ugm3")
    st.pyplot(fig)

    # Air quality map
    st.write("**Air Quality Hotspots Map**")
    try:
        deck = map_layer(df, value_col="pm2_5_ugm3", color=[200, 0, 80])
        st.pydeck_chart(deck)
    except:
        st.warning("Map view unavailable — missing coordinates.")


def resident_heat_panel(df):
    st.subheader("🔥 Heat Comfort")

    avg_temp = df["temperature"].mean()
    hot_hours = (df["temperature"] > 30).sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Temperature (°C)", f"{avg_temp:.1f}")
    col2.metric("Hot Hours (>30°C)", f"{hot_hours}")

    fig, ax = plt.subplots(figsize=(8, 3))
    df["temperature"].plot(ax=ax, color="#FF7F50")
    draw_threshold(ax, 30, "Heat discomfort threshold")
    st.pyplot(fig)


def resident_noise_panel(df):
    st.subheader("🔊 Noise Comfort")

    avg_noise = df["noise"].mean()
    loud_hours = (df["noise"] > 70).sum()

    col1, col2 = st.columns(2)
    col1.metric("Average Noise (dB)", f"{avg_noise:.1f}")
    col2.metric("Loud Hours (>70 dB)", f"{loud_hours}")

    fig, ax = plt.subplots(figsize=(8, 3))
    df["noise"].plot(ax=ax, color="#8A2BE2")
    draw_threshold(ax, 70, "Noise discomfort threshold")
    highlight_anomalies(ax, df, "noise")
    st.pyplot(fig)

    st.write("**Noise Hotspots Map**")
    try:
        deck = map_layer(df, value_col="noise", color=[0, 80, 255])
        st.pydeck_chart(deck)
    except:
        st.warning("Map not available — missing coordinates.")


def resident_dashboard(df):
    apply_resident_theme()

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
    fig, ax = plt.subplots(figsize=(8,3))
    df_health["health_score"].plot(ax=ax, color="#6BB6FF")
    st.pyplot(fig)

    # Detailed panels
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

    st.title("🚨 Smart City Control Hub")

    # Controller-only navigation
    page = st.sidebar.radio(
        "Controller Views",
        ["Dashboard", "Air-Risk Episodes", "Night Noise Disturbances", "Incident Map"],
        key="controller_nav"
    )

    # Precompute health score once
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
    avg_noise = df_range["noise"].mean()
    loud_hours = (df_range["noise"] > 70).sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Health Score", f"{avg_health:.1f}")
    c2.metric("Hours with bad air (score<40)", f"{pollution_peaks}")
    c3.metric("Avg Noise (dB)", f"{avg_noise:.1f}")
    c4.metric("Loud Hours (>70 dB)", f"{loud_hours}")

    st.markdown("---")

    st.subheader("Health Score Timeline (with risk band)")
    fig, ax = plt.subplots(figsize=(9, 3))
    df_h_range["health_score"].plot(ax=ax, color=DARK_NEON["primary"])
    draw_threshold(ax, 40, "Air-risk threshold (score<40)", color=DARK_NEON["danger"])
    st.pyplot(fig)

    st.subheader("Key Pollutant Levels (95th percentile bands)")
    pollutants = ["no2_ugm3", "pm10_ugm3", "pm2_5_ugm3", "o3_ugm3"]
    rows = []
    for pol in pollutants:
        series = df_range[pol]
        thr = series.quantile(0.95)
        rows.append({
            "Pollutant": pol,
            "Mean": series.mean(),
            "95th percentile threshold": thr,
        })

    st.dataframe(pd.DataFrame(rows).set_index("Pollutant").round(2))


def controller_air_risk_page(df_health):
    st.subheader("Air-Risk Episodes (Health Score < 40 & Noise > 70 dB)")

    df_alert, episodes = detect_air_risk_alerts(df_health)

    if episodes.empty:
        st.success("✅ No multi-hour air-risk episodes detected for this period.")
        return

    st.info("Episodes are periods where health score < 40 and noise > 70 dB for ≥ 3 hours.")

    st.dataframe(episodes)

    # Plot with episode shading
    st.subheader("Timeline with Air-Risk Episodes Shaded")

    fig, ax = plt.subplots(figsize=(10, 3))
    df_alert["health_score"].plot(ax=ax, color=DARK_NEON["primary"], label="Health Score")
    draw_threshold(ax, 40, "Health threshold (40)", color=DARK_NEON["danger"])

    # Shade air_risk_alert True segments
    in_episode = False
    start = None
    for idx, row in df_alert.iterrows():
        if row["air_risk_alert"] and not in_episode:
            in_episode = True
            start = idx
        elif not row["air_risk_alert"] and in_episode:
            in_episode = False
            ax.axvspan(start, idx, color="red", alpha=0.15)
    if in_episode:
        ax.axvspan(start, df_alert.index[-1], color="red", alpha=0.15)

    st.pyplot(fig)


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

    fig, ax = plt.subplots(figsize=(10, 3))
    df_range["noise"].plot(ax=ax, color="#58A6FF", label="Noise (dB)")
    draw_threshold(ax, 60, "Night disturbance threshold (60 dB)", color="#F0883E")

    # Highlight night-time periods visually
    hours = df_range.index.hour
    is_night = (hours >= 22) | (hours < 6)
    ax.fill_between(df_range.index, df_range["noise"].min(), df_range["noise"].max(),
                    where=is_night, color="grey", alpha=0.08, label="Night hours")

    ax.legend()
    st.pyplot(fig)


def controller_incident_map_page(df_health):
    st.subheader("Geospatial View of Air-Risk Hotspots")

    df_alert, episodes = detect_air_risk_alerts(df_health)
    df_alert = df_alert[df_alert["air_risk_alert"]]

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

def planner_dashboard(df):
    apply_planner_theme()

    st.title("🏙 City Planner Dashboard")

    # Planner-specific sidebar navigation
    page = st.sidebar.radio(
        "Planner Views",
        [
            "Overview",
            "Correlation Analysis",
            "Tree Priority & Chronic Stress",
            "Sensor Relationship Explorer",
            "Tree Priority Map"
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


# ---------------------------------------------------------
#  PLANNER: Overview Page
# ---------------------------------------------------------

def planner_overview_page(df):
    st.subheader("📊 Long-Term Environmental Overview")

    df_range = select_date_range(df)
    df_health = add_health_score(df_range.copy())

    avg_health = df_health["health_score"].mean()
    chronic_air = (df_health["health_score"] < 40).mean()
    chronic_noise = (df_range["noise"] > 70).mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Health Score", f"{avg_health:.1f}")
    col2.metric("Chronic Air Stress (%)", f"{chronic_air*100:.1f}")
    col3.metric("Chronic Noise Stress (%)", f"{chronic_noise*100:.1f}")

    st.markdown("### Health Score Over Time")
    fig, ax = plt.subplots(figsize=(9,4))
    df_health["health_score"].plot(ax=ax, color=PLANNER["primary"])
    draw_threshold(ax, 40, "Air-risk threshold")
    st.pyplot(fig)

    st.markdown("---")

    st.subheader("Key Pollutant Trends")
    pollutants = ["no2_ugm3", "pm10_ugm3", "pm2_5_ugm3", "o3_ugm3"]
    for pol in pollutants:
        st.markdown(f"#### {pol}")
        fig, ax = plt.subplots(figsize=(9,3))
        df_range[pol].plot(ax=ax, color=PLANNER["accent"])
        st.pyplot(fig)


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

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="Greens", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    fig.colorbar(im)
    st.pyplot(fig)

    st.markdown("Green = positive correlation, dark green = strong positive, white = zero, brown = negative.")


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

    st.subheader("Tree Priority Timeline")
    fig, ax = plt.subplots(figsize=(9,4))
    df_health["health_score"].plot(ax=ax, color=PLANNER["primary"])
    draw_threshold(ax, 40, "Air-risk threshold")
    st.pyplot(fig)


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
    # Heilbronn Marktplatz / City Center
    CENTER_LAT = 49.1427
    CENTER_LON = 9.2109

    # Generate points within ~2-3km radius of the city center
    # Using normal distribution to cluster points near the center
    lat_offsets = np.random.normal(0, 0.015, n_points)
    lon_offsets = np.random.normal(0, 0.025, n_points)

    lats = CENTER_LAT + lat_offsets
    lons = CENTER_LON + lon_offsets

    # Simulate Environment Stress (High density near center = High Stress)
    # Calculate distance from center to simulate "Urban Heat Island" effect
    dist_from_center = np.sqrt(lat_offsets ** 2 + lon_offsets ** 2)
    # Closer to center = higher base stress (0.0 to 1.0 inverted)
    urban_factor = 1 - (dist_from_center / dist_from_center.max())

    # Generate metrics with randomness + urban factor
    heat = np.clip(urban_factor * 10 + np.random.normal(0, 2, n_points), 1, 10)
    noise = np.clip(urban_factor * 10 + np.random.normal(0, 3, n_points), 1, 10)
    air = np.clip(urban_factor * 10 + np.random.normal(0, 1.5, n_points), 1, 10)

    # Weighted Priority: Heat & Air are critical for Heilbronn
    priority = (heat * 0.4) + (air * 0.4) + (noise * 0.2)
    # Scale to 0-100
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
#  PLANNER: Tree Priority Map (Fixed Base Map)
# ---------------------------------------------------------

def planner_tree_map_page(df):
    st.subheader("🗺️ Heilbronn Green Intervention Map")

    # --- 1. Generate Data for Heilbronn (Center Locked) ---
    # 模拟海尔布隆市中心的数据点
    map_data = simulate_city_grid(n_points=300)

    # --- 2. Define Color Logic ---
    # Green (Safe) -> Yellow (Warning) -> Red (Critical)
    def get_color(score):
        if score < 50:
            return [0, 255, 128, 160]  # Greenish
        elif score < 75:
            return [255, 200, 0, 180]  # Orange
        else:
            return [255, 0, 0, 200]  # Red

    map_data["color"] = map_data["tree_priority"].apply(get_color)

    critical_zones = len(map_data[map_data["tree_priority"] > 75])

    col1, col2 = st.columns(2)
    col1.metric("Total Monitored Zones", len(map_data))
    col2.metric("🔥 Critical Heat/Air Zones", critical_zones, delta="High Priority", delta_color="inverse")

    # --- 3. Pydeck Map Configuration ---
    layer = pdk.Layer(
        "ScatterplotLayer",
        map_data,
        get_position=["longitude", "latitude"],
        get_radius=120,  # 稍微调小一点半径，让点更清晰
        get_fill_color="color",
        pickable=True,
        opacity=0.9,
        stroked=True,
        get_line_color=[255, 255, 255],  # 给圆点加个白边，更像第一张图的风格
        line_width_min_pixels=1,
        filled=True,
    )

    # Initial View: Focused on Heilbronn City Center
    view_state = pdk.ViewState(
        latitude=49.1427,
        longitude=9.2109,
        zoom=13.5,
        pitch=45,
        bearing=0
    )

    # Tooltip
    tooltip = {
        "html": "<b>📍 Zone Priority: {tree_priority:.0f}</b><br/>"
                "🌡 Heat Stress: {heat_level:.1f}<br/>"
                "🌫 Air Quality: {air_quality_gap:.1f}<br/>"
                "📢 Noise Level: {noise_level:.1f}",
        "style": {
            "backgroundColor": "#1f2937",
            "color": "white",
            "fontSize": "12px",
            "padding": "10px"
        }
    }

    # --- 4. Render Map with CARTO Style (No Token Needed) ---
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        # 关键修改在这里：使用 'carto' 提供商和 'dark' 样式
        map_provider="carto",
        map_style="dark",
    )

    st.pydeck_chart(r)

    st.success("💡 **Map Loaded:** This view uses the OpenStreetMap/CARTO dark theme, which requires no API token.")
# =====================================================================
#  PART 6 — MAIN APP ASSEMBLY (patched for rerun & clean routing)
# =====================================================================

def main():
    st.set_page_config(
        page_title="Future City Intelligence Dashboard",
        layout="wide",
        page_icon="🌆",
    )

    init_session_state()

    # Load unified data once
    df = load_data()

    # If no role chosen → show landing
    if st.session_state.role is None:
        landing_page()
        return

    # Sidebar shared across roles
    st.sidebar.title("Future City Intelligence")
    st.sidebar.markdown(f"**Active Role:** `{st.session_state.role.capitalize()}`")

    # ---------- PATCH: back button uses callback ----------
    st.sidebar.button("⬅ Back to Role Selection", on_click=reset_role)
    st.sidebar.markdown("---")

    # Route to dashboards
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
