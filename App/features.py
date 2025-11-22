# python
import pandas as pd
import numpy as np
from typing import Iterable


def _require_columns(df: pd.DataFrame, cols: Iterable[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")


def prepare_time_index(df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
    """
    Ensure df has a DatetimeIndex based on time_col.
    Your dashboard already sets index to timestamp when loading real/dummy data,
    so this is mostly a safety helper.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col)
    return df.sort_index()


def _median_step_minutes(index: pd.DatetimeIndex, default_minutes: float) -> float:
    if len(index) < 2:
        return float(default_minutes)
    diffs = np.diff(index.values).astype("timedelta64[s]").astype(float)
    med_seconds = float(np.median(diffs))
    med_minutes = med_seconds / 60.0
    return med_minutes if med_minutes > 0 else float(default_minutes)



# ---------------------------------------------------------------------------
# 0) HEALTH SCORE  -----------------------------------------------------------
# ---------------------------------------------------------------------------

def add_health_score(
    df: pd.DataFrame,
    no2_col: str = "no2_ugm3",
    pm25_col: str = "pm2_5_ugm3",
    pm10_col: str = "pm10_ugm3",
    o3_col: str = "o3_ugm3",
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> pd.DataFrame:
    """
    Add PollutionIndex and HealthScore (0–100, higher = better air)
    using NO2, PM2.5, PM10 and O3 from the synthetic smart-city dataset.
    """

    required = [no2_col, pm25_col, pm10_col, o3_col]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df.copy()

    poll_cols = [no2_col, pm25_col, pm10_col, o3_col]

    # compute 5–95 percentile bounds
    bounds = {}
    for col in poll_cols:
        s = df[col]
        if s.dropna().empty:
            low = high = np.nan
        else:
            low = float(s.quantile(low_q))
            high = float(s.quantile(high_q))
        bounds[col] = (low, high)

    def scaled(series: pd.Series, col: str) -> pd.Series:
        low, high = bounds[col]
        if not np.isfinite(low) or not np.isfinite(high) or low == high:
            # fallback: normalise by min/max
            if series.isna().all():
                return pd.Series(0.0, index=series.index)
            smin, smax = series.min(), series.max()
            denom = (smax - smin) if smax != smin else 1.0
            return np.clip((series - smin) / denom, 0, 1)
        return np.clip((series - low) / (high - low), 0, 1)

    # Scale all pollutants
    for col in poll_cols:
        df[col + "_scaled"] = scaled(df[col], col)

    # Weighted pollution index
    df["pollution_index"] = (
        0.35 * df[no2_col + "_scaled"] +
        0.30 * df[pm25_col + "_scaled"] +
        0.25 * df[pm10_col + "_scaled"] +
        0.10 * df[o3_col + "_scaled"]
    )

    df["health_score"] = (1 - df["pollution_index"]) * 100
    return df



# ---------------------------------------------------------------------------
# 1) SMART CITY CONTROLLER ALERTS -------------------------------------------
# ---------------------------------------------------------------------------

def detect_air_risk_alerts(
    df: pd.DataFrame,
    health_threshold: float = 40,
    noise_threshold: float = 70.0,
    min_hours: int = 3,
    noise_col: str = "noise",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    _require_columns(df, ("health_score", noise_col))
    df = prepare_time_index(df).copy()

    df["bad_air"] = df["health_score"] < health_threshold
    df["loud"] = df[noise_col] > noise_threshold
    df["stagnant"] = True   # placeholder for wind/rain in real dataset

    df["air_risk_alert"] = df["bad_air"] & df["loud"] & df["stagnant"]

    step_minutes = _median_step_minutes(df.index, 60.0)
    s = df["air_risk_alert"]
    block_id = (s != s.shift()).cumsum()

    episodes = (
        df[s]
        .groupby(block_id[s])
        .agg(
            start=("air_risk_alert", "idxmin"),
            end=("air_risk_alert", "idxmax"),
            steps=("air_risk_alert", "size"),
            avg_health=("health_score", "mean"),
            avg_noise=(noise_col, "mean"),
        )
    )

    episodes["duration_hours"] = episodes["steps"] * step_minutes / 60.0
    episodes = episodes[episodes["duration_hours"] >= min_hours].reset_index(drop=True)
    return df, episodes



# ---------------------------------------------------------------------------
# 2) NIGHT NOISE EVENT DETECTOR ---------------------------------------------
# ---------------------------------------------------------------------------

def detect_night_noise_events(
    df: pd.DataFrame,
    noise_threshold: float = 60.0,
    min_duration_minutes: int = 60,
    noise_col: str = "noise",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    _require_columns(df, (noise_col,))
    df = prepare_time_index(df).copy()

    step_minutes = _median_step_minutes(df.index, float(min_duration_minutes))
    hours = df.index.hour
    is_night = (hours >= 22) | (hours < 6)

    df["noise_disturbance"] = is_night & (df[noise_col] > noise_threshold)

    s = df["noise_disturbance"]
    block_id = (s != s.shift()).cumsum()

    events = (
        df[s]
        .groupby(block_id[s])
        .agg(
            start=("noise_disturbance", "idxmin"),
            end=("noise_disturbance", "idxmax"),
            steps=("noise_disturbance", "size"),
            max_noise=(noise_col, "max"),
            avg_noise=(noise_col, "mean"),
        )
    )

    events["duration_min"] = events["steps"] * step_minutes
    events = events[events["duration_min"] >= min_duration_minutes].reset_index(drop=True)
    return df, events



# ---------------------------------------------------------------------------
# 3) TREE PRIORITY SCORE -----------------------------------------------------
# ---------------------------------------------------------------------------

def compute_tree_priority(
    df: pd.DataFrame,
    health_threshold: float = 40,
    noise_threshold: float = 65,
    noise_col: str = "noise",
) -> tuple[float, float, float]:

    _require_columns(df, ("health_score", noise_col))
    df = df.copy()

    df["bad_air"] = df["health_score"] < health_threshold
    df["loud"] = df[noise_col] > noise_threshold

    chronic_air = float(df["bad_air"].mean())
    chronic_noise = float(df["loud"].mean())

    tree_priority = 100 * (0.6 * chronic_air + 0.4 * chronic_noise)
    return chronic_air, chronic_noise, tree_priority



# ---------------------------------------------------------------------------
# 4) SENSOR RELATIONSHIP EXPLORER -------------------------------------------
# ---------------------------------------------------------------------------

def analyze_sensor_relationships(
    df: pd.DataFrame,
    no2_col: str = "no2_ugm3",
    noise_col: str = "noise",
    o3_col: str = "o3_ugm3",
    temp_col: str = "temperature_degC",
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
) -> dict:

    required = [no2_col, noise_col, o3_col, temp_col]
    for c in required:
        if c not in df.columns:
            return {
                "corr_no2_traffic": np.nan,
                "no2_traffic_factor": np.nan,
                "corr_o3_sunlight": np.nan,
                "o3_sunlight_factor": np.nan,
            }

    df = prepare_time_index(df)

    out = {
        "corr_no2_traffic": float(df[no2_col].corr(df[noise_col])),
        "corr_o3_sunlight": float(df[o3_col].corr(df[temp_col])),
    }

    # High vs low noise grouping
    hi_noise = df[df[noise_col] >= df[noise_col].quantile(high_quantile)][no2_col]
    lo_noise = df[df[noise_col] <= df[noise_col].quantile(low_quantile)][no2_col]

    if len(hi_noise) > 0 and len(lo_noise) > 0:
        out["no2_traffic_factor"] = float(hi_noise.mean() / lo_noise.mean())
    else:
        out["no2_traffic_factor"] = np.nan

    # High vs low temperature grouping (proxy for sunlight)
    hi_temp = df[df[temp_col] >= df[temp_col].quantile(high_quantile)][o3_col]
    lo_temp = df[df[temp_col] <= df[temp_col].quantile(low_quantile)][o3_col]

    if len(hi_temp) > 0 and len(lo_temp) > 0:
        out["o3_sunlight_factor"] = float(hi_temp.mean() / lo_temp.mean())
    else:
        out["o3_sunlight_factor"] = np.nan

    return out



# ---------------------------------------------------------------------------
# 5) EMISSION ANOMALY DETECTION  --------------------------------------------
# ---------------------------------------------------------------------------

def detect_emission_anomalies(
    df: pd.DataFrame,
    pollutant_cols=("no2_ugm3", "pm10_ugm3", "pm2_5_ugm3", "o3_ugm3"),
    window=24,
    threshold=3.5,
):
    """
    Detect emission anomalies using rolling Median Absolute Deviation (MAD).
    Adds *_anom columns for each pollutant.
    """
    df = df.copy()

    for col in pollutant_cols:
        if col not in df.columns:
            continue

        roll_med = df[col].rolling(window=window, min_periods=1).median()
        mad = (np.abs(df[col] - roll_med)).rolling(window=window, min_periods=1).median()

        modified_z = 0.6745 * (df[col] - roll_med) / (mad + 1e-6)
        df[col + "_anom"] = (np.abs(modified_z) > threshold).astype(int)

    return df



# ---------------------------------------------------------------------------
# 6) HEATWAVE EARLY WARNING --------------------------------------------------
# ---------------------------------------------------------------------------

def detect_heatwave_periods(
    df: pd.DataFrame,
    temp_col="temperature_degC",
    humidity_col="humidity_percent",
    temp_thr=30,
    hum_thr=40,
    min_hours=3
):
    """
    Detect heatwave episodes using temperature + humidity:
    - temperature >= temp_thr
    - humidity <= hum_thr
    - persistence >= min_hours
    """

    df = prepare_time_index(df).copy()

    df["hot"] = df[temp_col] >= temp_thr
    df["dry"] = df[humidity_col] <= hum_thr
    df["heatwave"] = df["hot"] & df["dry"]

    s = df["heatwave"]
    block_id = (s != s.shift()).cumsum()

    episodes = (
        df[s]
        .groupby(block_id[s])
        .agg(
            start=("heatwave", "idxmin"),
            end=("heatwave", "idxmax"),
            hours=("heatwave", "size"),
            avg_temp=(temp_col, "mean"),
            avg_humidity=(humidity_col, "mean"),
        )
    )

    episodes = episodes[episodes["hours"] >= min_hours].reset_index(drop=True)
    return df, episodes
