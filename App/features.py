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


# 0) HEALTH SCORE  -----------------------------------------------------------

def add_health_score(
    df: pd.DataFrame,
    no2_col: str = "NO2",
    pm10_col: str = "PM10",
    low_q: float = 0.05,
    high_q: float = 0.95,
) -> pd.DataFrame:
    """
    Add PollutionIndex and HealthScore (0–100, higher = better air)
    based on NO2 and PM10, using 5–95 percentile scaling.
    Works with your current clean_data_dummy.csv columns.
    """
    _require_columns(df, (no2_col, pm10_col))
    df = df.copy()

    poll_cols = [no2_col, pm10_col]

    # compute percentiles for scaling (safe for empty/constant data)
    bounds = {}
    for col in poll_cols:
        series = df[col]
        if series.dropna().empty:
            low = np.nan
            high = np.nan
        else:
            low = float(series.quantile(low_q))
            high = float(series.quantile(high_q))
        bounds[col] = (low, high)

    def scaled(series: pd.Series, col: str) -> pd.Series:
        low, high = bounds[col]
        # if bounds are invalid or equal, fallback to min/max scaling (or zeros for all-na)
        if not np.isfinite(low) or not np.isfinite(high) or high == low:
            if series.isna().all():
                return pd.Series(0.0, index=series.index)
            smin = series.min()
            smax = series.max()
            denom = (smax - smin) if smax != smin else 1.0
            return np.clip((series - smin) / denom, 0.0, 1.0)
        denom = high - low
        return np.clip((series - low) / denom, 0.0, 1.0)

    for col in poll_cols:
        df[col + "_scaled"] = scaled(df[col], col)

    # weights: 0.6 NO2, 0.4 PM10 (you can tweak)
    df["pollution_index"] = (
        0.6 * df[no2_col + "_scaled"] +
        0.4 * df[pm10_col + "_scaled"]
    )

    df["health_score"] = (1 - df["pollution_index"]) * 100
    return df


# 1) SMART CITY CONTROLLER ALERTS  ------------------------------------------

def detect_air_risk_alerts(
    df: pd.DataFrame,
    health_threshold: float = 40,
    noise_threshold: float = 70.0,
    min_hours: int = 3,
    noise_col: str = "noise",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Smart City Controller alerts for a single point.

    Conditions per time step:
      - bad_air: HealthScore < health_threshold
      - loud: noise > noise_threshold

    An Air Risk episode is when these are true for >= min_hours consecutive time steps.
    """
    _require_columns(df, ("health_score", noise_col))
    df = prepare_time_index(df)
    df = df.copy()

    df["bad_air"] = df["health_score"] < health_threshold
    df["loud"] = df[noise_col] > noise_threshold

    # placeholder so logic matches the description (wind/rain could be added later)
    df["stagnant"] = True

    df["air_risk_alert"] = df["bad_air"] & df["stagnant"] & df["loud"]

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
    episodes = episodes[episodes["duration_hours"] >= float(min_hours)].reset_index(drop=True)
    return df, episodes


# 2) NIGHT-TIME NOISE DISTURBANCE DETECTOR  ---------------------------------

def detect_night_noise_events(
    df: pd.DataFrame,
    noise_threshold: float = 60.0,
    min_duration_minutes: int = 60,
    noise_col: str = "noise",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Night-time noise disturbance episodes.

    Night: 22:00–06:00
    Disturbance: noise > noise_threshold
    Episode: continuous disturbance for >= min_duration_minutes.
    """
    _require_columns(df, (noise_col,))
    df = prepare_time_index(df)
    df = df.copy()

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
    events = events[events["duration_min"] >= float(min_duration_minutes)].reset_index(drop=True)
    return df, events


# 3) TREE PRIORITY SCORE  ----------------------------------------------------

def compute_tree_priority(
    df: pd.DataFrame,
    health_threshold: float = 40,
    noise_threshold: float = 65,
    noise_col: str = "noise",
) -> tuple[float, float, float]:
    """
    Long-term Green Intervention Planner for a single point.

    - chronic_air   = fraction of time with health_score < health_threshold
    - chronic_noise = fraction of time with noise > noise_threshold
    - tree_priority = 100 * (0.6*chronic_air + 0.4*chronic_noise)
    """
    _require_columns(df, ("health_score", noise_col))
    df = df.copy()

    df["bad_air"] = df["health_score"] < health_threshold
    df["loud"] = df[noise_col] > noise_threshold

    chronic_air = float(df["bad_air"].mean())    # 0..1
    chronic_noise = float(df["loud"].mean())    # 0..1

    tree_priority = 100 * (0.6 * chronic_air + 0.4 * chronic_noise)
    return chronic_air, chronic_noise, tree_priority