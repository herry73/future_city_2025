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

    Columns required:
        - no2_ugm3
        - pm2_5_ugm3
        - pm10_ugm3
        - o3_ugm3
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

    # Scale columns
    for col in poll_cols:
        df[col + "_scaled"] = scaled(df[col], col)

    # Weighted pollution index (0 = clean, 1 = dirty)
    df["pollution_index"] = (
        0.35 * df[no2_col + "_scaled"] +
        0.30 * df[pm25_col + "_scaled"] +
        0.25 * df[pm10_col + "_scaled"] +
        0.10 * df[o3_col + "_scaled"]
    )

    # Final Health Score (bigger = better)
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

    chronic_air = float(df["bad_air"].mean())     # 0..1
    chronic_noise = float(df["loud"].mean())     # 0..1

    tree_priority = 100 * (0.6 * chronic_air + 0.4 * chronic_noise)
    return chronic_air, chronic_noise, tree_priority


# 4) SENSOR RELATIONSHIP EXPLORER  ------------------------------------------

def analyze_sensor_relationships(
    df: pd.DataFrame,
    no2_col: str = "NO2",
    noise_col: str = "noise",
    o3_col: str = "O3",
    temp_col: str = "temperature",
    high_quantile: float = 0.75,
    low_quantile: float = 0.25,
) -> dict:
    """
    Compute a small set of summary statistics used by the dashboard's
    `insights_page` (returns a dict of floats). Fields returned:
      - corr_no2_traffic: Pearson correlation NO2 vs noise
      - no2_traffic_factor: mean(NO2|high_noise)/mean(NO2|low_noise)
      - corr_o3_sunlight: Pearson correlation O3 vs temperature
      - o3_sunlight_factor: mean(O3|high_temp)/mean(O3|low_temp)

    The function is robust to missing columns or small sample sizes and will
    return np.nan for metrics it cannot compute.
    """
    # be forgiving if some columns are missing: return NaNs rather than raising
    required = [no2_col, noise_col, o3_col, temp_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        # return all keys with NaN so dashboard doesn't crash
        return {
            "corr_no2_traffic": float(np.nan),
            "no2_traffic_factor": float(np.nan),
            "corr_o3_sunlight": float(np.nan),
            "o3_sunlight_factor": float(np.nan),
        }

    df = prepare_time_index(df)

    out = {
        "corr_no2_traffic": float(np.nan),
        "no2_traffic_factor": float(np.nan),
        "corr_o3_sunlight": float(np.nan),
        "o3_sunlight_factor": float(np.nan),
    }

    # correlation NO2 vs traffic (noise)
    sub = df[[no2_col, noise_col]].dropna()
    if len(sub) >= 2:
        try:
            out["corr_no2_traffic"] = float(sub[no2_col].corr(sub[noise_col]))
        except Exception:
            out["corr_no2_traffic"] = float(np.nan)

    # ratio: mean NO2 when noise is high vs low
    try:
        thresh = float(df[noise_col].quantile(high_quantile))
        high_noisy = df.loc[df[noise_col] >= thresh, no2_col].dropna()
        low_noisy = df.loc[df[noise_col] < thresh, no2_col].dropna()
        if len(high_noisy) and len(low_noisy) and low_noisy.mean() != 0:
            out["no2_traffic_factor"] = float(high_noisy.mean() / low_noisy.mean())
        else:
            out["no2_traffic_factor"] = float(np.nan)
    except Exception:
        out["no2_traffic_factor"] = float(np.nan)

    # correlation O3 vs temperature (proxy for sunlight)
    sub2 = df[[o3_col, temp_col]].dropna()
    if len(sub2) >= 2:
        try:
            out["corr_o3_sunlight"] = float(sub2[o3_col].corr(sub2[temp_col]))
        except Exception:
            out["corr_o3_sunlight"] = float(np.nan)

    # ratio: mean O3 during high temperature vs low temperature
    try:
        high_t = float(df[temp_col].quantile(high_quantile))
        low_t = float(df[temp_col].quantile(low_quantile))
        high_temp_o3 = df.loc[df[temp_col] >= high_t, o3_col].dropna()
        low_temp_o3 = df.loc[df[temp_col] <= low_t, o3_col].dropna()
        if len(high_temp_o3) and len(low_temp_o3) and low_temp_o3.mean() != 0:
            out["o3_sunlight_factor"] = float(high_temp_o3.mean() / low_temp_o3.mean())
        else:
            out["o3_sunlight_factor"] = float(np.nan)
    except Exception:
        out["o3_sunlight_factor"] = float(np.nan)

    return out
