"""Preprocessing for the climate-zone task.

Two stages, kept separate on purpose:

1. `make_features(df)`  -- deterministic, row-by-row transformations (no statistics are
   learned), so it can be applied to the whole dataset before splitting without leakage:
     * wind direction + speed  -> wind vector components (u, v); calm wind -> (0, 0)
     * Date                    -> day-of-year on a circle (sin, cos); the year is never used
     * Rainfall (and Evaporation) -> log1p, because they are heavily right-skewed
     * RainToday is dropped: it is exactly (Rainfall > 1 mm), so it carries no new information

2. `build_preprocessor(...)` -- the steps that *learn* statistics (median, mean, std).
   They live inside an sklearn Pipeline and are therefore fitted on training data only.
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 16-point compass -> angle in degrees (direction the wind blows FROM, clockwise from north)
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
DIR_TO_DEG = {d: i * 22.5 for i, d in enumerate(COMPASS)}

WIND_PAIRS = {  # (direction column, speed column, output prefix)
    "Gust": ("WindGustDir", "WindGustSpeed"),
    "Wind9am": ("WindDir9am", "WindSpeed9am"),
    "Wind3pm": ("WindDir3pm", "WindSpeed3pm"),
}

# Base measurements that every station records (<= ~11% missing, few stations fully missing).
CORE_NUMERIC = ["MinTemp", "MaxTemp", "Rainfall_log1p", "WindGustSpeed", "WindSpeed9am",
                "WindSpeed3pm", "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
                "Temp9am", "Temp3pm"]
WIND_VECTOR = [f"{p}_{c}" for p in WIND_PAIRS for c in ("u", "v")]
SEASON = ["doy_sin", "doy_cos"]
# Sensor columns that are missing for WHOLE stations (12-19 of 48 stations) -> fingerprint risk.
STATION_SENSOR = ["Sunshine", "Evaporation_log1p", "Cloud9am", "Cloud3pm"]

FEATURE_SETS = {
    "core": CORE_NUMERIC + WIND_VECTOR + SEASON,                    # default for the zone task
    "full": CORE_NUMERIC + WIND_VECTOR + SEASON + STATION_SENSOR,   # sensitivity analysis only
}


def wind_vector(direction, speed):
    """Meteorological convention: u = -s*sin(theta), v = -s*cos(theta).

    A missing direction with speed == 0 is a calm, not a missing value: in this data
    100% of zero-speed rows have no direction. Calm wind is therefore encoded as (0, 0).
    Any other missing direction or speed stays NaN and is imputed later.
    """
    theta = np.deg2rad(direction.map(DIR_TO_DEG))
    u = -speed * np.sin(theta)
    v = -speed * np.cos(theta)
    calm = speed.eq(0)
    u[calm], v[calm] = 0.0, 0.0
    return u, v


def make_features(df):
    """Row-wise feature construction. Returns a new DataFrame; the input is not modified."""
    out = pd.DataFrame(index=df.index)
    for col in ["MinTemp", "MaxTemp", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
                "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm", "Temp9am",
                "Temp3pm", "Sunshine", "Cloud9am", "Cloud3pm"]:
        out[col] = df[col]
    out["Rainfall_log1p"] = np.log1p(df["Rainfall"])
    out["Evaporation_log1p"] = np.log1p(df["Evaporation"])
    for prefix, (dcol, scol) in WIND_PAIRS.items():
        out[f"{prefix}_u"], out[f"{prefix}_v"] = wind_vector(df[dcol], df[scol])
    doy = df["Date"].dt.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def build_preprocessor(add_indicator=False):
    """Median imputation (+ optional missing indicators) followed by standardisation.

    Must be placed inside a Pipeline together with the model so that the medians, means
    and standard deviations are learned from the training fold only.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("scale", StandardScaler()),
    ])


def select(features, feature_set="core"):
    """Pick the columns of one feature set from the output of make_features."""
    return features[FEATURE_SETS[feature_set]]
