"""Project-wide constants for DSAA2011 Task A: "Where am I?" climate-zone classification.

Every notebook / script imports from here so that the random seed, the target
definition and the leakage exclusions are defined in exactly one place.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "weatherAUS.csv"           # Kaggle jsphyg/weather-dataset-rattle-package, version 2
RAW_SHA256 = "573fd715cd69fcacc4df32024d823b450ae3edaae7e8ff2eeb623adbed424014"
BOM_STATIONS_ZIP = ROOT / "data" / "external" / "bom_stations_IDCJMC0014_2026-11-09.zip"
LOCATION_STATION_CSV = ROOT / "data" / "external" / "location_station.csv"
LOCATION_ZONE_CSV = ROOT / "data" / "processed" / "location_zone.csv"

RANDOM_STATE = 42
TARGET = "Zone"

# ---------------------------------------------------------------------------
# Target definition: Köppen-Geiger code -> 5 project classes.
# Groups A and B are kept whole; group C is split by its 2nd/3rd letter,
# which are the only C sub-types that occur at our 49 stations.
# ---------------------------------------------------------------------------
KOPPEN_TO_ZONE = {
    "Af": "Tropical", "Am": "Tropical", "Aw": "Tropical", "As": "Tropical",
    "BWh": "Arid", "BWk": "Arid", "BSh": "Arid", "BSk": "Arid",
    "Cfa": "Subtropical", "Cwa": "Subtropical",
    "Csa": "Mediterranean", "Csb": "Mediterranean",
    "Cfb": "Oceanic", "Cfc": "Oceanic",
}
ZONE_ORDER = ["Tropical", "Arid", "Subtropical", "Mediterranean", "Oceanic"]

# ---------------------------------------------------------------------------
# Leakage audit (target = climate zone of the station).
# ---------------------------------------------------------------------------
# Hard exclusions: these columns reveal the answer or come from the future.
LEAKAGE_DROP = [
    "Location",      # the zone is a deterministic function of Location
    "RainTomorrow",  # information from the next day (future)
    "RISK_MM",       # next-day rainfall amount; absent in Kaggle v2 but checked at load time
]
# Columns that are not leakage but must not be used as raw features.
NON_FEATURE = [
    "Date",          # only derived month / day-of-year features are allowed; never the year
                     # (stations have different coverage periods, so the year identifies stations)
]
# Soft risks: preprocessing choices that can encode station identity.
# They are controlled by switches and compared in a sensitivity analysis.
STATION_FINGERPRINT_RISKS = {
    "missing_indicator": "Whole stations lack Sunshine/Evaporation/Cloud sensors, so is-missing flags identify stations.",
    "group_impute_by_location": "Filling with Location x Month medians copies station-specific values into features.",
}

NUMERIC_FEATURES = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine", "WindGustSpeed",
    "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm",
]
WIND_DIR_FEATURES = ["WindGustDir", "WindDir9am", "WindDir3pm"]
BINARY_FEATURES = ["RainToday"]
