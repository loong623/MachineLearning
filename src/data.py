"""Loading the raw data and attaching the climate-zone label, with integrity and leakage checks."""
import hashlib

import pandas as pd

from config import LEAKAGE_DROP, LOCATION_ZONE_CSV, RAW_CSV, RAW_SHA256, TARGET


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labeled(check_hash=True):
    """Return (df, groups).

    df     : raw weather rows of stations that have a zone label, leakage columns removed,
             plus the target column `Zone`.
    groups : the Location of every row, kept OUTSIDE the feature table. Used only for
             grouped splits (GroupKFold) and error analysis, never as a model input.
    """
    if check_hash:
        assert sha256(RAW_CSV) == RAW_SHA256, "weatherAUS.csv differs from the version we documented"
    raw = pd.read_csv(RAW_CSV, parse_dates=["Date"])
    present = [c for c in LEAKAGE_DROP if c in raw.columns]
    print(f"[leakage audit] columns present and removed: {present}; "
          f"not present: {[c for c in LEAKAGE_DROP if c not in raw.columns]}")

    zones = pd.read_csv(LOCATION_ZONE_CSV)[["Location", TARGET]]
    df = raw.merge(zones, on="Location", how="left", validate="many_to_one")
    n_unlabeled = df[TARGET].isna().sum()
    print(f"[labels] dropped {n_unlabeled} rows without a zone "
          f"({sorted(df.loc[df[TARGET].isna(), 'Location'].unique())})")
    df = df[df[TARGET].notna()].reset_index(drop=True)

    groups = df["Location"].copy()
    df = df.drop(columns=present)
    assert not set(LEAKAGE_DROP) & set(df.columns)
    return df, groups
