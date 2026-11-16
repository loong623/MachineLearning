"""Build the reproducible Location -> Köppen code -> Zone table.

Sources
-------
1. Station coordinates: Bureau of Meteorology station list, product IDCJMC0014
   (ftp://ftp.bom.gov.au/anon2/home/ncc/metadata/sitelists/stations.zip), downloaded 2026-11-09.
   The Location -> BoM site number match is hand-curated in data/external/location_station.csv.
2. Köppen-Geiger code: kgcpy 1.1.8 (Python port of the CRAN `kgc` package, Bryant et al.),
   which looks up the 100 arc-second (~3 km) Köppen-Geiger map of Rubel et al. (2017),
   reference period 1986-2010.

Usage:  python src/build_zone_labels.py
Output: data/processed/location_zone.csv
"""
import io
import zipfile

import pandas as pd
from kgcpy import lookupCZ, nearbyCZ

from config import BOM_STATIONS_ZIP, KOPPEN_TO_ZONE, LOCATION_STATION_CSV, LOCATION_ZONE_CSV


def read_bom_stations(zip_path):
    """Parse the fixed-width BoM station list into a DataFrame (site, name, lat, lon)."""
    with zipfile.ZipFile(zip_path) as z:
        text = z.read("stations.txt").decode("latin-1")
    rows = []
    for line in text.splitlines()[4:]:              # first 4 lines are a header
        if len(line) < 90 or not line[:7].strip().isdigit():
            continue
        rows.append({"bom_site": line[0:7].strip(), "bom_name_list": line[14:55].strip(),
                     "lat": float(line[71:80]), "lon": float(line[80:90])})
    return pd.DataFrame(rows)


def main():
    loc = pd.read_csv(LOCATION_STATION_CSV, dtype=str)
    st = read_bom_stations(BOM_STATIONS_ZIP)
    df = loc.merge(st, on="bom_site", how="left", validate="one_to_one")
    assert df["lat"].notna().all(), "unmatched BoM site number"
    assert (df["bom_name"] == df["bom_name_list"]).all(), "station name mismatch"

    df["koppen"] = [lookupCZ(a, b) for a, b in zip(df["lat"], df["lon"])]
    near = [nearbyCZ(a, b, 1) for a, b in zip(df["lat"], df["lon"])]   # 3x3 pixel window
    df["neighbour_match"] = [round(n[1], 2) for n in near]
    df["neighbour_codes"] = [";".join(sorted(set(n[2]))) for n in near]
    df["Zone"] = df["koppen"].map(KOPPEN_TO_ZONE)                    # NaN = no label (e.g. "Ocean")
    # A station is a boundary case if any neighbouring *land* pixel maps to a different zone
    # ("Ocean" pixels next to coastal stations are ignored).
    df["zone_boundary"] = [
        any(KOPPEN_TO_ZONE.get(c, c) != z for c in codes.split(";") if c and c != "Ocean")
        if isinstance(z, str) else True
        for codes, z in zip(df["neighbour_codes"], df["Zone"])
    ]
    cols = ["Location", "bom_site", "bom_name", "lat", "lon", "koppen",
            "neighbour_match", "neighbour_codes", "zone_boundary", "Zone"]
    LOCATION_ZONE_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(LOCATION_ZONE_CSV, index=False)
    print(df[cols].to_string())
    print(df["Zone"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
