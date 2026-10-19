"""Turn the raw DataSF Registered Business Locations export into a survival table.

Each row of the output is one business-at-a-location with:
    duration_days  how long we watched it
    event          1 if we saw it close, 0 if it was still open when the data ends ("censored")

Why survival framing: a business that opened 8 months ago and is still open is NOT a success
and NOT a failure. Dropping it biases the model toward old businesses; labelling it "survived"
is wrong. Survival analysis uses the partial information honestly.
"""
import re
import pandas as pd
import numpy as np
from . import config

# DataSF has renamed columns over the years and the API and CSV exports differ.
# We normalize names, then map known aliases onto one schema.
ALIASES = {
    "start": ["location_start_date", "dba_start_date", "business_start_date"],
    "end": ["location_end_date", "dba_end_date", "business_end_date"],
    "naics": ["naic_code", "naics_code"],
    "naics_desc": ["naic_code_description", "naics_code_description"],
    "name": ["dba_name"],
    "address": ["street_address", "full_business_address"],
    "city": ["city"],
    "point": ["business_location", "location"],
    "id": ["location_id", "uniqueid", "ttxid"],
    "neighborhood": ["neighborhoods_analysis_boundaries"],
}


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def _pick(df, key):
    for c in ALIASES[key]:
        if c in df.columns:
            return df[c]
    return pd.Series([np.nan] * len(df), index=df.index)


def _parse_point(s):
    """DataSF points look like 'POINT (-122.41 37.76)' or a dict-ish string."""
    if not isinstance(s, str):
        return (np.nan, np.nan)
    nums = re.findall(r"-?\d+\.\d+", s)
    if len(nums) < 2:
        return (np.nan, np.nan)
    a, b = float(nums[0]), float(nums[1])
    lon, lat = (a, b) if a < 0 else (b, a)
    return (lat, lon)


def load_registry(path, snapshot_date=None) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [_norm(c) for c in raw.columns]

    df = pd.DataFrame({
        "id": _pick(raw, "id").astype(str),
        "name": _pick(raw, "name"),
        "address": _pick(raw, "address"),
        "city": _pick(raw, "city").astype(str).str.strip().str.lower(),
        "naics": _pick(raw, "naics").astype(str).str.extract(r"(\d+)")[0],
        "naics_desc": _pick(raw, "naics_desc"),
        "neighborhood": _pick(raw, "neighborhood"),
        "start": pd.to_datetime(_pick(raw, "start"), errors="coerce"),
        "end": pd.to_datetime(_pick(raw, "end"), errors="coerce"),
    })
    latlon = _pick(raw, "point").map(_parse_point)
    df["lat"] = [p[0] for p in latlon]
    df["lon"] = [p[1] for p in latlon]

    # The data "ends" when it was exported. Anything open then is censored at that date.
    # Without an explicit date, use the latest recorded date, but never later than today
    # (registries contain a few typo'd future dates that would otherwise stretch every duration).
    snapshot = (pd.Timestamp(snapshot_date) if snapshot_date
                else min(df[["start", "end"]].max().max(), pd.Timestamp.today().normalize()))
    df.attrs["snapshot"] = snapshot
    return df


def to_survival_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all_located, storefronts).

    all_located keeps every in-city business with coordinates; it is the *context* used to build
    neighborhood features. storefronts is the subset we actually model.
    """
    snapshot = df.attrs["snapshot"]
    d = df[(df["city"].str.contains("san francisco", na=False)) & df["lat"].between(37.70, 37.84)
           & df["lon"].between(-122.53, -122.35) & df["start"].notna()].copy()
    # Records with end before start are data-entry errors.
    d = d[~(d["end"].notna() & (d["end"] < d["start"]))]
    d["end_obs"] = d["end"].where(d["end"] <= snapshot)
    d["event"] = d["end_obs"].notna().astype(int)
    d["duration_days"] = ((d["end_obs"].fillna(snapshot)) - d["start"]).dt.days.clip(lower=1)
    d["naics2"] = d["naics"].str[:2]
    d["naics3"] = d["naics"].str[:3]
    d["start_year"] = d["start"].dt.year

    is_store = d["naics"].fillna("").map(lambda n: n.startswith(config.STOREFRONT_NAICS_PREFIXES))
    stores = d[is_store & (d["start_year"] >= config.STUDY_START_YEAR)].copy()
    stores.attrs["snapshot"] = snapshot
    d.attrs["snapshot"] = snapshot
    return d, stores
