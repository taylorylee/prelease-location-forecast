"""Generate a fake registry in the exact DataSF export schema, with KNOWN effects planted in it.

This is a 'planted truth' test. If the pipeline can't recover effects we put there on purpose,
it can't be trusted on real data. Planted truth (hazard multipliers per 1 SD):
    activity_spread  -> 0.75  (all-day activity protects, the D'Silva finding)
    diversity        -> 0.85  (mixed-use protects)
    same-category competition -> 1.25
None of these numbers describe the real city. They exist only to test the code.
"""
import numpy as np
import pandas as pd
import h3
from . import config

NAICS_POOL = ["445110", "448140", "453220", "722511", "722513", "722515", "812111",
              "812112", "713940", "541110", "531210", "621111", "238220", "511210"]


def make(n=12000, seed=config.RANDOM_SEED, snapshot="2026-09-30"):
    rng = np.random.default_rng(seed)
    snapshot = pd.Timestamp(snapshot)
    # Corridors: dense clusters with their own "character".
    centers = np.array([[37.7599, -122.4148], [37.7880, -122.4075], [37.7700, -122.4470],
                        [37.7984, -122.4369], [37.7420, -122.4200], [37.7810, -122.4640]])
    c_idx = rng.integers(0, len(centers), n)
    lat = centers[c_idx, 0] + rng.normal(0, 0.006, n)
    lon = centers[c_idx, 1] + rng.normal(0, 0.007, n)
    start = pd.to_datetime("2012-01-01") + pd.to_timedelta(rng.integers(0, 365 * 14, n), unit="D")
    naics = rng.choice(NAICS_POOL, n, p=None)

    # Latent hex-level traits that the feature pipeline must rediscover from the data.
    cells = [h3.latlng_to_cell(a, b, config.H3_RES) for a, b in zip(lat, lon)]
    uniq = sorted(set(cells))
    spread = dict(zip(uniq, rng.beta(4, 2, len(uniq))))
    s = np.array([spread[c] for c in cells])

    # Diversity and competition are emergent from the mix, so approximate them for the hazard.
    df_tmp = pd.DataFrame({"cell": cells, "n2": [x[:3] for x in naics]})
    div = df_tmp.groupby("cell")["n2"].nunique().reindex(cells).values
    comp = df_tmp.groupby(["cell", "n2"])["n2"].transform("count").values
    z = lambda x: (x - x.mean()) / (x.std() + 1e-9)
    lin = np.log(0.75) * z(s) + np.log(0.85) * z(div) + np.log(1.25) * z(comp)
    base_rate = 1 / (365 * 4.5)                       # median life ~3 years
    t = rng.exponential(1 / (base_rate * np.exp(lin)))
    end = start + pd.to_timedelta(t.astype(int), unit="D")
    end = pd.Series(end).where(end <= snapshot)   # closures after the export date aren't recorded yet

    df = pd.DataFrame({
        "Location Id": [f"SYN{i:06d}" for i in range(n)],
        "DBA Name": [f"Business {i}" for i in range(n)],
        "Street Address": ["synthetic"] * n,
        "City": ["San Francisco"] * n,
        "Location Start Date": start.strftime("%m/%d/%Y"),
        "Location End Date": pd.Series(end).dt.strftime("%m/%d/%Y"),
        "NAICS Code": naics,
        "NAICS Code Description": ["synthetic"] * n,
        "Neighborhoods - Analysis Boundaries": [f"corridor_{i}" for i in c_idx],
        "Business Location": [f"POINT ({b:.6f} {a:.6f})" for a, b in zip(lat, lon)],
    })
    # Hourly foot traffic per hex: high 'spread' = flatter profile across the day.
    hours = np.arange(24)
    rows = []
    for c in uniq:
        peak = np.exp(-0.5 * ((hours - 13) / (2 + 10 * spread[c])) ** 2)
        visits = rng.poisson(200 * peak + 5)
        rows += [(c, h, v) for h, v in zip(hours, visits)]
    traffic = pd.DataFrame(rows, columns=["h3_cell", "hour", "visits"])
    return df, traffic


if __name__ == "__main__":
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    reg, traffic = make()
    reg.to_csv(config.DATA_RAW / "synthetic_registry.csv", index=False)
    traffic.to_csv(config.DATA_RAW / "synthetic_hourly_traffic.csv", index=False)
    print(f"wrote {len(reg)} synthetic businesses, {traffic.h3_cell.nunique()} hexes of traffic")
