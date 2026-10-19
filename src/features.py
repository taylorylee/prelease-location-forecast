"""Neighborhood features, computed AS OF each business's opening date.

The single most common mistake in this kind of model is leakage: describing a 2016 opening with
2024 neighborhood data. That quietly lets the future predict the past and inflates accuracy.
Every feature below only looks at businesses that were open on the focal business's start date
(and closures that had already happened by then).
"""
import math
import numpy as np
import pandas as pd
import h3
from . import config

# Distance-decay exponents tried for the Huff baseline. Calibration picks one on training data.
HUFF_BETAS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _entropy(counts):
    counts = np.asarray([c for c in counts if c > 0], dtype=float)
    if counts.size <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))   # normalized 0..1


def _haversine_km(a, b):
    (la1, lo1), (la2, lo2) = a, b
    r = 6371.0
    p1, p2 = math.radians(la1), math.radians(la2)
    dp, dl = p2 - p1, math.radians(lo2 - lo1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def _km_matrix(a, b):
    """Pairwise haversine km between point arrays a (n,2) and b (m,2), in lat/lon degrees."""
    if len(b) == 0:
        return np.zeros((len(a), 0))
    la1, lo1 = np.radians(a[:, :1]), np.radians(a[:, 1:])
    la2, lo2 = np.radians(b[:, 0])[None, :], np.radians(b[:, 1])[None, :]
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(h))


def activity_spread_by_cell(traffic: pd.DataFrame | None) -> dict:
    """How evenly a hex's foot traffic is spread across the 24 hours (0 = one spike, 1 = flat).

    This is the operational version of D'Silva et al.'s finding that venues busy around the clock
    survive longer. Source can be Advan/Dewey panels, Gatherground sensors, or your field counts.
    """
    if traffic is None or traffic.empty:
        return {}
    return traffic.groupby("h3_cell")["visits"].apply(lambda v: _entropy(v.values)).to_dict()


def build(context: pd.DataFrame, focal: pd.DataFrame, traffic: pd.DataFrame | None = None) -> pd.DataFrame:
    """context: every located business (the neighborhood). focal: rows we want features for.

    focal must have lat, lon, start (the as-of date), naics3. It can be historic businesses (for
    training) or candidate sites (for pre-registered forecasts, with start = today).
    """
    ctx = context.copy()
    ctx["cell"] = [h3.latlng_to_cell(a, b, config.H3_RES) for a, b in zip(ctx.lat, ctx.lon)]
    ctx_start = ctx["start"].values.astype("datetime64[D]")
    ctx_end = ctx["end_obs"].fillna(pd.Timestamp("2100-01-01")).values.astype("datetime64[D]")
    ctx_n3 = ctx["naics3"].fillna("").values
    ctx_n2 = ctx["naics2"].fillna("").values
    by_cell = ctx.groupby("cell").indices

    spread = activity_spread_by_cell(traffic)
    out = []
    for row in focal.itertuples():
        cell = h3.latlng_to_cell(row.lat, row.lon, config.H3_RES)
        ring = h3.grid_disk(cell, config.WALKSHED_RING)
        idx = np.concatenate([by_cell[c] for c in ring if c in by_cell]) if any(c in by_cell for c in ring) else np.array([], int)
        t0 = np.datetime64(pd.Timestamp(row.start).date(), "D")
        own_id = getattr(row, "id", None)

        if idx.size:
            active = idx[(ctx_start[idx] < t0) & (ctx_end[idx] >= t0)]
            if own_id is not None:
                active = active[ctx["id"].values[active] != own_id]
            lookback = t0 - np.timedelta64(730, "D")
            closed_recent = idx[(ctx_end[idx] < t0) & (ctx_end[idx] >= lookback)]
        else:
            active = closed_recent = np.array([], int)

        # --- Huff gravity inputs (wider ring-2 trade area) ---
        ring2 = h3.grid_disk(cell, 2)
        idx2 = [by_cell[c] for c in ring2 if c in by_cell]
        idx2 = np.concatenate(idx2) if idx2 else np.array([], int)
        act2 = idx2[(ctx_start[idx2] < t0) & (ctx_end[idx2] >= t0)] if idx2.size else idx2
        if own_id is not None and act2.size:
            act2 = act2[ctx["id"].values[act2] != own_id]
        demand = pd.Series(ctx["cell"].values[act2]).value_counts() if act2.size else pd.Series(dtype=int)
        comps = act2[ctx_n3[act2] == row.naics3] if act2.size else act2
        huff = {}
        if len(demand):
            dxy = np.array([h3.cell_to_latlng(c) for c in demand.index])
            dem = demand.values.astype(float)
            d_focal = np.maximum(_km_matrix(dxy, np.array([[row.lat, row.lon]]))[:, 0], 0.05)
            d_comp = np.maximum(_km_matrix(dxy, np.c_[ctx["lat"].values[comps], ctx["lon"].values[comps]]), 0.05)
            for beta in HUFF_BETAS:
                f = d_focal ** -beta
                others = (d_comp ** -beta).sum(axis=1) if comps.size else 0.0
                # Huff: P(customer at j chooses focal) = A_f d^-b / sum_k A_k d^-b, with equal A.
                huff[f"huff_b{beta}"] = float((dem * f / (f + others)).sum())
        else:
            huff = {f"huff_b{b}": 0.0 for b in HUFF_BETAS}

        n_active = active.size
        same_cat = int((ctx_n3[active] == row.naics3).sum()) if n_active else 0
        mix = pd.Series(ctx_n2[active]).value_counts().values if n_active else []
        out.append({
            "cell": cell,
            "n_active_walkshed": n_active,
            "same_category_walkshed": same_cat,
            "competition_share": same_cat / n_active if n_active else 0.0,
            "diversity": _entropy(mix),
            "recent_churn": closed_recent.size / (n_active + closed_recent.size) if (n_active + closed_recent.size) else 0.0,
            "km_to_bart": min(_haversine_km((row.lat, row.lon), s) for s in config.BART_STATIONS.values()),
            "activity_spread": np.mean([spread[c] for c in ring if c in spread]) if spread else np.nan,
            "is_food": int(str(row.naics3).startswith("722")),
            "is_covid_cohort": int(pd.Timestamp(row.start).year in (2020, 2021)),
            **huff,
        })
    feats = pd.DataFrame(out, index=focal.index)
    feats["spatial_block"] = [h3.cell_to_parent(c, config.SPATIAL_BLOCK_RES) for c in feats["cell"]]
    return feats


FEATURES = ["n_active_walkshed", "same_category_walkshed", "competition_share", "diversity",
            "recent_churn", "km_to_bart", "activity_spread", "is_food", "is_covid_cohort"]
