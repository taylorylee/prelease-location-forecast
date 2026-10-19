"""Pre-register forecasts for real candidate sites BEFORE their outcomes are known.

    python preregister.py --sites data/candidate_sites.csv --registry <same registry> [--traffic ...]

candidate_sites.csv columns: site_id, address, lat, lon, naics (6-digit), category, status
    status = 'vacant' (forecast for a hypothetical opening today) or 'new_open' (opened recently)
    optional: opened_on (YYYY-MM-DD) for new_open sites, registry_id once you find it in DataSF

Outputs predictions/preregistration_<date>.csv plus its SHA-256 hash. Commit both to a public
GitHub repo immediately; the commit timestamp is your proof the forecast came first. Posting the
hash publicly (Substack footnote, social post) adds a second independent timestamp.

Score later with:  python preregister.py --score predictions/preregistration_<date>.csv --registry <newer export>
"""
import argparse, hashlib, pickle, datetime as dt
import numpy as np
import pandas as pd
from src import config, ingest, features

p = argparse.ArgumentParser()
p.add_argument("--sites")
p.add_argument("--registry", required=True)
p.add_argument("--traffic", default=None)
p.add_argument("--score", default=None, help="path to an earlier preregistration file to score")
args = p.parse_args()
config.PREDICTIONS.mkdir(exist_ok=True)

raw = ingest.load_registry(args.registry)
context, _ = ingest.to_survival_table(raw)

if args.score:
    pre = pd.read_csv(args.score, parse_dates=["forecast_date"])
    snap = context.attrs["snapshot"]
    out = []
    for r in pre.itertuples():
        months = (snap - r.forecast_date).days / 30.4
        status = "unknown (find registry_id)"
        if isinstance(getattr(r, "registry_id", None), str):
            m = context[context.id == r.registry_id]
            if len(m):
                status = "closed" if m.event.iloc[0] else "open"
        out.append({"site_id": r.site_id, "months_elapsed": round(months, 1), "status": status,
                    "p_open_24mo": r.p_open_24mo})
    s = pd.DataFrame(out)
    k = s[s.status.isin(["open", "closed"])]
    # Interim score: still-open sites are not 24-month successes yet, so only closures are final.
    # At 6 and 12 months, report the Brier score using P(open at that month) from the stored curve.
    print(s.to_string(index=False))
    if len(k):
        y = (k.status == "open").astype(int)
        print(f"\nInterim Brier vs 24mo forecast on {len(k)} resolved sites: {((y - k.p_open_24mo) ** 2).mean():.3f}"
              " (provisional until 24 months)")
    raise SystemExit

bundle = pickle.load(open(config.ROOT / "models" / "final.pkl", "rb"))
rsf, conf, feat_cols = bundle["rsf"], bundle["conformal"], bundle["features"]

sites = pd.read_csv(args.sites, dtype={"naics": str})
today = pd.Timestamp(dt.date.today())
sites["start"] = pd.to_datetime(sites.get("opened_on", pd.Series([None] * len(sites)))).fillna(today)
sites["naics3"] = sites.naics.str[:3]
traffic = pd.read_csv(args.traffic) if args.traffic else None
X = features.build(context, sites, traffic)

p_open = rsf.predict_s(X)
lo, hi = rsf.predict_band(X)
out = sites[["site_id", "address", "lat", "lon", "category", "naics", "status"]].copy()
for col in ("registry_id",):
    out[col] = sites[col] if col in sites else ""
out["forecast_date"] = today.date()
out["p_open_24mo"] = p_open.round(3)
out["p_band_10"], out["p_band_90"] = lo.round(3), hi.round(3)
for m in (6, 12):
    out[f"p_open_{m}mo"] = rsf.predict_s(X, t=int(m * 30.4)).round(3)
out["conformal_80"] = conf.predict_set(p_open)
out["model_snapshot"] = bundle["snapshot"]
out = out.join(X[feat_cols].round(3))

path = config.PREDICTIONS / f"preregistration_{today.date()}.csv"
out.to_csv(path, index=False)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(path.with_suffix(".sha256")).write_text(f"{digest}  {path.name}\n")
print(out[["site_id", "category", "p_open_24mo", "p_band_10", "p_band_90", "conformal_80"]].to_string(index=False))
print(f"\nSHA-256: {digest}\nNow: git add predictions && git commit -m 'Pre-register {len(out)} forecasts' && git push")
