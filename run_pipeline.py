"""End to end: raw registry -> features -> baselines + survival models -> held-out evaluation.

    python run_pipeline.py --registry data/raw/Registered_Business_Locations.csv \
                           --traffic data/raw/hourly_traffic.csv          # traffic optional

Writes reports/results.md, reports/calibration_*.png, reports/shap_summary.png,
and models/ for pre-registration.
"""
import argparse, pickle, json
import numpy as np
import pandas as pd
from src import config, ingest, features, models, evaluate

p = argparse.ArgumentParser()
p.add_argument("--registry", default=str(config.DATA_RAW / "synthetic_registry.csv"))
p.add_argument("--traffic", default=None)
p.add_argument("--snapshot", default=None, help="date the export was downloaded (YYYY-MM-DD)")
p.add_argument("--sample", type=int, default=None, help="subsample storefronts for a quick run")
p.add_argument("--fast", action="store_true", help="fewer trees / bootstraps")
args = p.parse_args()

config.REPORTS.mkdir(exist_ok=True); config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
(config.ROOT / "models").mkdir(exist_ok=True)

# 1. Ingest -------------------------------------------------------------------------------
raw = ingest.load_registry(args.registry, args.snapshot)
context, stores = ingest.to_survival_table(raw)
if args.sample and len(stores) > args.sample:
    stores = stores.sample(args.sample, random_state=config.RANDOM_SEED)
traffic = pd.read_csv(args.traffic) if args.traffic else None
print(f"{len(context):,} located businesses as context; {len(stores):,} storefront openings modeled; "
      f"{stores.event.mean():.0%} observed closing; snapshot {context.attrs['snapshot'].date()}")

# 2. Features (point-in-time) -------------------------------------------------------------
X = features.build(context, stores, traffic)
feat_cols = [f for f in features.FEATURES if X[f].notna().any()]
if "activity_spread" not in feat_cols:
    print("No traffic file: activity_spread excluded. The D'Silva replication needs it.")
X.join(stores[["id", "start", "duration_days", "event", "naics3", "start_year"]]).to_parquet(
    config.DATA_PROCESSED / "model_table.parquet")

n_trees = 100 if args.fast else 300


def make_models():
    return {
        "Huff (baseline)": models.HuffBaseline(),
        "Analog k-NN (baseline)": models.AnalogBaseline(feat_cols),
        "Cox PH": models.CoxModel(feat_cols),
        "Random survival forest": models.ForestModel(feat_cols, n_estimators=n_trees),
    }


# 3. Evaluate under both holdouts ----------------------------------------------------------
rows, cal_tables = [], {}
splits = {"temporal": [evaluate.temporal_split(stores)],
          "spatial": [(stores.index[a], stores.index[b]) for a, b in evaluate.spatial_folds(X)]}
for split_name, fold_list in splits.items():
    preds = {k: pd.Series(np.nan, index=stores.index) for k in make_models()}
    for tr, te in fold_list:
        for name, m in make_models().items():
            m.fit(X.loc[tr], stores.loc[tr])
            preds[name].loc[te] = m.predict_s(X.loc[te])
    test_idx = np.concatenate([te for _, te in fold_list])
    for name, s in preds.items():
        s = s.loc[test_idx].values
        d = stores.loc[test_idx]
        rows.append({"holdout": split_name, "model": name, "C-index": evaluate.c_index(d, s),
                     "Brier@24mo": evaluate.brier_at_h(d, s), "n_test": len(d)})
        if split_name == "temporal":
            cal_tables[name] = evaluate.calibration_table(d, s)

results = pd.DataFrame(rows)
evaluate.plot_calibration(cal_tables, config.REPORTS / "calibration_temporal.png")
print(results.round(3).to_string(index=False))

# 4. Final fit on older cohorts, conformal calibration on the most recent training years ----
tr_all, te_all = evaluate.temporal_split(stores)
cal_years = stores.loc[tr_all].start_year.max() - 1
fit_idx = tr_all[stores.loc[tr_all].start_year < cal_years]
cal_idx = tr_all[stores.loc[tr_all].start_year >= cal_years]
rsf = models.ForestModel(feat_cols, n_estimators=n_trees).fit(X.loc[fit_idx], stores.loc[fit_idx])
conf = models.ConformalSet(alpha=0.2).fit(rsf.predict_s(X.loc[cal_idx]), stores.loc[cal_idx])
cov = conf.coverage(rsf.predict_s(X.loc[te_all]), stores.loc[te_all])

# 5. Replication test: do D'Silva et al.'s two findings hold here? -------------------------
fit_cox = lambda Xs, ds: models.CoxModel(feat_cols).fit(Xs, ds)
hr = evaluate.bootstrap_hazard_ratios(fit_cox, X[feat_cols + ["spatial_block"]], stores,
                                      n=30 if args.fast else 100)


def verdict(f):
    if f not in hr.index:
        return "not testable (feature missing)"
    r = hr.loc[f]
    if r.ci_high < 1: return f"REPLICATES (HR {r.hr_median:.2f}, 95% CI {r.ci_low:.2f}-{r.ci_high:.2f})"
    if r.ci_low > 1:  return f"OPPOSITE (HR {r.hr_median:.2f}, 95% CI {r.ci_low:.2f}-{r.ci_high:.2f})"
    return f"inconclusive (HR {r.hr_median:.2f}, 95% CI {r.ci_low:.2f}-{r.ci_high:.2f})"


# 6. SHAP: which features drive the forest's forecasts? -----------------------------------
try:
    import shap, matplotlib.pyplot as plt
    bg = X.loc[fit_idx, feat_cols].sample(min(100, len(fit_idx)), random_state=0)
    ex = X.loc[te_all, feat_cols].sample(min(200, len(te_all)), random_state=0)
    f = lambda A: rsf.predict_s(pd.DataFrame(A, columns=feat_cols))
    sv = shap.KernelExplainer(f, shap.kmeans(bg, 10)).shap_values(ex, nsamples=100, silent=True)
    shap.summary_plot(sv, ex, show=False, plot_size=(7, 4))
    plt.title("What moves P(open at 24 months)"); plt.tight_layout()
    plt.savefig(config.REPORTS / "shap_summary.png", dpi=150); plt.close()
    shap_note = "reports/shap_summary.png"
except Exception as e:
    shap_note = f"skipped ({e})"

with open(config.ROOT / "models" / "final.pkl", "wb") as fh:
    pickle.dump({"rsf": rsf, "conformal": conf, "features": feat_cols,
                 "snapshot": str(context.attrs["snapshot"].date())}, fh)

# 7. Report -------------------------------------------------------------------------------
cox_full = fit_cox(X.loc[tr_all], stores.loc[tr_all])
report = f"""# Results

Data: `{args.registry}` · snapshot {context.attrs['snapshot'].date()} · {len(stores):,} storefront openings ·
{stores.event.mean():.0%} observed closing (the rest are censored, still open at snapshot).

## Held-out performance

{results.round(3).to_markdown(index=False)}

A model earns its place only if it beats both baselines on **both** holdouts.
Calibration plot (temporal holdout): `reports/calibration_temporal.png`.

## Uncertainty

80% conformal sets, calibrated on {int(cal_years)}+ openings. Empirical coverage on the
temporal test set: **{cov:.0%}** (target 80%). Threshold q = {conf.q:.2f}.

## Replication of D'Silva et al. (Ubicomp 2018) in this data

| Finding | Result |
|---|---|
| All-day activity spread lowers closure risk | {verdict('activity_spread')} |
| Business-mix diversity lowers closure risk | {verdict('diversity')} |

Hazard ratios are per 1 SD, from 100 block-bootstrap refits (fewer with --fast).

{hr.round(2).to_markdown()}

## Cox hazard ratios (full training set)

{cox_full.hazard_ratios().round(2).to_frame('HR per 1 SD').to_markdown()}

Huff distance-decay beta chosen on training data: see beta scores
{json.dumps({str(k): round(v, 3) for k, v in models.HuffBaseline().fit(X.loc[tr_all], stores.loc[tr_all]).beta_scores.items()})}

SHAP: {shap_note}
"""
(config.REPORTS / "results.md").write_text(report)
print("\nwrote reports/results.md")
