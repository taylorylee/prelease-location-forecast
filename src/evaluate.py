"""How we know a model is right, without trusting the person who built it.

Three metrics, each answering a different question:
    C-index      Ranking. Of two businesses, does the model give the earlier-closing one higher risk?
                 0.5 = coin flip, 1.0 = perfect. Uses censored rows correctly.
    Brier @24mo  Accuracy of the probability itself (squared error), IPCW-weighted. Lower is better.
                 A model can rank well but be badly miscalibrated; Brier catches that.
    Calibration  When the model says 70%, do ~70% actually survive? Checked in bins against
                 Kaplan-Meier estimates. This is the one a landlord or tenant actually cares about.

Two kinds of holdout, because the model will be used on new places AND in the future:
    spatial      hold out whole H3 res-6 blocks (no peeking at neighbors of test sites)
    temporal     train on openings before TEMPORAL_TEST_FROM_YEAR, test on later ones
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sksurv.metrics import concordance_index_censored
from . import config
from .models import km_at, censoring_weights

H = config.HORIZON_DAYS


def c_index(df, s_pred):
    return concordance_index_censored(df.event.astype(bool).values, df.duration_days.values, -np.asarray(s_pred))[0]


def brier_at_h(df, s_pred):
    known, lab, w = censoring_weights(df.duration_days, df.event)
    return float((w[known] * (lab[known] - np.asarray(s_pred)[known]) ** 2).sum() / w[known].sum())


def calibration_table(df, s_pred, bins=5):
    d = pd.DataFrame({"p": s_pred, "t": df.duration_days.values, "e": df.event.values})
    d["bin"] = pd.qcut(d.p, bins, duplicates="drop")
    return d.groupby("bin", observed=True).apply(
        lambda g: pd.Series({"predicted": g.p.mean(), "observed_km": km_at(g.t, g.e), "n": len(g)}),
        include_groups=False).reset_index(drop=True)


def spatial_folds(feats, n_splits=5):
    return GroupKFold(n_splits=n_splits).split(feats, groups=feats["spatial_block"])


def temporal_split(df):
    train = df.index[df.start_year < config.TEMPORAL_TEST_FROM_YEAR]
    test = df.index[df.start_year >= config.TEMPORAL_TEST_FROM_YEAR]
    return train, test


def plot_calibration(tables: dict, path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ls="--", c="grey", lw=1, label="perfect")
    for name, t in tables.items():
        ax.plot(t.predicted, t.observed_km, marker="o", label=name)
    ax.set_xlabel("Predicted P(open at 24 months)")
    ax.set_ylabel("Observed (Kaplan-Meier)")
    ax.set_title("Calibration on held-out openings")
    ax.legend(frameon=False, fontsize=8)
    lo = min(min(t.predicted.min(), t.observed_km.min()) for t in tables.values()) - 0.05
    ax.set_xlim(max(lo, 0), 1); ax.set_ylim(max(lo, 0), 1)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def bootstrap_hazard_ratios(fit_cox, X, df, n=100, seed=config.RANDOM_SEED):
    """Refit Cox on resampled data to get 95% intervals on each hazard ratio.
    Resample whole spatial blocks, not rows: neighbors share conditions, so row-level resampling
    would pretend we have more independent evidence than we do."""
    rng = np.random.default_rng(seed)
    blocks = X["spatial_block"].unique()
    out = []
    for _ in range(n):
        pick = rng.choice(blocks, len(blocks), replace=True)
        idx = np.concatenate([np.flatnonzero(X["spatial_block"].values == b) for b in pick])
        m = fit_cox(X.iloc[idx], df.iloc[idx])
        out.append(m.hazard_ratios())
    hr = pd.DataFrame(out)
    return pd.DataFrame({"hr_median": hr.median(), "ci_low": hr.quantile(0.025), "ci_high": hr.quantile(0.975)})
