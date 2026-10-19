"""Four models that all answer the same question: P(still open at 24 months | site, category).

Baselines come first on purpose. A complicated model only earns its place by beating them on
data it never saw.

    HuffBaseline     60-year-old retail gravity model, adapted to a survival question
    AnalogBaseline   what brokers do: find similar past openings, see how they fared
    CoxModel         proportional hazards; interpretable hazard ratios
    ForestModel      random survival forest; captures interactions and nonlinearity
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from . import config
from .features import HUFF_BETAS

H = config.HORIZON_DAYS


def surv_y(df):
    return Surv.from_arrays(event=df["event"].astype(bool).values, time=df["duration_days"].values)


def km_at(time, event, t=H):
    """Kaplan-Meier survival at time t. Handles censoring: a business watched 8 months and still
    open contributes to the risk set for 8 months, then drops out without counting as a failure."""
    if len(time) == 0:
        return np.nan
    x, s = kaplan_meier_estimator(np.asarray(event, bool), np.asarray(time, float))
    return float(s[x <= t][-1]) if (x <= t).any() else 1.0


def censoring_weights(time, event, t=H):
    """Inverse-probability-of-censoring weights (IPCW).

    Of businesses that opened recently, many haven't had 24 months yet, so their outcome is unknown.
    If we just drop them, we oversample old cohorts. IPCW up-weights the known-outcome rows by
    1 / P(not yet censored), estimated with a Kaplan-Meier fit on the *censoring* times.
    Returns (known_mask, label_survived, weight).
    """
    time, event = np.asarray(time, float), np.asarray(event, int)
    known = (event == 1) & (time <= t) | (time > t)
    survived = (time > t).astype(int)
    xc, gc = kaplan_meier_estimator(~event.astype(bool), time)
    def G(u):
        idx = np.searchsorted(xc, u, side="right") - 1
        return np.where(idx >= 0, gc[np.clip(idx, 0, None)], 1.0)
    w = np.where(known, 1.0 / np.clip(G(np.minimum(time, t)), 0.05, None), 0.0)
    return known, survived, w


class HuffBaseline:
    """Huff predicts how much demand a site captures. We turn that into a survival probability by
    (1) choosing the distance-decay beta that best ranks survival on training data, then
    (2) fitting a one-variable logistic map from log(captured demand) to P(survive 24mo)."""

    def fit(self, X, df):
        y = surv_y(df)
        scores = {b: concordance_index_censored(y["event"], y["time"], -np.log1p(X[f"huff_b{b}"]))[0]
                  for b in HUFF_BETAS}
        self.beta = max(scores, key=scores.get)
        self.beta_scores = scores
        known, lab, w = censoring_weights(df.duration_days, df.event)
        z = np.log1p(X[f"huff_b{self.beta}"].values)[:, None]
        self.lr = LogisticRegression().fit(z[known], lab[known], sample_weight=w[known])
        return self

    def predict_s(self, X):
        return self.lr.predict_proba(np.log1p(X[f"huff_b{self.beta}"].values)[:, None])[:, 1]


class AnalogBaseline:
    """k nearest past openings in feature space; forecast = Kaplan-Meier of those analogs."""

    def __init__(self, features, k=50):
        self.features, self.k = features, k

    def fit(self, X, df):
        self.scaler = StandardScaler().fit(X[self.features])
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.scaler.transform(X[self.features]))
        self.t, self.e = df.duration_days.values, df.event.values
        return self

    def predict_s(self, X):
        _, idx = self.nn.kneighbors(self.scaler.transform(X[self.features]))
        return np.array([km_at(self.t[i], self.e[i]) for i in idx])

    def analogs(self, X, df_train, n=5):
        """For explaining a single forecast: which past businesses is it leaning on?"""
        _, idx = self.nn.kneighbors(self.scaler.transform(X[self.features]), n_neighbors=n)
        return [df_train.iloc[i][["name", "naics_desc", "start", "end_obs"]] for i in idx]


class _SksurvModel:
    def __init__(self, features):
        self.features = features

    def fit(self, X, df):
        self.scaler = StandardScaler().fit(X[self.features])
        self.model.fit(self.scaler.transform(X[self.features]), surv_y(df))
        return self

    def _Xs(self, X):
        return self.scaler.transform(X[self.features])

    def predict_s(self, X, t=H):
        fns = self.model.predict_survival_function(self._Xs(X))
        return np.array([fn(t) if t <= fn.x[-1] else fn.y[-1] for fn in fns])

    def risk(self, X):
        return self.model.predict(self._Xs(X))


class CoxModel(_SksurvModel):
    def __init__(self, features, alpha=0.1):
        super().__init__(features)
        self.model = CoxPHSurvivalAnalysis(alpha=alpha)   # small ridge penalty for stability

    def hazard_ratios(self):
        """exp(coef) per 1 SD of each feature. <1 protects, >1 raises closure risk."""
        return pd.Series(np.exp(self.model.coef_), index=self.features).sort_values()


class ForestModel(_SksurvModel):
    def __init__(self, features, n_estimators=300):
        super().__init__(features)
        self.model = RandomSurvivalForest(n_estimators=n_estimators, min_samples_leaf=25,
                                          max_features="sqrt", n_jobs=-1, random_state=config.RANDOM_SEED)


    def predict_band(self, X, t=H, lo=10, hi=90):
        """Spread of the forest's individual trees: a cheap read on model (epistemic) uncertainty.
        Wide band = the trees disagree, usually because few similar sites exist in training data."""
        Xs = self._Xs(X)
        per_tree = []
        for est in self.model.estimators_:
            fns = est.predict_survival_function(Xs)
            per_tree.append([fn(t) if t <= fn.x[-1] else fn.y[-1] for fn in fns])
        per_tree = np.array(per_tree)
        return np.percentile(per_tree, lo, axis=0), np.percentile(per_tree, hi, axis=0)


class ConformalSet:
    """Two kinds of uncertainty, kept separate on purpose:
      - the tree band above says how sure the model is about the PROBABILITY;
      - this conformal set says which OUTCOMES (survive / close) are plausible, with a guarantee.

    Split conformal: on held-out calibration rows, score = |observed outcome - p|. The IPCW-weighted
    (1-alpha) quantile q of those scores sets the threshold. An outcome y is in the set if |y-p|<=q.
    Guarantee: for future sites drawn like the calibration sites, the realized outcome lands in the
    set about (1-alpha) of the time. If both outcomes are in the set, the honest answer is
    'too close to call'. The guarantee is marginal (on average), not per-site."""

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def fit(self, p_cal, df_cal):
        known, lab, w = censoring_weights(df_cal.duration_days, df_cal.event)
        s, w = np.abs(lab[known] - p_cal[known]), w[known]
        order = np.argsort(s)
        cw = np.cumsum(w[order]) / w.sum()
        self.q = float(s[order][np.searchsorted(cw, 1 - self.alpha)]) if len(s) else 0.5
        return self

    def predict_set(self, p):
        p = np.asarray(p)
        surv, close = np.abs(1 - p) <= self.q, np.abs(p) <= self.q
        return np.where(surv & close, "too close to call", np.where(surv, "likely survives",
                        np.where(close, "likely closes", "neither (out of distribution)")))

    def coverage(self, p, df):
        """Empirical check of the guarantee on a test set (known-outcome rows, IPCW-weighted)."""
        known, lab, w = censoring_weights(df.duration_days, df.event)
        p = np.asarray(p)
        hit = np.abs(lab - p) <= self.q
        return float((hit[known] * w[known]).sum() / w[known].sum())
