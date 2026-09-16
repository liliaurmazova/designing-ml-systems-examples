"""Topic 6, Step 2: Churn prediction pipeline — CODE REVIEW EXERCISE.

This pipeline reports an excellent cross-validated ROC-AUC. It also
contains THREE deliberate data leakage flaws. Review the code and find
all three before reading any solution.

Scenario: subscription churn. One row per customer, ordered by signup
date. Features come from the product database and the support system.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(7)
N = 12_000

# --- Assemble the customer table (ordered by signup date) --------------------
signup_date = pd.date_range("2022-01-01", periods=N, freq="90min")
month = np.arange(N) / N * 24  # months since launch

usage_hours = rng.gamma(3, 8, N) * (1 - 0.6 * month / 24)
sessions_week = rng.poisson(np.clip(9 - 0.25 * month, 2, None))
plan_price = rng.choice([9.99, 19.99, 49.99], N, p=[0.5, 0.35, 0.15])
nps_score = np.clip(rng.normal(8 - 0.15 * month, 2), 0, 10)
tickets_open = rng.poisson(0.8, N)

churn_logit = (
    -1.2 - 0.05 * usage_hours - 0.25 * sessions_week
    - 0.35 * nps_score + 0.4 * tickets_open + 0.16 * month
)
churned = rng.binomial(1, 1 / (1 + np.exp(-churn_logit)))

# Support-system extract: total minutes each customer spent on phone with
# the retention team.
retention_call_minutes = np.where(
    churned == 1,
    rng.normal(25, 8, N).clip(2, None),
    rng.normal(3, 2, N).clip(0, None) * (rng.random(N) < 0.15),
)

df = pd.DataFrame({
    "usage_hours": usage_hours,
    "sessions_week": sessions_week,
    "plan_price": plan_price,
    "nps_score": nps_score,
    "tickets_open": tickets_open,
    "retention_call_minutes": retention_call_minutes,
})
# Simulate patchy telemetry: some usage values were never recorded
df.loc[rng.random(N) < 0.08, "usage_hours"] = np.nan

y = churned
print(f"{len(df)} customers, churn rate {y.mean():.1%}, "
      f"span {signup_date.min():%Y-%m} .. {signup_date.max():%Y-%m}")

# --- Preprocess --------------------------------------------------------------
X = SimpleImputer(strategy="mean").fit_transform(df)
X = StandardScaler().fit_transform(X)

# --- Cross-validated evaluation ----------------------------------------------
model = GradientBoostingClassifier(random_state=7)
cv = KFold(n_splits=5, shuffle=True, random_state=7)
scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print(f"Cross-validated ROC-AUC: {scores.mean():.4f} "
      f"(folds: {np.round(scores, 4).tolist()})")
print("Looks production-ready... or does it? Find the three leaks.")
