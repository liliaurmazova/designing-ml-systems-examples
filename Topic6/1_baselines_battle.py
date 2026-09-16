"""Topic 6, Step 1: The baseline hierarchy — earn complexity step by step.

Before deploying a complex model, climb the ladder and measure every rung:

  1. Zero-rule     — predict the majority class (or random). Costs nothing,
                     defines the floor every other model must beat.
  2. Heuristic     — one line of business logic a domain expert would write.
                     Often embarrassingly competitive; if ML can't beat it,
                     ship the if-statement.
  3. Simple ML     — LogisticRegression on lightly scaled features.
  4. Complex ML    — gradient-boosted trees (XGBoost).

The gap between rungs tells you where the value is: a big jump from 2 to 3
justifies ML at all; a small jump from 3 to 4 questions the complex model's
serving/maintenance cost (Huyen's "start simple" rule).

Dataset: synthetic customer churn with interpretable drivers (inactivity,
support friction, contract type), so the heuristic has real signal to use.
"""

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)
N = 20_000

# --- Synthetic churn data with named, interpretable drivers ------------------
df = pd.DataFrame({
    "tenure_months": rng.exponential(24, N).clip(1, 72),
    "monthly_charges": rng.normal(70, 25, N).clip(20, 160),
    "days_since_last_login": rng.gamma(2.0, 12.0, N).clip(0, 120),
    "support_tickets_90d": rng.poisson(1.2, N),
    "contract_monthly": rng.binomial(1, 0.55, N),
})
logit = (
    -3.0
    + 0.045 * df["days_since_last_login"]
    + 0.35 * df["support_tickets_90d"]
    + 1.1 * df["contract_monthly"]
    - 0.04 * df["tenure_months"]
    + 0.008 * (df["monthly_charges"] - 70)
)
y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
print(f"Churn rate: {y.mean():.1%} of {N} customers")

FEATURES = df.columns.tolist()
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

results = {}


def score(name, y_pred, train_seconds):
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="binary", zero_division=0)
    results[name] = {"Precision": p, "Recall": r, "F1": f1,
                     "Train time (s)": train_seconds}


# --- 1. Zero-rule: majority class -------------------------------------------
t0 = time.perf_counter()
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
score("1. Zero-rule (majority)", dummy.predict(X_test), time.perf_counter() - t0)

# --- 2. Rule-based heuristic: what a retention manager would write -----------
# "A churner is an inactive monthly-contract customer or one drowning in
# support tickets." No training at all — the 'fit' is domain knowledge.
t0 = time.perf_counter()
heuristic = (
    ((X_test["days_since_last_login"] > 30) & (X_test["contract_monthly"] == 1))
    | (X_test["support_tickets_90d"] >= 4)
).astype(int)
score("2. Heuristic (inactivity rule)", heuristic, time.perf_counter() - t0)

def best_train_threshold(model, X_tr):
    """Pick the F1-maximizing threshold on TRAIN predictions (never test).

    With a 20% positive rate, the default 0.5 cut makes probabilistic
    models overly conservative; the heuristic has no threshold, so a fair
    ladder lets each ML model pick its operating point — using only
    training data.
    """
    probs = model.predict_proba(X_tr)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [precision_recall_fscore_support(
        y_train, probs >= t, average="binary", zero_division=0)[2]
        for t in grid]
    return grid[int(np.argmax(f1s))]


# --- 3. Simple ML: LogisticRegression, scaling only --------------------------
t0 = time.perf_counter()
scaler = StandardScaler().fit(X_train)
logreg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
logreg.fit(scaler.transform(X_train), y_train)
thr = best_train_threshold(logreg, scaler.transform(X_train))
elapsed = time.perf_counter() - t0
score("3. LogisticRegression",
      (logreg.predict_proba(scaler.transform(X_test))[:, 1] >= thr).astype(int),
      elapsed)

# --- 4. Complex ML: XGBoost --------------------------------------------------
t0 = time.perf_counter()
xgb = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.08,
                    tree_method="hist", eval_metric="logloss",
                    random_state=RANDOM_STATE, n_jobs=-1)
xgb.fit(X_train, y_train)
thr = best_train_threshold(xgb, X_train)
elapsed = time.perf_counter() - t0
score("4. XGBoost",
      (xgb.predict_proba(X_test)[:, 1] >= thr).astype(int), elapsed)

# --- Summary -----------------------------------------------------------------
table = pd.DataFrame(results).T.round(4)
table["Train time (s)"] = table["Train time (s)"].round(3)
table.index.name = "baseline"
print("\n" + table.to_string())

f1s = table["F1"]
print(f"\nRung-to-rung F1 gains: heuristic {f1s.iloc[1] - f1s.iloc[0]:+.3f}, "
      f"simple ML {f1s.iloc[2] - f1s.iloc[1]:+.3f}, "
      f"complex ML {f1s.iloc[3] - f1s.iloc[2]:+.3f}")
