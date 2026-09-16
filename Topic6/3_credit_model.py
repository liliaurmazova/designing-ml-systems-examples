"""Topic 6, Step 3: Credit scoring model under behavioral test.

Synthetic loan-approval data. The applicant table carries sensitive
attributes (gender, race) and a zip-code region tier. The model is
trained WITHOUT gender/race — but WITH zip_risk_tier, mirroring a common
real-world setup ("we don't use protected attributes") whose weakness the
CheckList suite in test_model_behavior.py is designed to expose: zip code
is a well-documented proxy for protected attributes, and the historical
approval data embeds a small zip-tier bias.

Run the tests:  pytest test_model_behavior.py -v
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N = 20_000

# Model inputs: financial profile + zip tier. Gender/race are in the data
# but deliberately excluded from FEATURES.
FEATURES = ["income_k", "debt_to_income", "history_years",
            "late_payments", "bankruptcy_active", "zip_risk_tier"]


def generate_applicants(n=N, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "income_k": rng.lognormal(4.0, 0.5, n).clip(0, 400),      # ~60k median
        "debt_to_income": rng.beta(2, 5, n).round(3),
        "history_years": rng.uniform(0, 30, n).round(1),
        "late_payments": rng.poisson(1.0, n),
        "bankruptcy_active": rng.binomial(1, 0.04, n),
        "zip_risk_tier": rng.choice([0, 1, 2], n, p=[0.4, 0.4, 0.2]),
        "gender": rng.choice(["male", "female", "nonbinary"], n),
        "race": rng.choice(["group_a", "group_b", "group_c", "group_d"], n),
    })
    # Historical approvals: driven by finances, plus a residual zip-tier
    # bias baked into past decisions (the thing models happily re-learn).
    logit = (
        -1.0
        + 0.035 * df["income_k"]
        - 3.0 * df["debt_to_income"]
        + 0.08 * df["history_years"]
        - 0.6 * df["late_payments"]
        - 3.5 * df["bankruptcy_active"]
        - 0.5 * df["zip_risk_tier"]
    )
    df["approved"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return df


def train_model(seed=RANDOM_STATE):
    """Train the scorer; returns (model, scaler)."""
    df = generate_applicants(seed=seed)
    scaler = StandardScaler().fit(df[FEATURES])
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(scaler.transform(df[FEATURES]), df["approved"])
    return model, scaler


# Trained once at import; tests share this artifact.
MODEL, SCALER = train_model()


def credit_score(applicants: pd.DataFrame) -> np.ndarray:
    """Approval probability in [0, 1] for a DataFrame of applicants.

    Accepts the full applicant table (sensitive columns included) and
    uses only FEATURES — the public scoring API the tests exercise.
    """
    return MODEL.predict_proba(SCALER.transform(applicants[FEATURES]))[:, 1]


if __name__ == "__main__":
    df = generate_applicants(seed=7)
    print(f"Approval rate: {df.approved.mean():.1%}")
    coef = dict(zip(FEATURES, MODEL.coef_[0].round(3)))
    print("Learned coefficients (standardized):", coef)
