"""Behavioral test suite for the credit scorer (CheckList methodology,
Ribeiro et al. 2020) — Topic 6, Step 3.

Three test families:
  INV — Invariance: perturbing an attribute that SHOULD NOT matter must
        not change predictions.
  DIR — Directional expectation: moving an attribute in a direction with
        a known effect must never move the prediction the other way.
  MFT — Minimum functionality: hand-written critical cases with an
        obviously correct answer.

These tests probe model BEHAVIOR, not accuracy: a model can score high
AUC while failing every one of them.
"""

import numpy as np
import pandas as pd
import pytest

from step3_credit_model import credit_score, generate_applicants

SEED = 123
TOL = 0.01  # max tolerated probability shift for invariance


@pytest.fixture(scope="module")
def applicants():
    """A fresh sample of applicants the model was not trained on."""
    return generate_applicants(n=500, seed=SEED)


# --- INV: sensitive / irrelevant attributes ----------------------------------

@pytest.mark.parametrize("column, new_value", [
    ("gender", "female"),
    ("gender", "nonbinary"),
    ("race", "group_c"),
])
def test_inv_sensitive_attributes(applicants, column, new_value):
    """Flipping gender/race must change nothing at all."""
    base = credit_score(applicants)
    flipped = applicants.copy()
    flipped[column] = new_value
    assert np.allclose(base, credit_score(flipped), atol=1e-12), (
        f"Prediction depends on {column}")


def test_inv_zip_code(applicants):
    """Moving every applicant to a different zip tier should not move
    scores: zip code is not a financial attribute, and it is a
    well-documented proxy for protected characteristics."""
    base = credit_score(applicants)
    moved = applicants.copy()
    moved["zip_risk_tier"] = (moved["zip_risk_tier"] + 1) % 3
    shift = np.abs(base - credit_score(moved)).max()
    assert shift < TOL, (
        f"Zip change shifts approval probability by up to {shift:.3f} "
        f"(> {TOL}) — the model discriminates by geography")


# --- DIR: directional expectation --------------------------------------------

@pytest.mark.parametrize("raise_pct", [0.1, 0.5, 1.0])
def test_dir_income_monotonic(applicants, raise_pct):
    """A raise, everything else equal, must never LOWER the score."""
    base = credit_score(applicants)
    richer = applicants.copy()
    richer["income_k"] = richer["income_k"] * (1 + raise_pct)
    drop = (base - credit_score(richer)).max()
    assert drop <= 1e-9, (
        f"+{raise_pct:.0%} income DECREASED some score by {drop:.4f}")


# --- MFT: minimum functionality on critical cases ----------------------------

AVERAGE_PROFILE = dict(income_k=60.0, debt_to_income=0.28, history_years=15.0,
                       late_payments=1, bankruptcy_active=0, zip_risk_tier=0,
                       gender="male", race="group_a")


def make_applicant(**overrides):
    return pd.DataFrame([{**AVERAGE_PROFILE, **overrides}])


def test_mft_average_profile_approved():
    """Sanity anchor: a plainly average applicant should be approved."""
    assert credit_score(make_applicant())[0] >= 0.5


def test_mft_zero_income_denied():
    """No income -> no loan, even with an otherwise clean profile."""
    assert credit_score(make_applicant(income_k=0.0))[0] < 0.5


def test_mft_active_bankruptcy_denied():
    """Active bankruptcy must dominate an otherwise strong profile."""
    strong = make_applicant(income_k=90.0, debt_to_income=0.1,
                            history_years=20.0, late_payments=0,
                            bankruptcy_active=1)
    assert credit_score(strong)[0] < 0.5
