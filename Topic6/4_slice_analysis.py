"""Topic 6, Step 4: Slice-based evaluation — finding the model's blind spots.

Aggregate test metrics hide subgroup failures: a model can be 87% accurate
overall while being wrong half the time for a specific demographic slice.
This script automates subgroup evaluation on the Adult census income task:

  1. Slice the test set by every value of selected categorical features,
     plus feature INTERSECTIONS (e.g. age<25 AND workclass=Private).
  2. Per slice: Accuracy, F1, False Positive Rate, support.
  3. Flag "blind spots": slices whose F1 drops >20% below the overall
     F1 (with a minimum support so noise doesn't trigger flags).

FPR is reported alongside because slices can share accuracy yet differ
wildly in error TYPE — a high-FPR slice gets over-predicted as positive
(here: wrongly flagged as high earners), which is its own fairness issue.
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import scipy.sparse as sp

RANDOM_STATE = 42
MIN_SUPPORT = 50      # ignore slices smaller than this
DROP_THRESHOLD = 0.2  # blind spot = F1 more than 20% below overall

# --- Data and model (Adult census, as in Topic 5) ----------------------------
df = load_dataset("scikit-learn/adult-census-income")["train"].to_pandas()
y = (df.pop("income") == ">50K").astype(int)
CAT = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
NUM = [c for c in df.columns if c not in CAT]

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True).fit(X_train[CAT])
to_mat = lambda d: sp.hstack(
    [ohe.transform(d[CAT]), sp.csr_matrix(d[NUM].to_numpy(float))], format="csr")
clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                    tree_method="hist", eval_metric="logloss",
                    random_state=RANDOM_STATE, n_jobs=-1)
clf.fit(to_mat(X_train), y_train)
preds = clf.predict(to_mat(X_test))

# --- Slice definitions -------------------------------------------------------
# An age bucket turns the numeric age into a sliceable categorical.
test = X_test.copy()
test["age_bucket"] = pd.cut(test["age"], [0, 25, 45, 65, 200],
                            labels=["<25", "25-45", "45-65", "65+"])
SLICE_FEATURES = ["sex", "race", "age_bucket", "workclass", "marital.status"]
INTERSECTIONS = [("sex", "race"), ("age_bucket", "sex"),
                 ("age_bucket", "workclass")]


def slice_metrics(mask, name):
    """Accuracy / F1 / FPR / support for one boolean slice of the test set."""
    yt, yp = y_test[mask], preds[mask]
    if mask.sum() < MIN_SUPPORT or yt.nunique() < 2:
        return None  # too small or single-class: metrics would be noise
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    return {
        "slice": name, "n": int(mask.sum()),
        "Accuracy": accuracy_score(yt, yp),
        "F1": f1_score(yt, yp, zero_division=0),
        "FPR": fp / (fp + tn) if fp + tn else np.nan,
    }


def evaluate_slices(features, intersections):
    """All single-feature slices plus requested pairwise intersections."""
    rows = []
    for feat in features:
        for val in test[feat].dropna().unique():
            m = slice_metrics(test[feat] == val, f"{feat}={val}")
            if m:
                rows.append(m)
    for f1_, f2_ in intersections:
        for v1 in test[f1_].dropna().unique():
            for v2 in test[f2_].dropna().unique():
                mask = (test[f1_] == v1) & (test[f2_] == v2)
                m = slice_metrics(mask, f"{f1_}={v1} AND {f2_}={v2}")
                if m:
                    rows.append(m)
    return pd.DataFrame(rows)


overall_f1 = f1_score(y_test, preds)
overall_acc = accuracy_score(y_test, preds)
print(f"Overall: Accuracy={overall_acc:.4f}  F1={overall_f1:.4f}  "
      f"(test n={len(y_test)})\n")

report = evaluate_slices(SLICE_FEATURES, INTERSECTIONS)
report["F1 vs overall"] = report["F1"] / overall_f1 - 1
report["blind spot"] = report["F1 vs overall"] < -DROP_THRESHOLD
report = report.sort_values("F1 vs overall")

pd.set_option("display.max_rows", None, "display.width", 140)
fmt = report.copy()
for col in ["Accuracy", "F1", "FPR"]:
    fmt[col] = fmt[col].round(3)
fmt["F1 vs overall"] = (report["F1 vs overall"] * 100).round(1).astype(str) + "%"
fmt["blind spot"] = np.where(report["blind spot"], "<< BLIND SPOT", "")

print(f"{len(report)} slices evaluated (min support {MIN_SUPPORT}); "
      f"worst first:")
print(fmt.head(15).to_string(index=False))

n_blind = int(report["blind spot"].sum())
print(f"\n{n_blind} blind spot(s): slices with F1 >20% below overall "
      f"({overall_f1:.3f}). These subgroups need targeted data collection, "
      f"reweighting, or a dedicated model before deployment.")
