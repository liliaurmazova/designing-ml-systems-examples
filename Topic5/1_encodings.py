"""Topic 5, Step 1: Encoding categorical features when production sees
categories training never did.

Dataset: scikit-learn/adult-census-income (Hugging Face) — predict whether
income >50K from 6 numeric + 8 categorical features.

Setup: after an 80/20 stratified split, we ARTIFICIALLY delete the rarest
20% of category values (per column) from the TRAIN split only. The test
split keeps them — simulating production traffic containing categories the
model has never seen (new countries, new job titles, ...).

Three encodings are compared on XGBoost:

1) One-Hot Encoding (handle_unknown='ignore'): one column per known
   category. Unseen category -> ALL-ZERO block, i.e. the information is
   silently discarded. Feature count grows with cardinality.

2) Target Encoding (sklearn TargetEncoder): replace each category with a
   smoothed estimate of its target mean. Two leakage defenses:
     * smoothing — a category's encoding is pulled toward the global mean
       proportionally to how few samples it has:
           enc(c) = (n_c * mean_c + smooth * global_mean) / (n_c + smooth)
       so a 3-sample category can't memorize its 3 targets;
     * cross-fitting (built into fit_transform) — each train row is
       encoded using target stats from OTHER folds, never its own row;
     * plus explicit Gaussian noise added to TRAIN encodings only, so
       trees can't split on exact encoded values that fingerprint small
       categories. Test encodings stay noise-free.
   Unseen category -> global target mean. Always 1 column per feature.

3) Feature Hashing: hash the string "column=value" into a fixed-size
   vector — no fitted vocabulary AT ALL, so unseen categories are handled
   natively (they hash like any other string). The price is collisions:
   with hash space 2^10 different categories may share a slot; 2^14
   makes collisions rare at 16x the (sparse) width.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from datasets import load_dataset
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder
from xgboost import XGBClassifier

RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# --- Load and split ----------------------------------------------------------
df = load_dataset("scikit-learn/adult-census-income")["train"].to_pandas()
y = (df.pop("income") == ">50K").astype(int)

# pandas 3.x stores text as the dedicated 'str' dtype (not 'object'),
# so detect categorical columns via is_string_dtype
CAT_COLS = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
NUM_COLS = [c for c in df.columns if c not in CAT_COLS]

X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Split: {len(X_train)} train / {len(X_test)} test rows, "
      f"{len(CAT_COLS)} categorical + {len(NUM_COLS)} numeric features")

# --- Artificially remove rare categories from TRAIN only ---------------------
# Per categorical column: rank categories by train frequency, take the
# rarest 20% of distinct values, and DROP train rows containing them.
# Test is untouched, so these categories become "unseen in training".
removed = {}
drop_mask = pd.Series(False, index=X_train.index)
for col in CAT_COLS:
    counts = X_train[col].value_counts()
    n_remove = int(len(counts) * 0.2)
    if n_remove == 0:
        continue
    rare = counts.tail(n_remove).index.tolist()
    removed[col] = rare
    drop_mask |= X_train[col].isin(rare)

X_train, y_train = X_train[~drop_mask], y_train[~drop_mask]
print(f"Removed {int(drop_mask.sum())} train rows covering "
      f"{sum(len(v) for v in removed.values())} rare categories, e.g. "
      f"native.country: {removed['native.country'][:3]} ...")

# How much of the test set is now "unseen"?
seen = {c: set(X_train[c]) for c in CAT_COLS}
unseen_mask = pd.Series(False, index=X_test.index)
for col in CAT_COLS:
    unseen_mask |= ~X_test[col].isin(seen[col])
print(f"Test rows containing >=1 unseen category: {int(unseen_mask.sum())} "
      f"of {len(X_test)} ({unseen_mask.mean():.1%})\n")


def matrix_mb(X):
    """Memory footprint of a feature matrix in MB (sparse-aware)."""
    if sp.issparse(X):
        return (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    return X.nbytes / 1e6


def train_eval(X_tr, X_te):
    """Fit the shared baseline XGBoost and score the test set."""
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method="hist", eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)
    return accuracy_score(y_test, pred), f1_score(y_test, pred)


results = {}

# --- 1) One-Hot Encoding -----------------------------------------------------
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
X_tr = sp.hstack([ohe.fit_transform(X_train[CAT_COLS]),
                  sp.csr_matrix(X_train[NUM_COLS].to_numpy(float))], format="csr")
X_te = sp.hstack([ohe.transform(X_test[CAT_COLS]),
                  sp.csr_matrix(X_test[NUM_COLS].to_numpy(float))], format="csr")

# Demonstrate unseen handling: for a test row with an unseen native.country,
# its entire one-hot block for that column is all zeros.
col_idx = CAT_COLS.index("native.country")
block_start = sum(len(c) for c in ohe.categories_[:col_idx])
block = slice(block_start, block_start + len(ohe.categories_[col_idx]))
demo_row = X_test.index.get_indexer(
    X_test.index[~X_test["native.country"].isin(seen["native.country"])][:1])[0]
demo_val = X_test.iloc[demo_row]["native.country"]
block_sum = X_te[demo_row, block].sum()
print(f"OHE unseen demo: test row with native.country='{demo_val}' -> "
      f"one-hot block sums to {block_sum:.0f} (all zeros; category discarded)\n")

acc, f1 = train_eval(X_tr, X_te)
results["One-Hot (ignore)"] = {
    "Accuracy": acc, "F1": f1, "n_features": X_tr.shape[1],
    "Memory MB": matrix_mb(X_tr),
    "Unseen categories": "all-zero block (info lost)",
}

# --- 2) Target Encoding with smoothing + noise -------------------------------
te = TargetEncoder(
    target_type="binary", smooth=20.0,
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
)
# fit_transform cross-fits: each train row's encoding comes from folds that
# exclude that row, so its own label never leaks into its feature.
X_tr_cat = te.fit_transform(X_train[CAT_COLS], y_train)
X_tr_cat += rng.normal(0.0, 0.01, X_tr_cat.shape)  # train-only noise
X_te_cat = te.transform(X_test[CAT_COLS])          # test: no noise
X_tr = np.hstack([X_tr_cat, X_train[NUM_COLS].to_numpy(float)])
X_te = np.hstack([X_te_cat, X_test[NUM_COLS].to_numpy(float)])

acc, f1 = train_eval(X_tr, X_te)
results["Target (smooth+noise)"] = {
    "Accuracy": acc, "F1": f1, "n_features": X_tr.shape[1],
    "Memory MB": matrix_mb(X_tr),
    "Unseen categories": "global target mean",
}

# --- 3) Feature Hashing at 2^10 and 2^14 -------------------------------------
def hash_features(X_df, n_features):
    """Hash 'col=value' tokens per row into a fixed-size sparse vector."""
    tokens = X_df[CAT_COLS].apply(
        lambda row: [f"{c}={v}" for c, v in row.items()], axis=1)
    hashed = FeatureHasher(n_features=n_features, input_type="string"
                           ).transform(tokens)
    return sp.hstack([hashed, sp.csr_matrix(X_df[NUM_COLS].to_numpy(float))],
                     format="csr")

for power in (10, 14):
    X_tr = hash_features(X_train, 2 ** power)
    X_te = hash_features(X_test, 2 ** power)
    acc, f1 = train_eval(X_tr, X_te)
    results[f"Hashing 2^{power}"] = {
        "Accuracy": acc, "F1": f1, "n_features": X_tr.shape[1],
        "Memory MB": matrix_mb(X_tr),
        "Unseen categories": "native (hashes like any value)",
    }

# --- Summary -----------------------------------------------------------------
summary = pd.DataFrame(results).T
summary["Accuracy"] = summary["Accuracy"].astype(float).round(4)
summary["F1"] = summary["F1"].astype(float).round(4)
summary["Memory MB"] = summary["Memory MB"].astype(float).round(2)
summary.index.name = "encoding"
print(summary.to_string())
