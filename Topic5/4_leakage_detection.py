"""Topic 5, Step 4: Demonstrating data leakage from preprocessing-before-split.

Dataset: David-Egea/Creditcard-fraud-detection (the classic ULB credit-card
set: 284,807 transactions, features V1-V28 + Time + Amount, ~0.17% fraud).

Two pipelines, identical model, same final test rows:

  FLAWED : StandardScaler.fit + SMOTE on the WHOLE dataset, THEN split.
           Two leaks stack up:
             1. Scaler statistics include test rows (mild leak).
             2. SMOTE synthesizes minority points BETWEEN real neighbors
                BEFORE the split — so the test set fills with synthetic
                near-copies of minority samples the model trained on.
                Evaluation degenerates into "recognize your own clones".

  CORRECT: split FIRST on the raw data; fit scaler on X_train only;
           SMOTE X_train only. The test set stays real and untouched,
           at the true fraud rate.

A duplicate check runs first: the raw data contains exact duplicate rows,
which are themselves a leakage source under ANY random split (the same
transaction can land in both train and test), so they are removed.
"""

import numpy as np
from datasets import load_dataset
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

# --- Load + duplicate check --------------------------------------------------
df = load_dataset("David-Egea/Creditcard-fraud-detection")["train"].to_pandas()
n_before = len(df)
df = df.drop_duplicates()
print(f"Duplicate check: {n_before} rows -> {len(df)} after drop_duplicates() "
      f"({n_before - len(df)} exact duplicates removed — duplicates straddle "
      f"any random split and leak by themselves)")
y = df.pop("Class").to_numpy()
X = df.to_numpy(float)
print(f"Fraud rate: {y.mean():.4%} ({int(y.sum())} frauds)\n")


def fit_rf(X_tr, y_tr):
    clf = RandomForestClassifier(n_estimators=60, n_jobs=-1,
                                 random_state=RANDOM_STATE)
    return clf.fit(X_tr, y_tr)


def report(title, clf, X_te, y_te):
    probs = clf.predict_proba(X_te)[:, 1]
    ap = average_precision_score(y_te, probs)
    cm = confusion_matrix(y_te, clf.predict(X_te))
    print(f"{title}\n  PR-AUC (average precision): {ap:.4f}")
    print(f"  Confusion matrix [[TN FP],[FN TP]]:\n  {cm[0]}\n  {cm[1]}\n")
    return ap


# --- 1) FLAWED pipeline: scale + SMOTE BEFORE the split ----------------------
scaler_all = StandardScaler().fit(X)                     # leak 1: sees test rows
X_res, y_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
    scaler_all.transform(X), y)                          # leak 2: SMOTE over all
Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(
    X_res, y_res, test_size=0.2, stratify=y_res, random_state=RANDOM_STATE)
flawed_clf = fit_rf(Xf_tr, yf_tr)
ap_flawed = report("FLAWED (scale+SMOTE before split), its own test set:",
                   flawed_clf, Xf_te, yf_te)

# --- 2) CORRECT pipeline: split FIRST ----------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(X_tr)                      # train-only stats
X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)
X_tr_res, y_tr_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
    X_tr_s, y_tr)                                        # train-only SMOTE
correct_clf = fit_rf(X_tr_res, y_tr_res)
ap_correct = report("CORRECT (split first, train-only scale+SMOTE), clean test:",
                    correct_clf, X_te_s, y_te)

# --- 3) The tell: score the FLAWED model on the truly clean test set ---------
ap_reality = report("FLAWED model re-evaluated on the CLEAN test set:",
                    flawed_clf, scaler_all.transform(X_te), y_te)

# --- Explanation -------------------------------------------------------------
print("=" * 72)
print(f"""WHY THE FLAWED NUMBERS ARE A LIE
The flawed pipeline reports PR-AUC {ap_flawed:.4f}; the correct one {ap_correct:.4f}.

1. The flawed "test set" is ~50% synthetic fraud. SMOTE ran before the
   split, so every synthetic test point lies on a line segment between two
   real frauds — and those parents (or their other interpolants) were in
   training. The model is graded on recognizing near-duplicates of its
   own training data, hence a perfect score.
2. Its test class balance is 50:50 instead of the real {y.mean():.2%} fraud
   rate; PR-AUC rises mechanically as the positive class gets commoner —
   a random classifier's AP equals the positive rate (0.5 vs 0.0017).
3. StandardScaler fitted on the full data folded test-set statistics into
   every feature (minor here, but the same mistake corrupts more when
   distributions drift).

The third evaluation is the smoking gun: the flawed model scores
{ap_reality:.4f} with ZERO missed frauds even on the clean test set — not
because it generalizes, but because those exact test frauds (and their
synthetic clones) were inside its pre-split training pool. It has seen the
answers. Once contaminated, NO evaluation of that model can be trusted;
the only honest number on this screen is the correct pipeline's
{ap_correct:.4f}, obtained by splitting FIRST and fitting scaler and SMOTE
on the training fold alone.""")
