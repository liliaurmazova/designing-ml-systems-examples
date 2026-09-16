"""Step 3: SMOTE oversampling done correctly — Chapter 4 "Training Data",
Chip Huyen, "Designing Machine Learning Systems".

SMOTE (Synthetic Minority Oversampling TEchnique) balances the training set
by generating synthetic minority samples: for each class-1 sample it picks
one of its k nearest minority neighbors and creates a new point on the line
segment between them.

CRITICAL — resample ONLY the training set, AFTER the split:
  * If SMOTE ran before the split, synthetic points interpolated from a
    given real sample could land in train while that sample lands in test
    (or vice versa). The model would then be evaluated on near-copies of
    data it trained on — data leakage, inflating every test metric.
  * The test set must stay at the real-world 99:1 ratio anyway: it exists
    to estimate production performance, and production data is imbalanced.
"""

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Same dataset and stratified split as steps 1-2
from Eval.DesigningMachineLearningSystems.Topic4.step1_imbalanced_split import (
    X_test,
    X_train,
    print_distribution,
    y_test,
    y_train,
)

# --- SMOTE: applied to the TRAINING set only ---------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train) # type: ignore

print("Training set before and after SMOTE (test set untouched):")
print_distribution("Before", y_train)
print_distribution("After", y_train_res)
print_distribution("Test", y_test)  # still 99:1 — this is the point

# --- Two models: baseline (imbalanced train) vs SMOTE (balanced train) -------
# Baseline re-trained here with the same params/seed as step 2, so its
# numbers match step 2 exactly and the comparison is self-contained.
baseline = RandomForestClassifier(random_state=42)
baseline.fit(X_train, y_train)

smote_clf = RandomForestClassifier(random_state=42)
smote_clf.fit(X_train_res, y_train_res)

# Both models are evaluated on the ORIGINAL, non-resampled test set.
results = {}
for name, model in [("baseline", baseline), ("SMOTE", smote_clf)]:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    results[name] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "missed (FN)": int(cm[1, 0]),
        "false alarms (FP)": int(cm[0, 1]),
    }

print(f"\n{'metric':<18}{'baseline':>10}{'SMOTE':>10}")
for metric in results["baseline"]:
    b, s = results["baseline"][metric], results["SMOTE"][metric]
    fmt = "d" if isinstance(b, int) else ".4f"
    print(f"{metric:<18}{b:>10{fmt}}{s:>10{fmt}}")

# The trade-off to expect: SMOTE raises recall (fewer missed class-1 cases)
# at the cost of some precision (more false alarms) — accuracy barely moves,
# again showing it is the wrong lens for imbalanced problems.
recall_gain = results["SMOTE"]["recall"] - results["baseline"]["recall"]
print(f"\nRecall improvement over baseline: {recall_gain:+.4f} "
      f"({results['baseline']['recall']:.2f} -> {results['SMOTE']['recall']:.2f})")
