"""Step 2: The accuracy trap — Chapter 4 "Training Data",
Chip Huyen, "Designing Machine Learning Systems".

Trains a baseline RandomForest on the imbalanced 99:1 data from step 1
without any imbalance handling, then shows why the resulting ~99% accuracy
is meaningless while recall on the minority class collapses.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Reuse the dataset and stratified split from step 1 (its demo prints
# are under __main__, so importing stays silent).
from Eval.DesigningMachineLearningSystems.Topic4.step1_imbalanced_split import X_test, X_train, y_test, y_train

print("Baseline RandomForest, no imbalance handling:")

# Baseline: default RandomForest trained on the imbalanced data as-is —
# no class_weight, no resampling. It optimizes overall accuracy, and with
# 99% of training labels being class 0, "predict 0" is almost always right.
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Accuracy is misleading here: a model that ALWAYS predicts class 0 scores
# 99.00% on this test set (1980/2000) without detecting a single minority
# sample. Any accuracy near 99% tells us nothing about class 1 — the class
# we actually care about. Precision/recall/F1 (with pos_label=1) score only
# the minority class, so they expose what accuracy hides.
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}  <- ~0.99 even for a model that predicts all zeros")
print(f"Precision: {precision_score(y_test, y_pred):.4f}  (of predicted 1s, how many are real 1s)")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}  (of the 20 real 1s, how many were found)")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")

# Confusion matrix layout: rows = true class, cols = predicted class.
#   [[TN, FP],
#    [FN, TP]]
# FN (bottom-left) counts the minority instances the model MISSED —
# true class 1 predicted as 0. This is the number accuracy sweeps under
# the rug: each miss costs only 0.05% accuracy on this test set.
cm = confusion_matrix(y_test, y_pred)
fn, tp = cm[1, 0], cm[1, 1]
print(f"\nMissed minority instances (false negatives): {fn} of {fn + tp} class-1 test samples")

ConfusionMatrixDisplay(cm, display_labels=clf.classes_).plot()
plt.title("Baseline RandomForest — imbalanced 99:1 test set")
out_path = Path(__file__).parent / "confusion_matrix_baseline.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"Confusion matrix saved to {out_path}")
plt.show()
