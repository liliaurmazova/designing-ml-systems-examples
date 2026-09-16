"""Step 4: class_weight='balanced' — fixing imbalance in the loss function,
not the data. Chapter 4 "Training Data", Chip Huyen, "Designing ML Systems".

Instead of changing WHAT the model trains on (SMOTE, step 3), class weights
change HOW MUCH each training mistake costs. The dataset stays untouched.

The math of class_weight='balanced' (sklearn convention):

    w_c = n_samples / (n_classes * n_samples_of_class_c)

For our training set (n=8000, two classes, 7920 zeros / 80 ones):

    w_0 = 8000 / (2 * 7920) = 0.5051
    w_1 = 8000 / (2 *   80) = 50.0

so one minority sample "weighs" as much as ~99 majority samples
(w_1 / w_0 = 7920/80 = 99 — exactly the inverse of the class ratio),
and each class contributes the same TOTAL weight to the loss:
7920 * 0.5051 = 80 * 50.0 = 4000.

Where the weights enter:

  * LogisticRegression minimizes weighted log-loss — each sample's term
    is multiplied by its class weight:

        L = -(1/n) * sum_i  w_{y_i} * [ y_i*log(p_i) + (1-y_i)*log(1-p_i) ]

    Misclassifying a minority sample now costs 99x more than a majority
    one, so the decision boundary shifts toward the majority region and
    the learned intercept stops encoding "class 1 is a priori rare".

  * RandomForest uses the weights inside each tree's split criterion —
    impurity (Gini) is computed from weighted class frequencies:

        p_c = sum of w over class-c samples in node / total node weight
        Gini = 1 - sum_c p_c^2

    A node with 99 zeros and 1 one is "pure" unweighted (Gini ~ 0.02)
    but maximally impure weighted (both classes contribute ~equal weight,
    Gini ~ 0.5) — so trees keep splitting to isolate minority samples
    instead of writing them off. Leaf votes are weighted the same way.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

# Same dataset and stratified split as steps 1-3
from Eval.DesigningMachineLearningSystems.Topic4.step1_imbalanced_split import X_test, X_train, y_test, y_train

# Show the actual weights sklearn derives from the formula above
weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
print(f"class_weight='balanced' resolves to: w_0={weights[0]:.4f}, w_1={weights[1]:.4f} "
      f"(ratio {weights[1] / weights[0]:.0f}:1)\n")

models = {
    "RF baseline": RandomForestClassifier(random_state=42),
    "LogReg balanced": LogisticRegression(class_weight="balanced", max_iter=1000),
    "RF balanced": RandomForestClassifier(class_weight="balanced", random_state=42),
}

# All models train on the ORIGINAL imbalanced training set (no resampling)
# and are evaluated on the untouched 99:1 test set.
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
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

names = list(results)
print(f"{'metric':<18}" + "".join(f"{n:>17}" for n in names))
for metric in results[names[0]]:
    row = f"{metric:<18}"
    for n in names:
        v = results[n][metric]
        row += f"{v:>17d}" if isinstance(v, int) else f"{v:>17.4f}"
    print(row)

# Note on RandomForest + class_weight: weighted Gini changes how splits are
# chosen, but each tree still predicts by (weighted) majority in its leaf,
# and deep trees isolate samples into small pure leaves anyway — so the
# effect is often weaker than for LogisticRegression, where the weights
# directly bend a single global decision boundary.
