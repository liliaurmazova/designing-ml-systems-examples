"""Step 7: Final comparison of all imbalance-handling methods — Chapter 4
"Training Data", Chip Huyen, "Designing Machine Learning Systems".

Trains the five approaches from steps 2-6 and evaluates every one of them
on the SAME untouched, 99:1 test set:

    1. Baseline      — RandomForest, imbalance ignored          (step 2)
    2. SMOTE         — RandomForest on resampled train set      (step 3)
    3. Weighted loss — LogisticRegression, class_weight=balanced (step 4)
    4. Mixup         — MLP + BCE, mixup-augmented batches        (step 5)
    5. Focal loss    — MLP + FocalLoss(gamma=2)                  (step 6)

Leakage rules enforced throughout (and only need auditing in three spots):
  * SMOTE resamples X_train/y_train only — test stays at 99:1.
  * StandardScaler for the MLPs is fitted on X_train only.
  * Mixup blends training batches only.
All metrics use pos_label=1 (sklearn's default), i.e. they score the
minority class; PR-AUC is computed from probabilities, not hard labels.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from Eval.DesigningMachineLearningSystems.Topic4.step1_imbalanced_split import X_test, X_train, y_test, y_train
from Eval.DesigningMachineLearningSystems.Topic4.step6_focal_loss import MLP, FocalLoss

torch.manual_seed(42)
np.random.seed(42)


def evaluate(y_true, probs, threshold=0.5):
    """Minority-class metrics from probability scores.

    precision/recall use sklearn's default pos_label=1 — the minority
    class. PR-AUC (average precision) must be computed from the raw
    probabilities: feeding it thresholded 0/1 labels would collapse the
    precision-recall curve to a single point.
    """
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds),
        "PR-AUC": average_precision_score(y_true, probs),
    }


def train_mlp(X_tr, y_tr, criterion, mixup_alpha=None, epochs=30):
    """Train the step-6 MLP; optionally with mixup-augmented batches."""
    torch.manual_seed(42)  # identical init across MLP variants
    model = MLP(X_tr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr), batch_size=256, shuffle=True
    )
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            if mixup_alpha is not None:
                # Mixup (step 5), on the TRAINING batch only: blend inputs,
                # keep both label sets, and mix the LOSSES with the same
                # gamma — the loss-space equivalent of soft labels.
                gamma = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(len(xb))
                x_mixed = gamma * xb + (1 - gamma) * xb[idx]
                pred = model(x_mixed)
                loss = gamma * criterion(pred, yb) + (1 - gamma) * criterion(pred, yb[idx])
            else:
                loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


results = {}

# 1. Baseline RF — imbalance ignored (step 2)
clf = RandomForestClassifier(random_state=42).fit(X_train, y_train)
results["Baseline RF"] = evaluate(y_test, clf.predict_proba(X_test)[:, 1])

# 2. SMOTE RF — resample TRAIN ONLY, evaluate on original test (step 3)
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_train)  # type: ignore
clf = RandomForestClassifier(random_state=42).fit(X_res, y_res)
results["SMOTE RF"] = evaluate(y_test, clf.predict_proba(X_test)[:, 1])

# 3. Weighted loss — LogReg with class_weight='balanced' (step 4)
clf = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_train, y_train)
results["Weighted LogReg"] = evaluate(y_test, clf.predict_proba(X_test)[:, 1])

# Shared MLP preprocessing: scaler fitted on TRAIN only (no leakage)
scaler = StandardScaler().fit(X_train)
X_tr = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.float32)

# 4. Mixup MLP — BCE loss on mixup-augmented training batches (step 5)
model = train_mlp(X_tr, y_tr, nn.BCEWithLogitsLoss(), mixup_alpha=0.2)
with torch.no_grad():
    results["Mixup MLP"] = evaluate(y_test, torch.sigmoid(model(X_te)).numpy())

# 5. Focal loss MLP — FocalLoss(gamma=2), plain batches (step 6)
model = train_mlp(X_tr, y_tr, FocalLoss(gamma=2.0))
with torch.no_grad():
    results["Focal MLP"] = evaluate(y_test, torch.sigmoid(model(X_te)).numpy())

# Final comparison table — all rows scored on the identical untouched
# 99:1 test set at threshold 0.5 (PR-AUC is threshold-free).
comparison = pd.DataFrame(results).T.round(4)
comparison.index.name = "approach"
print("\nAll methods, untouched 99:1 test set (n=2000, 20 minority samples):")
print(comparison)
