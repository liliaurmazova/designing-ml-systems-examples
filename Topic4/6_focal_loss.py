"""Step 6: Focal Loss (Lin et al., 2017 — RetinaNet) — Chapter 4 "Training
Data", Chip Huyen, "Designing Machine Learning Systems".

Focal loss reshapes cross-entropy so training focuses on HARD examples:

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

where p_t is the probability the model assigns to the TRUE class
(p_t = p if y=1 else 1-p). The modulating factor (1 - p_t)^gamma:

  * easy example (p_t ~ 0.99): factor = 0.01^gamma ~ 0    -> loss vanishes
  * hard example (p_t ~ 0.1):  factor = 0.9^gamma  ~ 0.8  -> loss survives

With 99:1 imbalance, plain BCE is dominated by thousands of easy majority
samples, each contributing a small but nonzero loss; their sheer number
drowns out the few minority samples. Focal loss zeroes the easy crowd out,
so the gradient budget is spent on exactly the samples the model gets
wrong — mostly the minority class. gamma controls the strength (gamma=0
recovers plain BCE; the paper's default is gamma=2).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from Eval.DesigningMachineLearningSystems.Topic4.step1_imbalanced_split import X_test, X_train, y_test, y_train

torch.manual_seed(42)
np.random.seed(42)


class FocalLoss(nn.Module):
    """Binary focal loss on raw logits: FL(p_t) = -(1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        # BCE-with-logits gives -log(p_t) per sample in a numerically
        # stable way (never computes p then log(p) separately), so:
        #   bce  = -log(p_t)
        #   p_t  = exp(-bce)
        #   FL   = (1 - p_t)^gamma * bce
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


class MLP(nn.Module):
    """Small MLP: 20 features -> 64 -> 32 -> 1 logit."""

    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_and_eval(criterion, name, X_tr, y_tr, X_te, y_te, epochs=30):
    """Train a fresh MLP with the given loss; return test-set metrics."""
    torch.manual_seed(42)  # same init for both losses -> fair comparison
    model = MLP(X_tr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te)).numpy()
    preds = (probs >= 0.5).astype(int)
    y_true = y_te.numpy().astype(int)
    return {
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds),
        # PR-AUC (average precision): threshold-independent summary of the
        # precision-recall curve — the right "area under" metric for
        # imbalanced data (ROC-AUC is inflated by the huge TN count).
        "PR-AUC": average_precision_score(y_true, probs),
    }


if __name__ == "__main__":
    # Standardize features — fitted on TRAIN only (same no-leakage rule as
    # SMOTE in step 3), then applied to test. NNs care about feature scale
    # far more than tree models do.
    scaler = StandardScaler().fit(X_train)
    X_tr = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    results = {
        "BCE (baseline)": train_and_eval(nn.BCEWithLogitsLoss(), "bce", X_tr, y_tr, X_te, y_te),
        "FocalLoss g=2": train_and_eval(FocalLoss(gamma=2.0), "focal", X_tr, y_tr, X_te, y_te),
    }

    names = list(results)
    print(f"{'metric':<12}" + "".join(f"{n:>17}" for n in names))
    for metric in results[names[0]]:
        print(f"{metric:<12}" + "".join(f"{results[n][metric]:>17.4f}" for n in names))
