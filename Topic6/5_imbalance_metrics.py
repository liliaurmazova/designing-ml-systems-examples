"""Topic 6, Step 5: ROC-AUC vs PR-AUC as imbalance grows.

Same classifier, same data-generating process, three class balances:
50:50, 90:10, 99:1. ROC and PR curves side by side show why ROC-AUC
stays flattering under extreme imbalance while PR-AUC collapses.

The mechanism, in one sentence: the ROC x-axis (FPR = FP / all negatives)
divides false positives by an enormous negative class, so thousands of
false alarms barely move it — while PR's precision (TP / (TP + FP))
divides by predicted positives, feeling every false alarm at full weight.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
# Dataset sizes grow with imbalance so the MINORITY count stays ~2000 in
# all three datasets. Otherwise the 99:1 model would also be starved of
# positive training samples, and the genuinely-harder-learning effect
# would blur the pure metric effect this demo isolates. flip_y=0 for the
# same reason: label noise scales with the majority class and would
# manufacture fake positives at 99:1.
RATIOS = [(0.50, 4_000, "50:50", "#2a78d6"),
          (0.90, 20_000, "90:10", "#eb6834"),
          (0.99, 200_000, "99:1", "#1baf7a")]

fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
rows = []

for majority, n_samples, label, color in RATIOS:
    X, y = make_classification(
        n_samples=n_samples, n_features=20, n_informative=5, n_redundant=2,
        weights=[majority], flip_y=0.0, class_sep=0.4,
        random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE)

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                 random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    probs = clf.predict_proba(X_te)[:, 1]

    roc_auc = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)
    pos_rate = y_te.mean()
    rows.append((label, pos_rate, roc_auc, pr_auc))

    fpr, tpr, _ = roc_curve(y_te, probs)
    ax_roc.plot(fpr, tpr, color=color, lw=2,
                label=f"{label}  ROC-AUC={roc_auc:.3f}")

    prec, rec, _ = precision_recall_curve(y_te, probs)
    ax_pr.plot(rec, prec, color=color, lw=2,
               label=f"{label}  PR-AUC={pr_auc:.3f}")
    # Random-classifier baseline for THIS dataset: precision == positive
    # rate. It moves with imbalance — unlike ROC's fixed diagonal.
    ax_pr.axhline(pos_rate, color=color, lw=1, ls="--", alpha=0.5)

ax_roc.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--",
            label="random (AUC=0.5)")
ax_roc.set(title="ROC curves — all look strong",
           xlabel="False Positive Rate", ylabel="True Positive Rate")
ax_pr.set(title="PR curves — imbalance exposed\n(dashes: each dataset's random baseline)",
          xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))

for ax in (ax_roc, ax_pr):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9)

fig.suptitle("Same classifier, three class balances", y=1.0)
fig.tight_layout()
out = __file__.replace("step5_imbalance_metrics.py", "metrics_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Figure saved to {out}\n")

# --- Console breakdown -------------------------------------------------------
print(f"{'balance':<8}{'pos rate':>9}{'ROC-AUC':>9}{'PR-AUC':>8}{'PR random base':>16}")
for label, pos, roc, pr in rows:
    print(f"{label:<8}{pos:>9.3f}{roc:>9.3f}{pr:>8.3f}{pos:>16.3f}")

print("""
WHY ROC-AUC FLATTERS UNDER EXTREME IMBALANCE
* ROC-AUC stays in the same strong band across all three datasets, while
  PR-AUC roughly halves by 99:1. Neither metric is "wrong" — they answer
  different questions.
* ROC's x-axis is FPR = FP / (FP + TN). At 99:1 the test set holds
  ~59,000 negatives: even 3,000 false alarms yield FPR = 0.05 —
  invisibly close to the axis. ROC-AUC only measures RANKING: are
  positives, on average, scored above negatives? Class sizes cancel out.
* Precision = TP / (TP + FP) has no such cushion: every false alarm lands
  in the denominator next to a tiny TP count. At a 1% positive rate, a
  model must be extremely selective before precision looks good at all.
* The baselines tell the story: a random classifier gets ROC-AUC 0.5 on
  ANY balance, but PR-AUC equal to the positive rate (0.5 / 0.1 / 0.01
  here). ROC hides the difficulty of the problem; PR states it.
* Rule of thumb: report PR-AUC (and per-class metrics) whenever the
  positive class is rare and false alarms have cost — fraud, disease,
  churn. Keep ROC-AUC for balanced problems or when ranking quality is
  genuinely the question.""")
