"""Step 1: Class imbalance and stratified splitting — Chapter 4 "Training Data",
Chip Huyen, "Designing Machine Learning Systems".

Creates a synthetic binary dataset with a 99:1 class ratio and a stratified
80/20 train/test split. With only ~100 minority samples in 10,000, a plain
random split can easily land too few (or zero) minority samples in the test
set; stratify=y keeps the ratio identical in both splits.

Importable: step2_accuracy_trap.py does
    from step1_imbalanced_split import X_train, X_test, y_train, y_test
The demo prints below only run when this file is executed directly.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Synthetic dataset: 10,000 samples, 20 features, 99% class 0 / 1% class 1.
# weights sets the requested class proportions; flip_y=0 disables random
# label noise so the ratio stays exact rather than approximate.
X, y = make_classification(
    n_samples=10_000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    weights=[0.99, 0.01],
    flip_y=0,
    random_state=42,
)

# Stratified 80/20 split: stratify=y makes train_test_split sample within
# each class separately, so both splits keep the original 99:1 ratio.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


def print_distribution(name, labels):
    """Print exact per-class counts and percentages for a label array."""
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    parts = ", ".join(
        f"class {c}: {n} ({n / total:.2%})" for c, n in zip(classes, counts)
    )
    print(f"{name:>8} (n={total}): {parts}")


if __name__ == "__main__":
    print_distribution("Full", y)

    print("\nStratified split (stratify=y):")
    print_distribution("Train", y_train)
    print_distribution("Test", y_test)

    # For contrast: the same split WITHOUT stratification — the minority-class
    # count in the test set now depends on luck of the draw.
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\nPlain random split (no stratify), same seed — for comparison:")
    print_distribution("Train", y_tr2)
    print_distribution("Test", y_te2)
