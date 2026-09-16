"""Step 5: Mixup augmentation (Zhang et al., 2018) — Chapter 4 "Training
Data", Chip Huyen, "Designing Machine Learning Systems".

Mixup creates new training samples as convex combinations of existing ones:

    x' = gamma * x1 + (1 - gamma) * x2
    y' = gamma * y1 + (1 - gamma) * y2      (soft label)

with gamma ~ Beta(alpha, alpha). Instead of materializing the soft label,
the standard implementation returns BOTH label sets (y_a, y_b) plus gamma,
and the training loop mixes the losses:

    loss = gamma * criterion(pred, y_a) + (1 - gamma) * criterion(pred, y_b)

which is equivalent for linear losses like cross-entropy but works with
integer class labels directly.

Why alpha matters: Beta(0.2, 0.2) is U-shaped — gamma usually lands near
0 or 1, so most mixed samples stay close to a real sample and only a few
are strong blends. alpha=1 gives uniform mixing; alpha -> 0 recovers plain
unaugmented training.
"""

import numpy as np

from Eval.DesigningMachineLearningSystems.Topic4.step1_imbalanced_split import X_train, y_train


def mixup_data(x, y, alpha=0.2):
    """Mix a batch with a shuffled copy of itself.

    One gamma is drawn per call (per batch, as in the original paper) from
    Beta(alpha, alpha). Each sample i is blended with a random partner
    x[idx[i]] from the same batch.

    Returns:
        x_mixed: gamma * x + (1 - gamma) * x[idx]
        y_a:     original labels (weight gamma in the loss)
        y_b:     partner labels  (weight 1 - gamma)
        gamma:   the mixing coefficient actually drawn
    """
    gamma = np.random.beta(alpha, alpha)
    idx = np.random.permutation(len(x))
    x_mixed = gamma * x + (1 - gamma) * x[idx]
    return x_mixed, y, y[idx], gamma


if __name__ == "__main__":
    # Seed chosen so the 2-sample demo below draws a mid-range gamma and a
    # permutation that actually swaps the pair (Beta(0.2, 0.2) is U-shaped,
    # so an arbitrary seed would usually give gamma ~ 0 or ~ 1).
    np.random.seed(46)

    # Demo: mix one majority sample with one minority sample from our
    # training set — for an imbalanced problem this is the interesting
    # direction, as it synthesizes points BETWEEN the classes.
    i0 = np.where(y_train == 0)[0][0]  # first class-0 sample
    i1 = np.where(y_train == 1)[0][0]  # first class-1 sample
    x_pair = X_train[[i0, i1]]
    y_pair = y_train[[i0, i1]]

    x_mixed, y_a, y_b, gamma = mixup_data(x_pair, y_pair, alpha=0.2)

    print(f"gamma = {gamma:.4f}  (drawn from Beta(0.2, 0.2))\n")
    print(f"{'x1 (class ' + str(y_pair[0]) + ')':<22}", np.round(x_pair[0, :4], 3), "...")
    print(f"{'x2 (class ' + str(y_pair[1]) + ')':<22}", np.round(x_pair[1, :4], 3), "...")
    print(f"{'x_mixed[0]':<22}", np.round(x_mixed[0, :4], 3), "...")

    # Verify the formula by hand for row 0 (its partner is row 1 here):
    manual = gamma * x_pair[0] + (1 - gamma) * x_pair[1]
    assert np.allclose(x_mixed[0], manual), "mixup formula mismatch"
    print("\nRow 0 check: x_mixed[0] == gamma*x1 + (1-gamma)*x2  -> OK")

    # The label pair + gamma is what the loss would consume. The equivalent
    # soft label makes the interpretation visible:
    soft = gamma * y_a[0] + (1 - gamma) * y_b[0]
    print(f"Labels: y_a={y_a[0]}, y_b={y_b[0]}, "
          f"soft label = {gamma:.2f}*{y_a[0]} + {1 - gamma:.2f}*{y_b[0]} = {soft:.4f}")
    print(f"-> this mixed sample counts as {soft:.0%} class 1 in the loss")
