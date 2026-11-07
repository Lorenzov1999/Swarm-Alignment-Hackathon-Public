"""Scores against binary labels, and a small L2 logistic regression."""
import numpy as np
from scipy import optimize
from scipy.stats import rankdata


def auc(score, y):
    """Probability that a random positive outscores a random negative (ties count half)."""
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at_fpr(score, y, fpr=0.10):
    """Share of positives above the threshold that flags `fpr` of the negatives.

    Positives tied with the threshold count as missed, so a feature that is zero on
    most pairs is not credited for ties."""
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    if not y.any() or y.all():
        return float('nan')
    thr = np.quantile(score[~y], 1 - fpr)
    return float((score[y] > thr).mean())


def summary(score, y):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    fired = score > 0
    reach = fired[y].mean() if y.any() else float('nan')
    base = fired[~y].mean() if (~y).any() else float('nan')
    return {
        'reach': round(float(reach), 3),
        'fires_on_negatives': round(float(base), 3),
        'lift': round(float(reach / base), 2) if base > 0 else None,
        'auc': round(auc(score, y), 3),
        'recall_at_10pct_fa': round(recall_at_fpr(score, y), 3),
    }


class Logistic:
    """L2-penalised logistic regression on standardised features."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1.0
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        n = len(y)

        def loss(w):
            s = Z @ w
            ll = np.logaddexp(0, s) - y * s
            pen = 0.5 * self.l2 * (w[1:] @ w[1:])
            g = Z.T @ (1 / (1 + np.exp(-s)) - y)
            g[1:] += self.l2 * w[1:]
            return ll.sum() / n + pen / n, g / n

        res = optimize.minimize(loss, np.zeros(Z.shape[1]), jac=True, method='L-BFGS-B')
        self.w = res.x
        return self

    def decision(self, X):
        Z = (np.asarray(X, dtype=float) - self.mu) / self.sd
        return self.w[0] + Z @ self.w[1:]

    def coefficients(self, names):
        if len(names) != len(set(names)):
            raise ValueError('Duplicate feature names would hide repeated regularized columns')
        return dict(zip(names, np.round(self.w[1:], 3)))
