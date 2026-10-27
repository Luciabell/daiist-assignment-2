"""Cost logic shared by train.py (picks defaults) and app.py (live dashboard).

A "case" = one worker. Each case has:
  score   = model's confidence that the worker has NO hardhat (0 if nothing detected)
  is_viol = True if the worker really has no hardhat (ground truth)

Decision with two thresholds:
  score <  t_low            -> PASS   (model says compliant)
  t_low <= score < t_high   -> HUMAN REVIEW
  score >= t_high           -> FAIL   (model says violation)
"""
import numpy as np

# ---- Assumptions (defend each one in REPORT.md) ----
BASE_RATE = 0.05        # share of workers who really have no hardhat
C_FN = 5000.0           # EUR: violation passes unnoticed (fine + injury risk)
C_FP = 5.0              # EUR: compliant worker stopped at the gate by mistake
C_REVIEW = 1.5          # EUR: one human check (2 min at ~45 EUR/h loaded cost)
CASES_PER_DAY = 1000    # workers entering per day
MIN_PER_REVIEW = 2.0    # minutes per human check
PRODUCTIVE_HOURS = 6.5  # productive hours per person per day


def decide(scores, t_low, t_high):
    """0 = pass, 1 = review, 2 = fail."""
    scores = np.asarray(scores)
    return np.where(scores >= t_high, 2, np.where(scores >= t_low, 1, 0))


def evaluate(scores, is_viol, t_low, t_high, base_rate=BASE_RATE,
             c_fn=C_FN, c_fp=C_FP, c_review=C_REVIEW):
    scores = np.asarray(scores)
    is_viol = np.asarray(is_viol, dtype=bool)
    d = decide(scores, t_low, t_high)

    # raw confusion matrix on the test set: rows = truth (violation, compliant)
    cm = np.array([[np.sum(m & (d == k)) for k in range(3)] for m in (is_viol, ~is_viol)], float)

    # re-weight to the assumed base rate -> expected cases per 1,000
    rates = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    per1000 = rates * (1000 * np.array([[base_rate], [1 - base_rate]]))

    review = per1000[:, 1].sum()
    cost = per1000[0, 0] * c_fn + per1000[1, 2] * c_fp + review * c_review
    return {"cm": cm, "per1000": per1000, "cost_per_1000": float(cost),
            "review_share": float(review / 1000)}


def sweep(scores, is_viol, **kw):
    """Cost vs a single threshold (no review band)."""
    ts = np.round(np.arange(0.05, 0.96, 0.01), 2)
    costs = np.array([evaluate(scores, is_viol, t, t, **kw)["cost_per_1000"] for t in ts])
    return ts, costs


def best_single(scores, is_viol, **kw):
    ts, costs = sweep(scores, is_viol, **kw)
    i = int(np.argmin(costs))
    return float(ts[i]), float(costs[i])


def best_band(scores, is_viol, **kw):
    """Grid search over (t_low, t_high) with review in between."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    best = (None, None, float("inf"))
    for lo in grid:
        for hi in grid:
            if hi > lo:
                c = evaluate(scores, is_viol, lo, hi, **kw)["cost_per_1000"]
                if c < best[2]:
                    best = (float(lo), float(hi), float(c))
    return best


def staffing(review_share, cases_per_day=CASES_PER_DAY, minutes=MIN_PER_REVIEW):
    n = review_share * cases_per_day
    hours = n * minutes / 60
    return n, hours, hours / PRODUCTIVE_HOURS