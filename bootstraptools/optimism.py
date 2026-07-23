"""Efron–Gong optimism correction as PURE functions over stored artifacts.

The correction is metric-agnostic: callers pass a `metric(y_true, p,
sample_weight=None) -> float` callable (sklearn's `brier_score_loss`,
`roc_auc_score`, `log_loss`, ... all satisfy this signature).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from bootstraptools.query import membership_matrix, query_apparent, query_run_table

MetricFn = Callable[..., float]
LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def squared_error_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-sample squared error `(p - y)**2`; mean over samples is the Brier score."""
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def zero_one_loss(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Per-sample 0/1 error: 1 if the thresholded prediction disagrees with `y`."""
    y = np.asarray(y)
    p = np.asarray(p)
    return ((p >= threshold).astype(int) != y).astype(float)


def optimism_correction(
    y: np.ndarray,
    apparent_p: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    metric: MetricFn,
) -> dict[str, Any]:
    """Efron–Gong optimism correction for one metric, given stored predictions.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
        True labels of the original sample.
    apparent_p : np.ndarray, shape (n,)
        Predictions of the full-data ("apparent") model on the original
        sample.
    replicate_ps : np.ndarray, shape (B, n)
        Each bootstrap replicate model's predictions on all n ORIGINAL
        samples.
    counts : np.ndarray, shape (B, n), int
        Per-replicate resample multiplicities (fit_counts). `counts[b, i]`
        is the number of times original sample i appeared in replicate b's
        fit set (0 for out-of-bag).
    metric : callable
        `metric(y_true, p, sample_weight=None) -> float`.

    Returns
    -------
    dict with keys:
        "corrected" : float
            `apparent - optimism`, the bias-corrected point estimate.
        "apparent" : float
            `metric(y, apparent_p)`, the apparent (in-sample) performance.
        "optimism" : float
            `mean_b(optimism_b)`, the estimated optimism to subtract off.
        "optimism_per_replicate" : np.ndarray, shape (B,)
            Per-replicate optimism values `p_boot_b - p_orig_b`.

    Notes
    -----
    `corrected` is a POINT ESTIMATE. `optimism_per_replicate` is the
    Monte-Carlo spread of the bias-correction term across replicates and is
    a DIAGNOSTIC only — it is NOT a confidence interval on model
    performance. A real CI on the corrected estimate is a separate, later
    facility.

    The formula is uniform (no higher/lower-is-better branch): the
    optimism term is self-signed, so `corrected = apparent - optimism`
    works identically whether `metric` is a loss (lower is better, e.g.
    Brier score) or a score (higher is better, e.g. AUC).
    """
    y = np.asarray(y)
    apparent_p = np.asarray(apparent_p)
    replicate_ps = np.asarray(replicate_ps)
    counts = np.asarray(counts)

    apparent = float(metric(y, apparent_p))

    B = replicate_ps.shape[0]
    optimism_per_replicate = np.empty(B, dtype=float)
    for b in range(B):
        p_boot = float(metric(y, replicate_ps[b], sample_weight=counts[b]))
        p_orig = float(metric(y, replicate_ps[b]))
        optimism_per_replicate[b] = p_boot - p_orig

    optimism = float(optimism_per_replicate.mean())
    corrected = apparent - optimism

    return {
        "corrected": corrected,
        "apparent": apparent,
        "optimism": optimism,
        "optimism_per_replicate": optimism_per_replicate,
    }


def _load_run_arrays(
    store: str | Path, run_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an `optimism_bootstrap` run's artifacts into plain arrays.

    Requires a run produced by the `optimism_bootstrap` procedure with a
    logged apparent record (`Run.log_apparent`) and, for every replicate,
    predictions covering the full original sample (`eval_indices ==
    arange(n)`).

    Returns
    -------
    (y, apparent_p, replicate_ps, counts) : tuple of np.ndarray
        `y` and `apparent_p` have shape (n,); `replicate_ps` and `counts`
        have shape (B, n).

    Raises
    ------
    FileNotFoundError
        If the run has no apparent record (propagated from `query_apparent`).
    ValueError
        If any replicate's predictions do not cover exactly the full
        `0..n-1` original sample.
    """
    ap = query_apparent(store, run_id).sort("sample_idx")
    y = ap["y_true"].to_numpy()
    apparent_p = ap["p"].to_numpy()
    n = len(y)

    preds = query_run_table(store, run_id, "predictions").sort(
        ["replicate_idx", "sample_idx"]
    )
    replicate_idxs = preds["replicate_idx"].unique().sort()
    B = len(replicate_idxs)

    if preds.height != B * n:
        raise ValueError(
            "_load_run_arrays requires every replicate to have exactly "
            f"n={n} prediction rows over sample_idx 0..{n - 1}; got "
            f"{preds.height} total rows across {B} replicates."
        )
    for rep_idx, group in preds.group_by("replicate_idx"):
        sample_idx = group["sample_idx"].sort().to_numpy()
        if len(sample_idx) != n or not np.array_equal(sample_idx, np.arange(n)):
            raise ValueError(
                f"replicate_idx={rep_idx!r} does not cover sample_idx 0..{n - 1}; "
                "_load_run_arrays assumes an optimism_bootstrap run where "
                "eval = full original sample."
            )

    replicate_ps = preds["p"].to_numpy().reshape(B, n)

    counts = membership_matrix(store, run_id)
    if counts.shape != (B, n):
        raise ValueError(
            f"membership matrix shape {counts.shape} does not match "
            f"expected (B, n) = ({B}, {n})"
        )

    return y, apparent_p, replicate_ps, counts


def optimism_from_run(store: str | Path, run_id: str, metric: MetricFn) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `optimism_correction`.

    Requires a run produced by the `optimism_bootstrap` procedure with a
    logged apparent record (`Run.log_apparent`) and, for every replicate,
    predictions covering the full original sample (`eval_indices ==
    arange(n)`).

    Raises
    ------
    FileNotFoundError
        If the run has no apparent record (propagated from `query_apparent`).
    ValueError
        If any replicate's predictions do not cover exactly the full
        `0..n-1` original sample.
    """
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return optimism_correction(y, apparent_p, replicate_ps, counts, metric)


def _oob_error(
    y: np.ndarray, replicate_ps: np.ndarray, counts: np.ndarray, loss_fn: LossFn
) -> float:
    """Leave-one-out bootstrap (out-of-bag) per-sample error `eps0`.

    For each original sample i, average its loss over the replicates where
    it was out-of-bag (counts[b, i] == 0); then average those per-sample
    averages over samples that were out-of-bag at least once.
    """
    n = len(y)
    B = replicate_ps.shape[0]
    oob_mask = counts == 0  # (B, n)
    q = oob_mask.sum(axis=0)  # (n,), number of replicates i is OOB in

    per_sample_sum = np.zeros(n, dtype=float)
    for b in range(B):
        losses_b = loss_fn(y, replicate_ps[b])
        per_sample_sum += np.where(oob_mask[b], losses_b, 0.0)

    valid = q > 0
    eps0_i = np.zeros(n, dtype=float)
    eps0_i[valid] = per_sample_sum[valid] / q[valid]

    return float(eps0_i[valid].mean())


def error_632(
    y: np.ndarray,
    apparent_p: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    loss_fn: LossFn,
) -> dict[str, Any]:
    """The `.632` bootstrap estimate of out-of-sample per-sample loss.

    For PER-SAMPLE-LOSS metrics (0/1 error rate, squared error, log-loss —
    the discontinuous / improper-scoring-rule family this estimator was
    designed for), combining the apparent (in-sample) error with the
    leave-one-out bootstrap (out-of-bag) error:

        err_632 = 0.368 * err_app + 0.632 * eps0

    Unlike `optimism_correction` (Efron–Gong), which tests replicate models
    on the FULL original sample, this estimator tests replicate b's model
    only on samples that were OUT-OF-BAG for that replicate.

    Parameters
    ----------
    y, apparent_p : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
    counts : np.ndarray, shape (B, n), int
    loss_fn : callable
        `loss_fn(y_true, p) -> np.ndarray, shape (len,)` returning
        PER-SAMPLE losses (not a scalar metric).

    Returns
    -------
    dict with keys "estimate", "apparent", "oob".

    References
    ----------
    Efron, B. and Tibshirani, R. (1997). "Improvements on Cross-Validation:
    The .632+ Bootstrap Method." JASA.
    """
    y = np.asarray(y)
    apparent_p = np.asarray(apparent_p)
    replicate_ps = np.asarray(replicate_ps)
    counts = np.asarray(counts)

    err_app = float(np.mean(loss_fn(y, apparent_p)))
    eps0 = _oob_error(y, replicate_ps, counts, loss_fn)
    err_632 = 0.368 * err_app + 0.632 * eps0

    return {"estimate": err_632, "apparent": err_app, "oob": eps0}


def error_632_plus(
    y: np.ndarray,
    apparent_p: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    loss_fn: LossFn,
) -> dict[str, Any]:
    """The `.632+` bootstrap estimate of out-of-sample per-sample loss.

    Extends `error_632` with a no-information-rate correction that adapts
    the weight given to the out-of-bag error based on the degree of
    overfitting, per Efron & Tibshirani (1997), "Improvements on
    Cross-Validation: The .632+ Bootstrap Method," JASA:

        gamma = no-information rate (all-pairs mean loss under apparent
                predictions)
        R = relative overfitting rate
          = (eps0 - err_app) / (gamma - err_app)  if eps0 > err_app and
            gamma > err_app, else 0; clamped to [0, 1]
        eps0' = min(eps0, gamma)
        w = 0.632 / (1 - 0.368 * R)          # canonical E&T 1997 weight
        err_632+ = (1 - w) * err_app + w * eps0'

    This is the canonical E&T 1997 weight form (NOT the alternate
    `0.632*R / (0.632*R + (1 - R))` form used by some implementations,
    e.g. mlxtend).

    Parameters
    ----------
    y, apparent_p : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
    counts : np.ndarray, shape (B, n), int
    loss_fn : callable
        `loss_fn(y_true, p) -> np.ndarray, shape (len,)` returning
        PER-SAMPLE losses (not a scalar metric).

    Returns
    -------
    dict with keys "estimate", "apparent", "oob", "no_information_rate",
    "relative_overfitting_rate", "weight".
    """
    y = np.asarray(y)
    apparent_p = np.asarray(apparent_p)
    replicate_ps = np.asarray(replicate_ps)
    counts = np.asarray(counts)
    n = len(y)

    err_app = float(np.mean(loss_fn(y, apparent_p)))
    eps0 = _oob_error(y, replicate_ps, counts, loss_fn)

    gamma = float(
        np.mean([np.mean(loss_fn(y, np.full(n, apparent_p[j]))) for j in range(n)])
    )

    if eps0 > err_app and gamma > err_app:
        R = (eps0 - err_app) / (gamma - err_app)
    else:
        R = 0.0
    R = float(np.clip(R, 0.0, 1.0))

    eps0_prime = min(eps0, gamma)
    w = 0.632 / (1 - 0.368 * R)
    err_632_plus = (1 - w) * err_app + w * eps0_prime

    return {
        "estimate": err_632_plus,
        "apparent": err_app,
        "oob": eps0,
        "no_information_rate": gamma,
        "relative_overfitting_rate": R,
        "weight": w,
    }


def error_632_from_run(store: str | Path, run_id: str, loss_fn: LossFn) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `error_632`."""
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return error_632(y, apparent_p, replicate_ps, counts, loss_fn)


def error_632_plus_from_run(
    store: str | Path, run_id: str, loss_fn: LossFn
) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `error_632_plus`."""
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return error_632_plus(y, apparent_p, replicate_ps, counts, loss_fn)
