"""Efron-Gong optimism correction as pure functions over stored artifacts.

The correction is metric-agnostic: callers pass a `metric(y_true, p,
sample_weight=None) -> float` callable (sklearn's `brier_score_loss`,
`roc_auc_score`, `log_loss`, ... all satisfy this signature).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtri

from bootstraptools.query import membership_matrix, query_apparent, query_run_table
from bootstraptools.uq import percentile

MetricFn = Callable[..., float]
LossFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def squared_error_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-sample squared error `(p - y)**2` where mean over samples is the Brier score."""
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def zero_one_loss(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Per-sample 0/1 error: 1 if the thresholded prediction disagrees with `y`."""
    y = np.asarray(y)
    p = np.asarray(p)
    return ((p >= threshold).astype(int) != y).astype(float)


def _per_replicate_perf(
    y: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    metric: MetricFn,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-replicate apparent-on-resample and test-on-original performance.

    Parameters
    ----------
    y : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
        Each bootstrap replicate model's predictions on all n ORIGINAL
        samples.
    counts : np.ndarray, shape (B, n), int
        Per-replicate resample multiplicities.
    metric : callable
        `metric(y_true, p, sample_weight=None) -> float`.

    Returns
    -------
    (p_boot, p_orig) : tuple of np.ndarray, each shape (B,)
        `p_boot[b]` is replicate b's performance on the resample it was
        fit on (`metric(y, replicate_ps[b], sample_weight=counts[b])`);
        `p_orig[b]` is its performance on the original sample
        (`metric(y, replicate_ps[b])`). `p_boot - p_orig` is the
        per-replicate optimism.
    """
    B = replicate_ps.shape[0]
    p_boot = np.empty(B, dtype=float)
    p_orig = np.empty(B, dtype=float)
    for b in range(B):
        p_boot[b] = float(metric(y, replicate_ps[b], sample_weight=counts[b]))
        p_orig[b] = float(metric(y, replicate_ps[b]))
    return p_boot, p_orig


def optimism_correction(
    y: np.ndarray,
    apparent_p: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    metric: MetricFn,
) -> dict[str, Any]:
    """Efron-Gong optimism correction for one metric, given stored predictions.

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
    definitionaly NOT a confidence interval on model performance.
    """
    y = np.asarray(y)
    apparent_p = np.asarray(apparent_p)
    replicate_ps = np.asarray(replicate_ps)
    counts = np.asarray(counts)

    apparent = float(metric(y, apparent_p))

    p_boot, p_orig = _per_replicate_perf(y, replicate_ps, counts, metric)
    optimism_per_replicate = p_boot - p_orig

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


def optimism_from_run(
    store: str | Path, run_id: str, metric: MetricFn
) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `optimism_correction`.

    Requires a run produced by the `optimism_bootstrap` procedure with a
    logged apparent record (`Run.log_apparent`) and, for every replicate,
    predictions covering the full original sample (`eval_indices ==
    arange(n)`).
    """
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return optimism_correction(y, apparent_p, replicate_ps, counts, metric)


def optimism_location_shifted_ci(
    y: np.ndarray,
    apparent_p: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    metric: MetricFn,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Location-shifted CI on the Efron-Gong optimism-corrected estimate.

    This is Noma et al. (2021) "method 1": a plain percentile CI (see
    `bootstraptools.uq.percentile`) of the apparent-on-resample
    distribution `theta_boot = p_boot` (metric(y, replicate_ps[b],
    sample_weight=counts[b]) across replicates b), relocated toward the
    corrected estimate by subtracting `optimism` from both endpoints. The
    shift is applied identically regardless of whether `metric` is
    higher-better or lower-better, since both `corrected` and the percentile
    endpoints are computed in `metric`'s native direction.

    This interval is `(q_lo, q_hi) - optimism`, NOT `corrected ± (something)`.
    Because the apparent statistic's bootstrap distribution (`p_boot`) carries
    its own bias relative to the full-data apparent value, the interval is not
    guaranteed to be centered on or contain the `corrected` point estimate.

    COVERAGE CAVEAT: this interval covers well in large samples but
    under-covers in small samples (empirically ~70-80% actual coverage at
    a nominal 95% level), because it ignores the sampling variability of
    the optimism term `O` itself. Only the spread of `theta_boot` is
    used, and `O` is treated as a fixed shift. The double bootstrap
    (Noma et al. 2021 "method 2") corrects this but is much more expensive.

    Parameters
    ----------
    y, apparent_p : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
        Each bootstrap replicate model's predictions on all n original
        samples.
    counts : np.ndarray, shape (B, n), int
        Per-replicate resample multiplicities.
    metric : callable
        `metric(y_true, p, sample_weight=None) -> float`.
    alpha : float, default 0.05
        Total two-sided level; alpha/2 is spent per tail.

    Returns
    -------
    dict with keys:
        "corrected" : float
            Same as `optimism_correction`'s `corrected`.
        "apparent" : float
            Same as `optimism_correction`'s `apparent`.
        "optimism" : float
            Same as `optimism_correction`'s `optimism`.
        "ci" : tuple[float, float]
            The location-shifted interval, `(q_lo - optimism, q_hi -
            optimism)`, centered on `corrected`.
        "apparent_ci" : tuple[float, float]
            The UNSHIFTED percentile interval `(q_lo, q_hi)` of
            `theta_boot`, returned for transparency.
        "alpha" : float

    References
    ----------
    Noma, H. et al. (2021). "Confidence intervals of prediction accuracy
    measures for multivariable prediction models based on the
    bootstrap-based optimism correction methods." Statistics in Medicine.
    """
    y = np.asarray(y)
    apparent_p = np.asarray(apparent_p)
    replicate_ps = np.asarray(replicate_ps)
    counts = np.asarray(counts)

    apparent = float(metric(y, apparent_p))

    p_boot, p_orig = _per_replicate_perf(y, replicate_ps, counts, metric)
    optimism = float(np.mean(p_boot - p_orig))
    corrected = apparent - optimism

    q_lo, q_hi = percentile(p_boot, alpha)
    ci = (q_lo - optimism, q_hi - optimism)

    return {
        "corrected": corrected,
        "apparent": apparent,
        "optimism": optimism,
        "ci": ci,
        "apparent_ci": (q_lo, q_hi),
        "alpha": alpha,
    }


def optimism_ci_from_run(
    store: str | Path, run_id: str, metric: MetricFn, alpha: float = 0.05
) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `optimism_location_shifted_ci`.

    Requires a run produced by the `optimism_bootstrap` procedure with a
    logged apparent record (`Run.log_apparent`) and, for every replicate,
    predictions covering the full original sample (`eval_indices ==
    arange(n)`).
    """
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return optimism_location_shifted_ci(
        y, apparent_p, replicate_ps, counts, metric, alpha=alpha
    )


def _dual_sd(x: np.ndarray, nmin: int = 10) -> tuple[float, float]:
    """Hmisc `dualSD`: per-side RMS distance from the overall mean.

    Splits `x` at its overall mean `m` into a "bottom" group (values <= m)
    and a "top" group (values >= m); values equal to `m` fall in both
    groups. Each group's SD is the RMS distance from `m` (not from the
    group's own mean), with Bessel's correction (`group_size - 1`).

    Falls back to `std(x, ddof=1)` for both sides if `x` has fewer than
    `nmin` finite values, or if either group would have fewer than 2
    elements (avoids division by zero).
    """
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]

    overall_sd = float(np.std(finite, ddof=1)) if len(finite) >= 2 else 0.0

    if len(finite) < nmin:
        return overall_sd, overall_sd

    m = finite.mean()
    bottom = finite[finite <= m]
    top = finite[finite >= m]

    if len(bottom) < 2 or len(top) < 2:
        return overall_sd, overall_sd

    sd_bottom = float(np.sqrt(np.sum((bottom - m) ** 2) / (len(bottom) - 1)))
    sd_top = float(np.sqrt(np.sum((top - m) ** 2) / (len(top) - 1)))
    return sd_bottom, sd_top


def optimism_abcloc_ci(
    y: np.ndarray,
    apparent_p: np.ndarray,
    replicate_ps: np.ndarray,
    counts: np.ndarray,
    metric: MetricFn,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """ABCLOC ("sd2rev wtd4") CI on the Efron-Gong optimism-corrected estimate.

    An asymmetric, approximately-pivotal interval built AROUND the ordinary
    optimism-corrected point estimate (`optimism_correction`'s `corrected`),
    per Harrell's "bootcal" post and the `dualSD` helper in Hmisc's
    `rms::calibrate`/`validate` internals.

    "wtd4" quantity: for each bootstrap replicate b, form

        x_b = p_boot_b - 1.25 * p_orig_b

    i.e. the replicate's apparent-on-resample ("train") performance minus
    1.25 times its performance on the original sample ("test"). The 1.25
    weight is an EMPIRICALLY TUNED constant (see SCOPE CAVEAT below), not a
    theoretically derived one.

    `dualSD`: rather than a single SD of `x`, Hmisc splits `x` at its
    overall mean `m` into a "bottom" half (`x <= m`) and a "top" half
    (`x >= m`), and computes each half's SD as the RMS distance from the
    OVERALL mean `m` (not from the half's own mean), each with Bessel's
    correction. This allows the two tails of `x`'s distribution to imply
    different spreads when `x` is skewed.

    "sd2rev" reversal: the interval is built around `corrected` using the
    OPPOSITE-side SD for each tail -- the LOWER limit uses `sd_top` and the
    UPPER limit uses `sd_bottom`:

        ci_low  = corrected - sd_top    * z
        ci_high = corrected + sd_bottom * z

    where `z = ndtri(1 - alpha/2)`. Because the interval is symmetric in
    form around `corrected` (just with two different SDs), `ci_low <=
    corrected <= ci_high` ALWAYS holds -- unlike `optimism_location_shifted_ci`
    ("method 1") or `double_bootstrap_ci` ("method 2"), which are percentile
    intervals that need not contain the point estimate.

    SCOPE CAVEAT (read before using): this method was validated for PROPER
    SCORING RULES -- Brier score (~94.7% empirical coverage) and calibration
    slope (~95.5%) -- at a nominal 95% level. Do NOT use it for rank-based
    indices: Somers' Dxy / AUROC under-cover with this method (~85% at small
    n). The 1.25 "wtd4" constant is itself empirically tuned in a
    proper-score / regularized-logistic-regression setting; treat it as a
    heuristic, not a universal constant, for other model classes. This
    method was not part of the head-to-head comparison that included Noma
    et al.'s methods 1 and 2 (`optimism_location_shifted_ci`,
    `double_bootstrap_ci`).

    Parameters
    ----------
    y, apparent_p : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
        Each bootstrap replicate model's predictions on all n original
        samples.
    counts : np.ndarray, shape (B, n), int
        Per-replicate resample multiplicities.
    metric : callable
        `metric(y_true, p, sample_weight=None) -> float`.
    alpha : float, default 0.05
        Total two-sided level; alpha/2 is spent per tail.

    Returns
    -------
    dict with keys:
        "corrected" : float
            Same as `optimism_correction`'s `corrected`.
        "apparent" : float
            Same as `optimism_correction`'s `apparent`.
        "optimism" : float
            Same as `optimism_correction`'s `optimism`.
        "ci" : tuple[float, float]
            `(corrected - sd_top * z, corrected + sd_bottom * z)`.
        "sd_bottom" : float
        "sd_top" : float
        "alpha" : float

    References
    ----------
    Harrell, F. "bootcal" post (dualSD / sd2rev / wtd4 heuristics for
    resampling-based calibration/CI construction); Hmisc `dualSD` (source
    of the per-side RMS-from-overall-mean SD with `nmin=10` fallback).
    """
    y = np.asarray(y)
    apparent_p = np.asarray(apparent_p)
    replicate_ps = np.asarray(replicate_ps)
    counts = np.asarray(counts)

    apparent = float(metric(y, apparent_p))

    p_boot, p_orig = _per_replicate_perf(y, replicate_ps, counts, metric)
    optimism = float(np.mean(p_boot - p_orig))
    corrected = apparent - optimism

    x = p_boot - 1.25 * p_orig
    sd_bottom, sd_top = _dual_sd(x)

    z = float(ndtri(1 - alpha / 2))
    ci = (corrected - sd_top * z, corrected + sd_bottom * z)

    return {
        "corrected": corrected,
        "apparent": apparent,
        "optimism": optimism,
        "ci": ci,
        "sd_bottom": sd_bottom,
        "sd_top": sd_top,
        "alpha": alpha,
    }


def optimism_abcloc_ci_from_run(
    store: str | Path, run_id: str, metric: MetricFn, alpha: float = 0.05
) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `optimism_abcloc_ci`.

    Requires a run produced by the `optimism_bootstrap` procedure with a
    logged apparent record (`Run.log_apparent`) and, for every replicate,
    predictions covering the full original sample (`eval_indices ==
    arange(n)`).
    """
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return optimism_abcloc_ci(y, apparent_p, replicate_ps, counts, metric, alpha=alpha)


def _oob_error(
    y: np.ndarray, replicate_ps: np.ndarray, counts: np.ndarray, loss_fn: LossFn
) -> float:
    """Leave-one-out bootstrap (out-of-bag) per-sample error `eps0`.

    For each original sample i, average its loss over the replicates where
    it was out-of-bag (counts[b, i] == 0) and then average those per-sample
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

    For per-sample-loss metrics (e.g., 0/1 error rate, squared error, log-loss
    the discontinuous / improper-scoring-rule family), combining the apparent
    (in-sample) error with the leave-one-out bootstrap (out-of-bag) error:

        err_632 = 0.368 * err_app + 0.632 * eps0

    Unlike `optimism_correction` (Efron-Gong), which tests replicate models
    on the full original sample, this estimator tests replicate b's model
    only on samples that were out-of-bag for that replicate.

    Parameters
    ----------
    y, apparent_p : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
    counts : np.ndarray, shape (B, n), int
    loss_fn : callable
        `loss_fn(y_true, p) -> np.ndarray, shape (len,)` returning
        per-sample losses (not a scalar metric).

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

    Parameters
    ----------
    y, apparent_p : np.ndarray, shape (n,)
    replicate_ps : np.ndarray, shape (B, n)
    counts : np.ndarray, shape (B, n), int
    loss_fn : callable
        `loss_fn(y_true, p) -> np.ndarray, shape (len,)` returning
        per-sample losses (not a scalar metric).

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


def error_632_from_run(
    store: str | Path, run_id: str, loss_fn: LossFn
) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `error_632`."""
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return error_632(y, apparent_p, replicate_ps, counts, loss_fn)


def error_632_plus_from_run(
    store: str | Path, run_id: str, loss_fn: LossFn
) -> dict[str, Any]:
    """Load an `optimism_bootstrap` run's artifacts and apply `error_632_plus`."""
    y, apparent_p, replicate_ps, counts = _load_run_arrays(store, run_id)
    return error_632_plus(y, apparent_p, replicate_ps, counts, loss_fn)


def double_bootstrap_ci(theta_corr: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """CI on the Efron-Gong optimism-corrected estimate via Noma (2021) "method 2".

    `theta_corr` holds the R per-OUTER-replicate optimism-corrected estimates
    (`theta_corr_r`) produced by a double/two-stage (nested) bootstrap: for
    each outer bootstrap resample r of the original data, an inner bootstrap
    re-runs the whole optimism-correction procedure treating the outer
    resample as "the data," yielding one corrected estimate theta_corr_r per
    outer replicate. This function's CI is simply the PLAIN percentile
    interval (see `bootstraptools.uq.percentile`) of that {theta_corr_r}
    distribution -- no location shift is applied, since (unlike method 1's
    `theta_boot`) the outer corrected values are already centered around the
    population target.

    Because each theta_corr_r already reflects one full inner-bootstrap
    optimism correction, the spread across outer replicates captures BOTH
    the model-refit variability across outer resamples AND the sampling
    variability of the optimism-estimation step itself. This is why method 2
    covers well in small samples where method 1 (`optimism_location_shifted_ci`,
    which only captures the apparent-on-resample spread and treats the
    optimism term as a fixed shift) empirically under-covers. The cost is
    O(R x B) model fits instead of O(B).

    IMPORTANT -- point estimate: the method-2 POINT estimate is the ordinary
    optimism-corrected value on the ORIGINAL data, i.e. the `corrected` value
    from a standard `optimism_correction`/`optimism_from_run` call (single
    bootstrap on the original sample). `outer_mean`/`outer_median` returned
    here summarize the distribution of outer-replicate corrected values and
    are NOT a substitute for that point estimate. The recommended usage is to
    pair a `double_optimism_bootstrap` run (for this CI) with a plain
    `optimism_bootstrap` run (for the point estimate) that share the SAME
    `bootstrap_seed`, per the repo's shared-seed paired-run pattern.

    Parameters
    ----------
    theta_corr : array-like, shape (R,)
        Per-outer-replicate optimism-corrected estimates.
    alpha : float, default 0.05
        Total two-sided level; alpha/2 is spent per tail.

    Returns
    -------
    dict with keys:
        "ci" : tuple[float, float]
            `percentile(theta_corr, alpha)`.
        "outer_mean" : float
        "outer_median" : float
        "n_outer" : int
        "alpha" : float

    References
    ----------
    Noma, H. et al. (2021). "Confidence intervals of prediction accuracy
    measures for multivariable prediction models based on the
    bootstrap-based optimism correction methods." Statistics in Medicine.
    """
    theta_corr = np.asarray(theta_corr, dtype=float)

    ci = percentile(theta_corr, alpha)

    return {
        "ci": ci,
        "outer_mean": float(theta_corr.mean()),
        "outer_median": float(np.median(theta_corr)),
        "n_outer": int(len(theta_corr)),
        "alpha": alpha,
    }


def double_bootstrap_ci_from_run(
    store: str | Path, run_id: str, col: str = "theta_corr", alpha: float = 0.05
) -> dict[str, Any]:
    """Load a `double_optimism_bootstrap` run's replicates and apply `double_bootstrap_ci`.

    Requires a run whose `replicates` table (one row per OUTER replicate,
    logged via `Run.log_replicate(outer_plan, metrics={...})`) has a `col`
    column holding each outer replicate's optimism-corrected estimate
    (default "theta_corr"). Null values are dropped before computing the CI.
    """
    replicates = query_run_table(store, run_id, "replicates")
    if col not in replicates.columns:
        raise ValueError(
            f"Column {col!r} not found in run {run_id!r}'s replicates table; "
            f"available columns: {replicates.columns}"
        )
    theta_corr = replicates[col].drop_nulls().to_numpy()
    return double_bootstrap_ci(theta_corr, alpha=alpha)
