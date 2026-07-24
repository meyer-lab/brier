"""Confidence-interval construction over a stored bootstrap replicate distribution.

All functions here are PURE functions over `theta_boot` (and related inputs)
and return a plain `(low: float, high: float)` tuple. `alpha` is the total
two-sided level (default 0.05 -> 95% CI; alpha/2 is spent per tail).
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr, ndtri


def _q(theta_boot: np.ndarray, a: float) -> float:
    """Linear-interpolation quantile of the replicate distribution."""
    return float(np.quantile(theta_boot, a, method="linear"))


def percentile(theta_boot: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI: `(q(alpha/2), q(1-alpha/2))`."""
    theta_boot = np.asarray(theta_boot)
    return _q(theta_boot, alpha / 2), _q(theta_boot, 1 - alpha / 2)


def basic(
    theta_boot: np.ndarray, theta_hat: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Basic (reflected percentile) CI."""
    theta_boot = np.asarray(theta_boot)
    return (
        2 * theta_hat - _q(theta_boot, 1 - alpha / 2),
        2 * theta_hat - _q(theta_boot, alpha / 2),
    )


def normal(
    theta_boot: np.ndarray, theta_hat: float, alpha: float = 0.05
) -> tuple[float, float]:
    """Plain (not bias-corrected) normal-approximation CI."""
    theta_boot = np.asarray(theta_boot)
    se = np.std(theta_boot, ddof=1)
    z = ndtri(1 - alpha / 2)
    return float(theta_hat - z * se), float(theta_hat + z * se)


def _z0(theta_boot: np.ndarray, theta_hat: float) -> float:
    """Bias-correction z0 (SciPy convention, kept finite at the edges)."""
    theta_boot = np.asarray(theta_boot)
    B = len(theta_boot)
    prop = (
        np.sum(theta_boot < theta_hat) + np.sum(theta_boot <= theta_hat)
    ) / (2 * B)
    return float(ndtri(prop))


def _acceleration_from_jack(jack_values: np.ndarray) -> float:
    """Acceleration from leave-one-out jackknife values (Efron & Tibshirani / SciPy sign)."""
    jack = np.asarray(jack_values, float)
    dot = jack.mean()
    d = dot - jack  # mean minus leave-one-out
    num = np.sum(d**3)
    den = np.sum(d**2)
    return float(num / (6 * den**1.5))


def _jab_theta_tilde(theta_boot: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Jackknife-after-bootstrap theta-tilde values, no refits required.

    `counts` has shape (B, universe_size). Entry [b, i] is the number of
    times unit i appeared in replicate b's fit set. For each unit i,
    theta_tilde_(i) is the mean of `theta_boot[b]` over replicates b where
    unit i was ABSENT (`counts[b, i] == 0`). Only units resampled at least
    once overall and absent from at least one replicate are included.
    """
    theta_boot = np.asarray(theta_boot)
    counts = np.asarray(counts)
    resampled = counts.sum(axis=0) > 0
    absent_somewhere = (counts == 0).any(axis=0)
    included = np.nonzero(resampled & absent_somewhere)[0]
    if len(included) < 2:
        raise ValueError(
            "JAB acceleration undefined: fewer than 2 qualifying units "
            f"(got {len(included)})."
        )
    theta_tilde = np.empty(len(included), dtype=float)
    for pos, i in enumerate(included):
        absent_mask = counts[:, i] == 0
        theta_tilde[pos] = theta_boot[absent_mask].mean()
    return theta_tilde


def bca(
    theta_boot: np.ndarray,
    theta_hat: float,
    *,
    jackknife: np.ndarray | None = None,
    membership: np.ndarray | None = None,
    acceleration: float | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bias-corrected and accelerated (BCa) CI.

    Exactly one of `acceleration`, `jackknife`, or `membership` must be
    provided to determine the acceleration `a`.
    """
    theta_boot = np.asarray(theta_boot)
    z0 = _z0(theta_boot, theta_hat)

    if acceleration is not None:
        a = acceleration
    elif jackknife is not None:
        a = _acceleration_from_jack(jackknife)
    elif membership is not None:
        a = _acceleration_from_jack(_jab_theta_tilde(theta_boot, membership))
    else:
        raise ValueError(
            "BCa requires one of: acceleration, jackknife, or membership"
        )

    zL = ndtri(alpha / 2)
    zU = ndtri(1 - alpha / 2)

    def adj(z: float) -> float:
        return float(ndtr(z0 + (z0 + z) / (1 - a * (z0 + z))))

    a1 = adj(zL)
    a2 = adj(zU)
    return _q(theta_boot, a1), _q(theta_boot, a2)


def studentized(
    theta_boot: np.ndarray,
    theta_hat: float,
    se_boot: np.ndarray,
    se_hat: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap-t (studentized) CI.

    `theta_boot` and `se_boot` are aligned arrays of length B (`se_boot[b]`
    is replicate b's own SE estimate). `se_hat` is the SE of the point
    estimate. Note the quantile swap: the UPPER t-quantile (1 - alpha/2)
    maps to the LOWER endpoint, and vice versa.
    """
    theta_boot = np.asarray(theta_boot)
    se_boot = np.asarray(se_boot)
    t_star = (theta_boot - theta_hat) / se_boot
    low = theta_hat - _q(t_star, 1 - alpha / 2) * se_hat
    high = theta_hat - _q(t_star, alpha / 2) * se_hat
    return float(low), float(high)


def bayesian_bootstrap(
    values: np.ndarray,
    n_draws: int = 2000,
    rng: int | np.random.Generator | None = None,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Rubin (1981) Bayesian bootstrap CI for a mean-type statistic.

    `values` holds the per-observation contributions l_1..l_n (e.g.
    per-patient squared errors for Brier score). Each draw reweights the
    observations by a Dirichlet(1,...,1) vector (all-ones parameter) and
    takes the weighted mean. This is the smoothed analogue of the ordinary
    (multinomial) percentile bootstrap. Returns the percentile interval of
    the resulting draws.

    `rng` accepts an int seed, an existing `np.random.Generator`, or None.
    """
    values = np.asarray(values, float)
    n = len(values)
    rng = np.random.default_rng(rng)
    w = rng.dirichlet(np.ones(n), size=n_draws)
    draws = w @ values
    return _q(draws, alpha / 2), _q(draws, 1 - alpha / 2)


def ci(
    theta_boot: np.ndarray,
    theta_hat: float | None = None,
    method: str = "bca",
    **kwargs: object,
) -> tuple[float, float]:
    """Dispatch to the requested CI method.

    `method` in {"percentile", "basic", "normal", "bca", "studentized"}.
    `theta_hat` is required for "basic", "normal", "bca", and "studentized".
    Remaining `kwargs` (e.g. `alpha`, `jackknife`, `membership`,
    `acceleration`, `se_boot`, `se_hat`) are forwarded.

    `bayesian_bootstrap` is not dispatched here as it operates on raw
    per-observation `values` rather than the replicate distribution
    `theta_boot`, so it is called directly instead.
    """
    if method == "percentile":
        return percentile(theta_boot, **kwargs)  # type: ignore[arg-type]
    if method == "basic":
        if theta_hat is None:
            raise ValueError("basic requires theta_hat")
        return basic(theta_boot, theta_hat, **kwargs)  # type: ignore[arg-type]
    if method == "normal":
        if theta_hat is None:
            raise ValueError("normal requires theta_hat")
        return normal(theta_boot, theta_hat, **kwargs)  # type: ignore[arg-type]
    if method == "bca":
        if theta_hat is None:
            raise ValueError("bca requires theta_hat")
        return bca(theta_boot, theta_hat, **kwargs)  # type: ignore[arg-type]
    if method == "studentized":
        if theta_hat is None:
            raise ValueError("studentized requires theta_hat")
        if "se_boot" not in kwargs or "se_hat" not in kwargs:
            raise ValueError("studentized requires se_boot and se_hat")
        return studentized(theta_boot, theta_hat, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown method {method!r}")
