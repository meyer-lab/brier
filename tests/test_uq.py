from __future__ import annotations

import numpy as np
import pytest
import scipy.stats
from scipy.special import ndtri

import bootstraptools as bs
import bootstraptools.uq as uq


def test_percentile_basic_normal_hand_computed():
    theta_boot = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    alpha = 0.2  # 80% CI, easy quantiles
    lo, hi = uq.percentile(theta_boot, alpha=alpha)
    expected_lo = np.quantile(theta_boot, 0.1, method="linear")
    expected_hi = np.quantile(theta_boot, 0.9, method="linear")
    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)

    theta_hat = 5.5
    blo, bhi = uq.basic(theta_boot, theta_hat, alpha=alpha)
    assert blo == pytest.approx(2 * theta_hat - expected_hi)
    assert bhi == pytest.approx(2 * theta_hat - expected_lo)


def test_normal_matches_hand_formula():
    rng = np.random.default_rng(42)
    theta_boot = rng.normal(loc=3.0, scale=1.0, size=500)
    theta_hat = 3.0
    lo, hi = uq.normal(theta_boot, theta_hat, alpha=0.05)
    se = np.std(theta_boot, ddof=1)
    z = ndtri(0.975)
    assert lo == pytest.approx(theta_hat - z * se)
    assert hi == pytest.approx(theta_hat + z * se)


def test_bca_matches_scipy():
    rng = np.random.default_rng(0)
    data = rng.normal(size=60)

    def stat(x, axis=-1):
        return np.mean(x, axis=axis)

    res = scipy.stats.bootstrap(
        (data,), stat, n_resamples=2000, method="BCa", random_state=1, vectorized=True
    )
    theta_boot = res.bootstrap_distribution
    theta_hat = float(np.mean(data))
    jack = np.array([np.mean(np.delete(data, i)) for i in range(len(data))])

    lo, hi = uq.bca(theta_boot, theta_hat, jackknife=jack, alpha=0.05)

    assert lo == pytest.approx(res.confidence_interval.low, abs=1e-6)
    assert hi == pytest.approx(res.confidence_interval.high, abs=1e-6)


def test_jab_approximates_exact_jackknife_acceleration():
    rng = np.random.default_rng(0)
    data = rng.normal(size=60)
    n = len(data)
    jack = np.array([np.mean(np.delete(data, i)) for i in range(n)])

    B = 3000
    boot_rng = np.random.default_rng(123)
    theta_boot = np.empty(B)
    counts = np.empty((B, n), dtype=int)
    for b in range(B):
        drawn = boot_rng.integers(0, n, size=n)
        counts[b] = np.bincount(drawn, minlength=n)
        theta_boot[b] = data[drawn].mean()

    a_exact = uq._acceleration_from_jack(jack)
    a_jab = uq._acceleration_from_jack(uq._jab_theta_tilde(theta_boot, counts))

    print(f"a_exact={a_exact}, a_jab={a_jab}")
    assert np.sign(a_exact) == np.sign(a_jab) or abs(a_exact) < 1e-6
    assert abs(a_jab - a_exact) < 0.02


def test_bca_requires_acceleration_source():
    theta_boot = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        uq.bca(theta_boot, 2.0)


def test_membership_matrix_roundtrip(tmp_path):
    plans = bs.inbag_oob(universe_size=12, n_replicates=4, bootstrap_seed=5)
    run = bs.init(
        tmp_path / "store",
        procedure="inbag_oob",
        dataset="synth",
        model_label="dummy",
    )
    for plan in plans:
        run.log_replicate(plan, metrics={"bs": 0.1})
    run.finish()

    dense = bs.membership_matrix(tmp_path / "store", run.run_dir.name)
    assert dense.shape == (4, 12)

    for plan in plans:
        np.testing.assert_array_equal(dense[plan.replicate_idx], plan.fit_counts)


def test_ci_dispatcher():
    theta_boot = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    theta_hat = 3.0

    assert uq.ci(theta_boot, method="percentile") == uq.percentile(theta_boot)
    assert uq.ci(theta_boot, theta_hat, method="basic") == uq.basic(
        theta_boot, theta_hat
    )
    assert uq.ci(theta_boot, theta_hat, method="normal") == uq.normal(
        theta_boot, theta_hat
    )
    assert uq.ci(
        theta_boot, theta_hat, method="bca", acceleration=0.0
    ) == uq.bca(theta_boot, theta_hat, acceleration=0.0)

    with pytest.raises(ValueError):
        uq.ci(theta_boot, method="bogus")

    with pytest.raises(ValueError):
        uq.ci(theta_boot, method="basic")  # missing theta_hat


def test_studentized_matches_hand_formula():
    theta_boot = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    se_boot = np.array([0.5, 0.6, 0.4, 0.9, 1.2, 0.7, 0.5, 1.1, 0.8, 1.0])
    theta_hat = 5.0
    se_hat = 1.3
    alpha = 0.2

    lo, hi = uq.studentized(theta_boot, theta_hat, se_boot, se_hat, alpha=alpha)

    t_star = (theta_boot - theta_hat) / se_boot
    expected_lo = theta_hat - np.quantile(t_star, 1 - alpha / 2, method="linear") * se_hat
    expected_hi = theta_hat - np.quantile(t_star, alpha / 2, method="linear") * se_hat

    assert lo == pytest.approx(expected_lo)
    assert hi == pytest.approx(expected_hi)
    assert lo < hi


def test_studentized_quantile_swap_ordering():
    # Right-skewed t_star: a few large positive outliers stretch the upper
    # tail. The lower endpoint uses the UPPER (0.975) t-quantile, so it
    # should be pulled further from theta_hat than the upper endpoint (which
    # uses the lower/0.025 t-quantile).
    theta_boot = np.concatenate([np.full(95, 5.0), np.array([50.0, 60.0, 70.0, 80.0, 90.0])])
    se_boot = np.full(100, 1.0)
    theta_hat = 5.0
    se_hat = 1.0
    alpha = 0.05

    lo, hi = uq.studentized(theta_boot, theta_hat, se_boot, se_hat, alpha=alpha)

    t_star = (theta_boot - theta_hat) / se_boot
    q_upper = np.quantile(t_star, 0.975, method="linear")
    q_lower = np.quantile(t_star, 0.025, method="linear")
    assert q_upper > abs(q_lower)  # confirm the skew we set up
    assert lo == pytest.approx(theta_hat - q_upper * se_hat)
    assert hi == pytest.approx(theta_hat - q_lower * se_hat)
    # the large upper t-quantile drags the lower endpoint far below theta_hat
    assert (theta_hat - lo) > (hi - theta_hat)


def test_bayesian_bootstrap_deterministic_seed():
    values = np.random.default_rng(1).normal(size=50)
    lo1, hi1 = uq.bayesian_bootstrap(values, n_draws=500, rng=7)
    lo2, hi2 = uq.bayesian_bootstrap(values, n_draws=500, rng=7)
    assert lo1 == lo2
    assert hi1 == hi2

    lo3, hi3 = uq.bayesian_bootstrap(values, n_draws=500, rng=8)
    assert (lo1, hi1) != (lo3, hi3)


def test_bayesian_bootstrap_centered_and_covers_mean():
    rng = np.random.default_rng(3)
    values = rng.normal(size=200)
    n_draws = 4000

    w_rng = np.random.default_rng(99)
    w = w_rng.dirichlet(np.ones(len(values)), size=n_draws)
    draws = w @ values

    se = values.std(ddof=1) / np.sqrt(len(values))
    assert draws.mean() == pytest.approx(values.mean(), abs=5 * se)

    lo, hi = uq.bayesian_bootstrap(values, n_draws=n_draws, rng=99)
    assert lo < values.mean() < hi


def test_bayesian_bootstrap_close_to_percentile_bootstrap():
    rng = np.random.default_rng(11)
    values = rng.normal(loc=2.0, scale=3.0, size=300)
    alpha = 0.05

    bb_lo, bb_hi = uq.bayesian_bootstrap(values, n_draws=5000, rng=42, alpha=alpha)

    boot_rng = np.random.default_rng(43)
    n = len(values)
    B = 5000
    boot_means = np.empty(B)
    for b in range(B):
        idx = boot_rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    pct_lo, pct_hi = uq.percentile(boot_means, alpha=alpha)

    width_bb = bb_hi - bb_lo
    width_pct = pct_hi - pct_lo
    assert width_bb == pytest.approx(width_pct, rel=0.25)
    assert bb_lo == pytest.approx(pct_lo, rel=0.25, abs=0.1)
    assert bb_hi == pytest.approx(pct_hi, rel=0.25, abs=0.1)


def test_bayesian_bootstrap_accepts_generator_instance():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    gen = np.random.default_rng(0)
    lo, hi = uq.bayesian_bootstrap(values, n_draws=200, rng=gen)
    assert lo < hi


def test_ci_dispatcher_studentized():
    theta_boot = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    se_boot = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
    theta_hat = 3.0
    se_hat = 0.5

    got = uq.ci(
        theta_boot, theta_hat, method="studentized", se_boot=se_boot, se_hat=se_hat
    )
    expected = uq.studentized(theta_boot, theta_hat, se_boot, se_hat)
    assert got == expected

    with pytest.raises(ValueError):
        uq.ci(theta_boot, theta_hat, method="studentized")  # missing se_boot/se_hat

    with pytest.raises(ValueError):
        uq.ci(theta_boot, method="studentized", se_boot=se_boot, se_hat=se_hat)  # missing theta_hat
