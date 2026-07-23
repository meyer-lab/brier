from __future__ import annotations

import numpy as np
import pytest

import bootstraptools as bs
from bootstraptools import uq
from bootstraptools.optimism import double_bootstrap_ci, double_bootstrap_ci_from_run


def test_double_bootstrap_ci_matches_plain_percentile():
    theta_corr = np.array([0.10, 0.12, 0.09, 0.15, 0.11, 0.08, 0.13, 0.14, 0.10, 0.11])
    alpha = 0.05

    result = double_bootstrap_ci(theta_corr, alpha=alpha)

    expected_ci = uq.percentile(theta_corr, alpha)
    assert result["ci"][0] == pytest.approx(expected_ci[0], abs=1e-12)
    assert result["ci"][1] == pytest.approx(expected_ci[1], abs=1e-12)

    assert result["outer_mean"] == pytest.approx(float(theta_corr.mean()), abs=1e-12)
    assert result["outer_median"] == pytest.approx(float(np.median(theta_corr)), abs=1e-12)
    assert result["n_outer"] == len(theta_corr)
    assert result["alpha"] == alpha


def test_double_bootstrap_ci_alpha_propagation():
    rng = np.random.default_rng(0)
    theta_corr = rng.normal(loc=0.2, scale=0.05, size=200)

    result_05 = double_bootstrap_ci(theta_corr, alpha=0.05)
    result_10 = double_bootstrap_ci(theta_corr, alpha=0.1)

    assert result_05["ci"][0] == pytest.approx(uq.percentile(theta_corr, 0.05)[0], abs=1e-12)
    assert result_05["ci"][1] == pytest.approx(uq.percentile(theta_corr, 0.05)[1], abs=1e-12)
    assert result_10["ci"][0] == pytest.approx(uq.percentile(theta_corr, 0.1)[0], abs=1e-12)
    assert result_10["ci"][1] == pytest.approx(uq.percentile(theta_corr, 0.1)[1], abs=1e-12)

    # The 90% interval must be narrower than (nested within) the 95% one.
    assert result_10["ci"][0] >= result_05["ci"][0]
    assert result_10["ci"][1] <= result_05["ci"][1]

    assert result_05["alpha"] == 0.05
    assert result_10["alpha"] == 0.1


def _build_double_optimism_run(tmp_path, theta_corr_values):
    """Build a metrics-only double_optimism_bootstrap run without model fitting."""
    store = tmp_path / "store"
    n = 20
    n_outer = len(theta_corr_values)
    n_inner = 3

    double_plans = bs.double_optimism_bootstrap(
        universe_size=n, n_outer=n_outer, n_inner=n_inner, bootstrap_seed=123
    )

    run = bs.init(
        store,
        procedure="double_optimism_bootstrap",
        dataset="synthetic",
        model_label="stub",
    )
    for dp, theta_corr in zip(double_plans, theta_corr_values):
        run.log_replicate(
            dp.outer_plan,
            metrics={
                "theta_corr": theta_corr,
                "theta_app": theta_corr + 0.01,
                "inner_optimism": 0.01,
            },
        )
    run.finish()

    return store, run.run_dir.name


def test_double_bootstrap_ci_from_run_matches_direct_call(tmp_path):
    theta_corr_values = [0.10, 0.12, 0.09, 0.15, 0.11, 0.08]
    store, run_id = _build_double_optimism_run(tmp_path, theta_corr_values)

    from_run = double_bootstrap_ci_from_run(store, run_id)
    direct = double_bootstrap_ci(theta_corr_values)

    assert from_run["ci"][0] == pytest.approx(direct["ci"][0], abs=1e-12)
    assert from_run["ci"][1] == pytest.approx(direct["ci"][1], abs=1e-12)
    assert from_run["outer_mean"] == pytest.approx(direct["outer_mean"], abs=1e-12)
    assert from_run["outer_median"] == pytest.approx(direct["outer_median"], abs=1e-12)
    assert from_run["n_outer"] == direct["n_outer"]


def test_double_bootstrap_ci_from_run_missing_column_raises(tmp_path):
    theta_corr_values = [0.10, 0.12, 0.09]
    store, run_id = _build_double_optimism_run(tmp_path, theta_corr_values)

    with pytest.raises(ValueError, match="available columns"):
        double_bootstrap_ci_from_run(store, run_id, col="nope")
