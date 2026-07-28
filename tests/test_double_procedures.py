from __future__ import annotations

import numpy as np

from bootstraptools.procedures import (
    DoubleResamplePlan,
    ResamplePlan,
    double_optimism_bootstrap,
)
from bootstraptools.resample import draw_counts


def test_double_optimism_bootstrap_returns_list_of_double_plans():
    plans = double_optimism_bootstrap(
        universe_size=20, n_outer=4, n_inner=3, bootstrap_seed=0
    )
    assert isinstance(plans, list)
    assert len(plans) == 4
    assert all(isinstance(p, DoubleResamplePlan) for p in plans)
    for p in plans:
        assert isinstance(p.outer_plan, ResamplePlan)
        assert len(p.inner_plans) == 3
        assert all(isinstance(ip, ResamplePlan) for ip in p.inner_plans)


def test_double_optimism_bootstrap_outer_idx_matches_position():
    plans = double_optimism_bootstrap(
        universe_size=20, n_outer=5, n_inner=2, bootstrap_seed=0
    )
    for r, p in enumerate(plans):
        assert p.outer_idx == r
        assert p.outer_plan.replicate_idx == r


def test_double_optimism_bootstrap_inner_replicate_idx_matches_position():
    plans = double_optimism_bootstrap(
        universe_size=20, n_outer=3, n_inner=4, bootstrap_seed=0
    )
    for p in plans:
        for b, ip in enumerate(p.inner_plans):
            assert ip.replicate_idx == b


def test_double_optimism_bootstrap_outer_draws_from_original_universe():
    universe_size = 20
    plans = double_optimism_bootstrap(
        universe_size=universe_size, n_outer=3, n_inner=2, bootstrap_seed=0
    )
    expected_eval = np.arange(universe_size)
    for p in plans:
        outer = p.outer_plan
        assert outer.procedure == "double_optimism_bootstrap"
        assert len(outer.fit_indices) == universe_size
        assert outer.fit_indices.min() >= 0
        assert outer.fit_indices.max() < universe_size
        np.testing.assert_array_equal(outer.eval_indices, expected_eval)
        # with-replacement draw of size == universe_size should have dups
        assert len(set(outer.fit_indices.tolist())) < len(outer.fit_indices)


def test_double_optimism_bootstrap_inner_draws_from_local_positions():
    universe_size = 20
    plans = double_optimism_bootstrap(
        universe_size=universe_size, n_outer=3, n_inner=5, bootstrap_seed=0
    )
    expected_eval = np.arange(universe_size)
    for p in plans:
        for ip in p.inner_plans:
            assert ip.procedure == "double_optimism_bootstrap"
            assert len(ip.fit_indices) == universe_size
            assert ip.fit_indices.min() >= 0
            assert ip.fit_indices.max() < universe_size
            np.testing.assert_array_equal(ip.eval_indices, expected_eval)


def test_double_optimism_bootstrap_fit_counts_consistency():
    universe_size = 15
    plans = double_optimism_bootstrap(
        universe_size=universe_size, n_outer=3, n_inner=3, bootstrap_seed=0
    )
    for p in plans:
        outer = p.outer_plan
        assert len(outer.fit_counts) == universe_size
        assert outer.fit_counts.sum() == len(outer.fit_indices)
        np.testing.assert_array_equal(
            outer.fit_counts, draw_counts(outer.fit_indices, universe_size)
        )
        for ip in p.inner_plans:
            assert len(ip.fit_counts) == universe_size
            assert ip.fit_counts.sum() == len(ip.fit_indices)
            np.testing.assert_array_equal(
                ip.fit_counts, draw_counts(ip.fit_indices, universe_size)
            )


def test_double_optimism_bootstrap_deterministic_same_seed():
    p1 = double_optimism_bootstrap(
        universe_size=20, n_outer=3, n_inner=3, bootstrap_seed=42
    )
    p2 = double_optimism_bootstrap(
        universe_size=20, n_outer=3, n_inner=3, bootstrap_seed=42
    )
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a.outer_plan.fit_indices, b.outer_plan.fit_indices)
        assert a.outer_plan.resample_seed == b.outer_plan.resample_seed
        for ia, ib in zip(a.inner_plans, b.inner_plans):
            np.testing.assert_array_equal(ia.fit_indices, ib.fit_indices)
            assert ia.resample_seed == ib.resample_seed


def test_double_optimism_bootstrap_different_seed_differs():
    p1 = double_optimism_bootstrap(
        universe_size=20, n_outer=3, n_inner=3, bootstrap_seed=1
    )
    p2 = double_optimism_bootstrap(
        universe_size=20, n_outer=3, n_inner=3, bootstrap_seed=2
    )
    outer_diffs = [
        not np.array_equal(a.outer_plan.fit_indices, b.outer_plan.fit_indices)
        for a, b in zip(p1, p2)
    ]
    assert any(outer_diffs)


def test_double_optimism_bootstrap_distinct_outer_replicates_have_distinct_inner_draws():
    plans = double_optimism_bootstrap(
        universe_size=20, n_outer=3, n_inner=3, bootstrap_seed=0
    )
    # inner seeds are derived from each outer replicate's own outer_seed,
    # so different outer replicates should (with overwhelming probability)
    # produce different inner fit draws.
    all_inner_fits = [
        tuple(ip.fit_indices.tolist())
        for p in plans
        for ip in p.inner_plans
    ]
    assert len(set(all_inner_fits)) == len(all_inner_fits)


def test_double_optimism_bootstrap_size_override_respected():
    universe_size = 20
    size = universe_size // 2
    plans = double_optimism_bootstrap(
        universe_size=universe_size,
        n_outer=2,
        n_inner=2,
        bootstrap_seed=0,
        size=size,
    )
    for p in plans:
        assert len(p.outer_plan.fit_indices) == size
        assert p.outer_plan.fit_counts.sum() == size
        for ip in p.inner_plans:
            assert len(ip.fit_indices) == size
            assert ip.fit_counts.sum() == size


def test_double_optimism_bootstrap_inner_indices_address_the_outer_resample():
    """Inner plans must be usable as positions into the outer resample."""
    universe_size = 20
    size = universe_size // 2
    plans = double_optimism_bootstrap(
        universe_size=universe_size,
        n_outer=2,
        n_inner=2,
        bootstrap_seed=0,
        size=size,
    )
    X = np.arange(universe_size) * 10  # stand-in for a row-indexable dataset

    for p in plans:
        X_out = X[p.outer_plan.fit_indices]
        assert len(X_out) == size
        for ip in p.inner_plans:
            # Must index the outer resample without raising.
            assert ip.fit_indices.max() < size
            assert ip.eval_indices.max() < size
            X_inner = X_out[ip.fit_indices]
            assert len(X_inner) == size
            # fit_counts is the local multiplicity vector over the outer
            # resample, so it must be one weight per outer-resample row.
            assert len(ip.fit_counts) == size
            assert np.array_equal(
                ip.fit_counts, draw_counts(ip.fit_indices, size)
            )


def test_double_optimism_bootstrap_default_size_unchanged_by_local_source():
    """The size==universe_size path (every current caller) must be untouched."""
    plans = double_optimism_bootstrap(
        universe_size=15, n_outer=3, n_inner=4, bootstrap_seed=7
    )
    for p in plans:
        for ip in p.inner_plans:
            assert len(ip.fit_counts) == 15
            assert ip.fit_indices.max() < 15
            # explicit-size call must agree with the default exactly
    explicit = double_optimism_bootstrap(
        universe_size=15, n_outer=3, n_inner=4, bootstrap_seed=7, size=15
    )
    for p, q in zip(plans, explicit):
        assert np.array_equal(p.outer_plan.fit_indices, q.outer_plan.fit_indices)
        for ip, iq in zip(p.inner_plans, q.inner_plans):
            assert np.array_equal(ip.fit_indices, iq.fit_indices)
