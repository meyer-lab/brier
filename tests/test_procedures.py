from __future__ import annotations

import numpy as np

from bootstraptools.procedures import (
    ResamplePlan,
    inbag_oob,
    optimism_bootstrap,
    train_resample_holdout,
)
from bootstraptools.resample import out_of_bag


def test_inbag_oob_returns_list_of_plans():
    plans = inbag_oob(universe_size=20, n_replicates=5, bootstrap_seed=0)
    assert isinstance(plans, list)
    assert len(plans) == 5
    assert all(isinstance(p, ResamplePlan) for p in plans)


def test_inbag_oob_default_size_matches_universe():
    universe_size = 20
    plans = inbag_oob(universe_size=universe_size, n_replicates=3, bootstrap_seed=0)
    for p in plans:
        assert len(p.fit_indices) == universe_size


def test_inbag_oob_fit_eval_disjoint():
    plans = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=0)
    for p in plans:
        assert set(p.fit_indices.tolist()).isdisjoint(set(p.eval_indices.tolist()))


def test_inbag_oob_eval_matches_out_of_bag():
    universe_size = 20
    plans = inbag_oob(universe_size=universe_size, n_replicates=3, bootstrap_seed=0)
    source = np.arange(universe_size)
    for p in plans:
        expected = out_of_bag(p.fit_indices, source)
        np.testing.assert_array_equal(p.eval_indices, expected)


def test_inbag_oob_fit_counts_sum_to_size():
    plans = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=0)
    for p in plans:
        assert p.fit_counts.sum() == len(p.fit_indices)


def test_inbag_oob_fit_counts_zero_at_eval_positions():
    plans = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=0)
    for p in plans:
        for i in p.eval_indices:
            assert p.fit_counts[i] == 0


def test_inbag_oob_deterministic_same_seed():
    p1 = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=42)
    p2 = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=42)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a.fit_indices, b.fit_indices)
        np.testing.assert_array_equal(a.eval_indices, b.eval_indices)
        assert a.resample_seed == b.resample_seed


def test_inbag_oob_different_seed_differs():
    p1 = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=1)
    p2 = inbag_oob(universe_size=20, n_replicates=3, bootstrap_seed=2)
    diffs = [
        not np.array_equal(a.fit_indices, b.fit_indices) for a, b in zip(p1, p2)
    ]
    assert any(diffs)


def test_inbag_oob_size_override_respected():
    universe_size = 20
    size = universe_size // 2
    plans = inbag_oob(
        universe_size=universe_size, n_replicates=3, bootstrap_seed=0, size=size
    )
    for p in plans:
        assert len(p.fit_indices) == size
        assert p.fit_counts.sum() == size


def test_inbag_oob_procedure_label():
    plans = inbag_oob(universe_size=20, n_replicates=2, bootstrap_seed=0)
    for p in plans:
        assert p.procedure == "inbag_oob"


def test_train_resample_holdout_returns_list_of_plans():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    plans = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=4, bootstrap_seed=0
    )
    assert isinstance(plans, list)
    assert len(plans) == 4
    assert all(isinstance(p, ResamplePlan) for p in plans)


def test_train_resample_holdout_eval_fixed_across_replicates():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    plans = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=5, bootstrap_seed=0
    )
    expected = np.unique(val_idx)
    for p in plans:
        np.testing.assert_array_equal(p.eval_indices, expected)


def test_train_resample_holdout_fit_indices_subset_of_train():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    plans = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=3, bootstrap_seed=0
    )
    for p in plans:
        assert set(p.fit_indices.tolist()) <= set(train_idx.tolist())


def test_train_resample_holdout_fit_counts_zero_outside_train():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    universe_size = 20
    plans = train_resample_holdout(
        train_idx, val_idx, universe_size=universe_size, n_replicates=3, bootstrap_seed=0
    )
    for p in plans:
        for i in val_idx:
            assert p.fit_counts[i] == 0


def test_train_resample_holdout_fit_counts_sum_default():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    plans = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=3, bootstrap_seed=0
    )
    for p in plans:
        assert p.fit_counts.sum() == len(train_idx)


def test_train_resample_holdout_deterministic_same_seed():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    p1 = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=3, bootstrap_seed=7
    )
    p2 = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=3, bootstrap_seed=7
    )
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a.fit_indices, b.fit_indices)
        assert a.resample_seed == b.resample_seed


def test_train_resample_holdout_size_override_respected():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    size = len(train_idx) // 2
    plans = train_resample_holdout(
        train_idx,
        val_idx,
        universe_size=20,
        n_replicates=3,
        bootstrap_seed=0,
        size=size,
    )
    for p in plans:
        assert len(p.fit_indices) == size
        assert p.fit_counts.sum() == size


def test_train_resample_holdout_procedure_label():
    train_idx = np.arange(0, 15)
    val_idx = np.arange(15, 20)
    plans = train_resample_holdout(
        train_idx, val_idx, universe_size=20, n_replicates=2, bootstrap_seed=0
    )
    for p in plans:
        assert p.procedure == "train_resample_holdout"


def test_optimism_bootstrap_returns_list_of_plans():
    plans = optimism_bootstrap(universe_size=20, n_replicates=5, bootstrap_seed=0)
    assert isinstance(plans, list)
    assert len(plans) == 5
    assert all(isinstance(p, ResamplePlan) for p in plans)


def test_optimism_bootstrap_eval_is_full_original_every_replicate():
    universe_size = 20
    plans = optimism_bootstrap(
        universe_size=universe_size, n_replicates=5, bootstrap_seed=0
    )
    expected = np.arange(universe_size)
    for p in plans:
        np.testing.assert_array_equal(p.eval_indices, expected)


def test_optimism_bootstrap_fit_indices_default_size_and_range():
    universe_size = 20
    plans = optimism_bootstrap(
        universe_size=universe_size, n_replicates=3, bootstrap_seed=0
    )
    for p in plans:
        assert len(p.fit_indices) == universe_size
        assert p.fit_indices.min() >= 0
        assert p.fit_indices.max() < universe_size


def test_optimism_bootstrap_fit_drawn_with_replacement_has_duplicates():
    universe_size = 20
    plans = optimism_bootstrap(
        universe_size=universe_size, n_replicates=3, bootstrap_seed=0
    )
    for p in plans:
        assert len(set(p.fit_indices.tolist())) < len(p.fit_indices)


def test_optimism_bootstrap_fit_counts_sum_and_length():
    universe_size = 20
    plans = optimism_bootstrap(
        universe_size=universe_size, n_replicates=3, bootstrap_seed=0
    )
    for p in plans:
        assert p.fit_counts.sum() == len(p.fit_indices)
        assert len(p.fit_counts) == universe_size


def test_optimism_bootstrap_deterministic_same_seed():
    p1 = optimism_bootstrap(universe_size=20, n_replicates=3, bootstrap_seed=42)
    p2 = optimism_bootstrap(universe_size=20, n_replicates=3, bootstrap_seed=42)
    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a.fit_indices, b.fit_indices)
        np.testing.assert_array_equal(a.eval_indices, b.eval_indices)
        assert a.resample_seed == b.resample_seed


def test_optimism_bootstrap_different_seed_differs():
    p1 = optimism_bootstrap(universe_size=20, n_replicates=3, bootstrap_seed=1)
    p2 = optimism_bootstrap(universe_size=20, n_replicates=3, bootstrap_seed=2)
    diffs = [
        not np.array_equal(a.fit_indices, b.fit_indices) for a, b in zip(p1, p2)
    ]
    assert any(diffs)


def test_optimism_bootstrap_size_override_respected():
    universe_size = 20
    size = universe_size // 2
    plans = optimism_bootstrap(
        universe_size=universe_size, n_replicates=3, bootstrap_seed=0, size=size
    )
    for p in plans:
        assert len(p.fit_indices) == size
        assert p.fit_counts.sum() == size


def test_optimism_bootstrap_procedure_label():
    plans = optimism_bootstrap(universe_size=20, n_replicates=2, bootstrap_seed=0)
    for p in plans:
        assert p.procedure == "optimism_bootstrap"
