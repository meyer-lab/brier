from __future__ import annotations

import numpy as np
import pytest

from bootstraptools.resample import draw, draw_counts, out_of_bag, select


def test_draw_deterministic_same_seed():
    source_idx = np.arange(10)
    d1 = draw(np.random.default_rng(42), source_idx)
    d2 = draw(np.random.default_rng(42), source_idx)
    np.testing.assert_array_equal(d1, d2)


def test_draw_different_seed_differs():
    source_idx = np.arange(10)
    d1 = draw(np.random.default_rng(1), source_idx)
    d2 = draw(np.random.default_rng(2), source_idx)
    assert not np.array_equal(d1, d2)


def test_draw_size_defaults_to_len_source_idx():
    source_idx = np.arange(7)
    drawn = draw(np.random.default_rng(0), source_idx)
    assert len(drawn) == len(source_idx)


def test_draw_size_smaller_than_source_with_replace():
    source_idx = np.arange(10)
    drawn = draw(np.random.default_rng(0), source_idx, size=3, replace=True)
    assert len(drawn) == 3


def test_draw_size_larger_than_source_with_replace():
    source_idx = np.arange(5)
    drawn = draw(np.random.default_rng(0), source_idx, size=20, replace=True)
    assert len(drawn) == 20


def test_draw_no_replace_all_unique():
    source_idx = np.arange(10)
    drawn = draw(np.random.default_rng(0), source_idx, size=5, replace=False)
    assert len(drawn) == len(set(drawn.tolist()))


def test_draw_no_replace_size_too_large_raises():
    source_idx = np.arange(5)
    with pytest.raises(ValueError):
        draw(np.random.default_rng(0), source_idx, size=10, replace=False)


def test_draw_non_contiguous_source_idx():
    source_idx = [10, 20, 30, 40]
    drawn = draw(np.random.default_rng(0), source_idx, size=50, replace=True)
    assert set(drawn.tolist()) <= set(source_idx)


def test_draw_counts_sums_to_size():
    source_idx = np.arange(10)
    drawn = draw(np.random.default_rng(0), source_idx, size=17, replace=True)
    counts = draw_counts(drawn, universe_size=10)
    assert counts.sum() == 17


def test_draw_counts_zero_at_oob_positions():
    source_idx = np.array([0, 1, 2, 3, 4])
    drawn = draw(np.random.default_rng(0), source_idx, size=5, replace=False)
    counts = draw_counts(drawn, universe_size=len(source_idx))
    oob = out_of_bag(drawn, source_idx)
    for i in oob:
        assert counts[i] == 0


def test_out_of_bag_exact_complement():
    source_idx = np.arange(10)
    drawn = np.array([0, 2, 4, 6, 8])
    oob = out_of_bag(drawn, source_idx)
    np.testing.assert_array_equal(oob, np.array([1, 3, 5, 7, 9]))


def test_select_numpy_array():
    X = np.array([100, 200, 300, 400])
    idx = np.array([0, 2, 2])
    np.testing.assert_array_equal(select(X, idx), np.array([100, 300, 300]))


def test_select_python_list_of_objects():
    X = ["a", "b", "c", "d"]
    idx = [3, 1, 1]
    assert select(X, idx) == ["d", "b", "b"]
