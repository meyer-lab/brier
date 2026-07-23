from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

import bootstraptools as bs
from bootstraptools.optimism import (
    error_632,
    error_632_from_run,
    error_632_plus,
    optimism_correction,
    optimism_from_run,
    zero_one_loss,
)


def test_weighted_metric_equals_literal_expansion_additive_and_rank():
    """Storing predictions-on-originals + counts recovers apparent-on-resample
    exactly, for both an additive (Brier) and a rank (AUC) metric."""
    rng = np.random.default_rng(0)
    n = 40

    for _ in range(20):
        y = rng.integers(0, 2, size=n)
        p = rng.uniform(0.01, 0.99, size=n)
        c = np.bincount(rng.integers(0, n, size=n), minlength=n)

        idx = np.repeat(np.arange(n), c)
        y_exp = y[idx]
        p_exp = p[idx]

        if len(np.unique(y_exp)) < 2:
            continue  # need both classes for AUC; try again

        weighted_bs = brier_score_loss(y, p, sample_weight=c)
        expanded_bs = brier_score_loss(y_exp, p_exp)
        assert weighted_bs == pytest.approx(expanded_bs, abs=1e-12)

        weighted_auc = roc_auc_score(y, p, sample_weight=c)
        expanded_auc = roc_auc_score(y_exp, p_exp)
        assert weighted_auc == pytest.approx(expanded_auc, abs=1e-9)
        return

    pytest.fail("never generated an expanded sample with both classes present")


def _overfit_replicates(rng, X, y, n_replicates=6):
    """Fit an unrestricted 1-NN model on bootstrap resamples of (X, y),
    predicting on all n original samples; return (replicate_ps, counts)."""
    n = len(y)
    replicate_ps = np.empty((n_replicates, n))
    counts = np.empty((n_replicates, n), dtype=int)
    for b in range(n_replicates):
        fit_idx = rng.integers(0, n, size=n)
        c = np.bincount(fit_idx, minlength=n)
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X[fit_idx], y[fit_idx])
        replicate_ps[b] = model.predict_proba(X)[:, 1]
        counts[b] = c
    return replicate_ps, counts


def _make_dataset(rng, n=60):
    X = rng.normal(size=(n, 5))
    y = (rng.uniform(size=n) < 0.5).astype(int)
    return X, y


def test_optimism_correction_overfit_auc_is_pessimistic():
    rng = np.random.default_rng(1)
    X, y = _make_dataset(rng)

    apparent_model = KNeighborsClassifier(n_neighbors=1)
    apparent_model.fit(X, y)
    apparent_p = apparent_model.predict_proba(X)[:, 1]

    replicate_ps, counts = _overfit_replicates(rng, X, y, n_replicates=8)

    metric = lambda yt, pp, sample_weight=None: roc_auc_score(
        yt, pp, sample_weight=sample_weight
    )

    result = optimism_correction(y, apparent_p, replicate_ps, counts, metric)

    assert result["apparent"] > 0.95  # 1-NN memorizes the training data
    assert result["optimism"] > 0
    assert result["corrected"] < result["apparent"]
    assert result["optimism_per_replicate"].shape == (8,)


def test_optimism_correction_overfit_brier_is_pessimistic():
    rng = np.random.default_rng(2)
    X, y = _make_dataset(rng)

    apparent_model = KNeighborsClassifier(n_neighbors=1)
    apparent_model.fit(X, y)
    apparent_p = apparent_model.predict_proba(X)[:, 1]

    replicate_ps, counts = _overfit_replicates(rng, X, y, n_replicates=8)

    result = optimism_correction(y, apparent_p, replicate_ps, counts, brier_score_loss)

    # Brier is a loss (lower is better); overfitting makes apparent
    # optimistically LOW, so the corrected value is higher (worse).
    assert result["corrected"] > result["apparent"]
    assert result["optimism_per_replicate"].shape == (8,)


def _write_optimism_run(store, rng, n=30, n_replicates=5):
    X, y = _make_dataset(rng, n=n)

    apparent_model = KNeighborsClassifier(n_neighbors=1)
    apparent_model.fit(X, y)
    apparent_p = apparent_model.predict_proba(X)[:, 1]

    plans = bs.optimism_bootstrap(
        universe_size=n, n_replicates=n_replicates, bootstrap_seed=7
    )
    run = bs.init(
        store,
        procedure="optimism_bootstrap",
        dataset="synthetic",
        model_label="knn1",
    )
    run.log_apparent(y_true=y, p=apparent_p)

    replicate_ps = np.empty((n_replicates, n))
    counts = np.empty((n_replicates, n), dtype=int)
    for plan in plans:
        model = KNeighborsClassifier(n_neighbors=1)
        model.fit(X[plan.fit_indices], y[plan.fit_indices])
        p = model.predict_proba(X)[:, 1]
        run.log_replicate(plan, y_true=y, p=p)
        replicate_ps[plan.replicate_idx] = p
        counts[plan.replicate_idx] = plan.fit_counts
    run.finish()

    return run.run_dir.name, y, apparent_p, replicate_ps, counts


def test_optimism_from_run_matches_direct_call(tmp_path):
    store = tmp_path / "store"
    rng = np.random.default_rng(3)
    run_id, y, apparent_p, replicate_ps, counts = _write_optimism_run(store, rng)

    direct = optimism_correction(y, apparent_p, replicate_ps, counts, brier_score_loss)
    via_run = optimism_from_run(store, run_id, brier_score_loss)

    assert via_run["apparent"] == pytest.approx(direct["apparent"], abs=1e-12)
    assert via_run["optimism"] == pytest.approx(direct["optimism"], abs=1e-12)
    assert via_run["corrected"] == pytest.approx(direct["corrected"], abs=1e-12)
    np.testing.assert_allclose(
        via_run["optimism_per_replicate"], direct["optimism_per_replicate"], atol=1e-12
    )


def test_optimism_from_run_missing_apparent_raises(tmp_path):
    store = tmp_path / "store"
    plans = bs.inbag_oob(universe_size=10, n_replicates=2, bootstrap_seed=1)
    run = bs.init(
        store, procedure="inbag_oob", dataset="d", model_label="m"
    )
    for plan in plans:
        n = len(plan.eval_indices)
        run.log_replicate(
            plan,
            y_true=np.zeros(n, dtype=int),
            p=np.full(n, 0.3),
        )
    run.finish()

    with pytest.raises(FileNotFoundError):
        optimism_from_run(store, run.run_dir.name, brier_score_loss)


def test_632_overfit_1nn_random_labels():
    """1-NN memorizes random labels: apparent error ~0, OOB error ~0.5,
    heavy overfitting (R near 1), and .632+ >= .632 estimate."""
    rng = np.random.default_rng(0)
    X, y = _make_dataset(rng, n=80)

    apparent_model = KNeighborsClassifier(n_neighbors=1)
    apparent_model.fit(X, y)
    apparent_p = apparent_model.predict_proba(X)[:, 1]

    replicate_ps, counts = _overfit_replicates(rng, X, y, n_replicates=20)

    r632 = error_632(y, apparent_p, replicate_ps, counts, zero_one_loss)
    r632p = error_632_plus(y, apparent_p, replicate_ps, counts, zero_one_loss)

    print("err_app:", r632["apparent"])
    print("eps0:", r632["oob"])
    print("gamma:", r632p["no_information_rate"])
    print("R:", r632p["relative_overfitting_rate"])
    print("error_632 estimate:", r632["estimate"])
    print("error_632_plus estimate:", r632p["estimate"])

    assert r632["apparent"] == pytest.approx(0.0, abs=1e-9)
    assert r632["oob"] > 0.3  # clearly overfit, OOB error near 0.5
    assert r632["estimate"] == pytest.approx(0.632 * r632["oob"], abs=0.01)
    assert r632p["no_information_rate"] == pytest.approx(0.5, abs=0.1)
    assert r632p["relative_overfitting_rate"] > 0.9  # near 1
    assert r632p["estimate"] >= r632["estimate"]


def test_632_plus_no_overfit_degenerate_case():
    """When every replicate's OOB predictions equal the apparent
    predictions (no overfitting), R == 0, weight == 0.632, and .632+
    collapses exactly to .632."""
    n = 5
    B = 3
    y = np.array([0, 1, 0, 1, 0])
    apparent_p = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

    # Every replicate predicts identically to the apparent model everywhere.
    replicate_ps = np.tile(apparent_p, (B, 1))
    # Each sample is out-of-bag in exactly one replicate (round-robin), and
    # in-bag otherwise.
    counts = np.ones((B, n), dtype=int)
    for i in range(n):
        counts[i % B, i] = 0

    r632 = error_632(y, apparent_p, replicate_ps, counts, zero_one_loss)
    r632p = error_632_plus(y, apparent_p, replicate_ps, counts, zero_one_loss)

    assert r632["apparent"] == pytest.approx(r632["oob"], abs=1e-12)
    assert r632p["relative_overfitting_rate"] == 0.0
    assert r632p["weight"] == pytest.approx(0.632, abs=1e-12)
    assert r632p["estimate"] == pytest.approx(r632["estimate"], abs=1e-12)


def test_632_plus_hand_computed_formula():
    """Tiny n=4, B=2 example with known counts and squared-error loss;
    values below are computed by hand (see PLAN derivation)."""
    y = np.array([0, 0, 1, 1])
    apparent_p = np.array([0.3, 0.6, 0.4, 0.9])
    replicate_ps = np.array(
        [
            [0.3, 0.7, 0.5, 0.5],
            [0.5, 0.5, 0.1, 0.9],
        ]
    )
    counts = np.array(
        [
            [0, 0, 2, 2],
            [2, 2, 0, 0],
        ]
    )

    r632 = error_632(y, apparent_p, replicate_ps, counts, bs.squared_error_loss)
    r632p = error_632_plus(y, apparent_p, replicate_ps, counts, bs.squared_error_loss)

    assert r632["apparent"] == pytest.approx(0.205, abs=1e-9)
    assert r632["oob"] == pytest.approx(0.35, abs=1e-9)
    assert r632["estimate"] == pytest.approx(0.29664, abs=1e-9)

    assert r632p["no_information_rate"] == pytest.approx(0.305, abs=1e-9)
    # (eps0 - err_app) / (gamma - err_app) = 0.145 / 0.100 = 1.45, clamped to 1
    assert r632p["relative_overfitting_rate"] == pytest.approx(1.0, abs=1e-9)
    assert r632p["weight"] == pytest.approx(1.0, abs=1e-9)
    # eps0' = min(eps0, gamma) = 0.305; err_632+ = (1-1)*err_app + 1*0.305
    assert r632p["estimate"] == pytest.approx(0.305, abs=1e-9)


def test_632_from_run_matches_direct_call(tmp_path):
    store = tmp_path / "store"
    rng = np.random.default_rng(4)
    run_id, y, apparent_p, replicate_ps, counts = _write_optimism_run(store, rng)

    direct = error_632(y, apparent_p, replicate_ps, counts, zero_one_loss)
    via_run = error_632_from_run(store, run_id, zero_one_loss)

    assert via_run["estimate"] == pytest.approx(direct["estimate"], abs=1e-12)
    assert via_run["apparent"] == pytest.approx(direct["apparent"], abs=1e-12)
    assert via_run["oob"] == pytest.approx(direct["oob"], abs=1e-12)
