from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

import bootstraptools as bs


def _make_plans(universe_size=20, n_replicates=3, bootstrap_seed=7):
    return bs.inbag_oob(
        universe_size=universe_size,
        n_replicates=n_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def _log_all(run, plans):
    for k, plan in enumerate(plans):
        run.log_replicate(
            plan,
            metrics={"bs": 0.1 + k * 0.01, "auroc": 0.8},
            y_true=np.zeros(len(plan.eval_indices), dtype=int),
            p=np.full(len(plan.eval_indices), 0.3),
            arrays={"vec": np.arange(5)},
            model_state={"W": np.eye(2)},
            model_fit_seed=k,
        )


def test_full_run_lifecycle(tmp_path):
    plans = _make_plans()
    run = bs.init(
        tmp_path / "store",
        procedure="inbag_oob",
        dataset="synth",
        model_label="dummy",
        config={"B": 3},
        tags=["t1"],
        store_model_state=True,
    )
    _log_all(run, plans)
    run.finish(summary={"mean_bs": 0.11})

    meta = json.loads((run.run_dir / "meta.json").read_text())
    assert meta["status"] == "finished"
    assert meta["n_replicates"] == 3
    assert meta["store_model_state"] is True
    assert meta["tags"] == ["t1"]
    assert meta["summary"] == {"mean_bs": 0.11}

    replicates = pl.read_parquet(run.run_dir / "replicates.parquet")
    assert replicates.height == 3
    for col in [
        "bs",
        "auroc",
        "resample_seed",
        "n_fit",
        "n_eval",
        "model_fit_seed",
        "status",
    ]:
        assert col in replicates.columns

    predictions = pl.read_parquet(run.run_dir / "predictions.parquet")
    expected_pred_rows = sum(len(p.eval_indices) for p in plans)
    assert predictions.height == expected_pred_rows
    assert set(predictions["split"].unique().to_list()) == {"oob"}

    membership = pl.read_parquet(run.run_dir / "membership.parquet")
    assert (membership["count"] > 0).all()
    for plan in plans:
        rep_rows = membership.filter(pl.col("replicate_idx") == plan.replicate_idx)
        n_fit_unique = int((plan.fit_counts > 0).sum())
        assert rep_rows["sample_idx"].n_unique() == n_fit_unique
        assert rep_rows.height == n_fit_unique

    arr = np.load(run.run_dir / "arrays" / "replicate_0000.npz")
    np.testing.assert_array_equal(arr["vec"], np.arange(5))

    state = np.load(run.run_dir / "model_state" / "replicate_0000.npz")
    np.testing.assert_array_equal(state["W"], np.eye(2))


def test_model_state_requires_opt_in(tmp_path):
    plans = _make_plans(n_replicates=1)
    run = bs.init(
        tmp_path / "store",
        procedure="inbag_oob",
        dataset="synth",
        model_label="dummy",
        store_model_state=False,
    )
    plan = plans[0]
    with pytest.raises(ValueError):
        run.log_replicate(
            plan,
            metrics={"bs": 0.1},
            model_state={"W": np.eye(2)},
        )


def test_context_manager_writes_finished_meta(tmp_path):
    plans = _make_plans(n_replicates=2)
    with bs.init(
        tmp_path / "store",
        procedure="inbag_oob",
        dataset="synth",
        model_label="dummy",
    ) as run:
        for plan in plans:
            run.log_replicate(plan, metrics={"bs": 0.2})
        run_dir = run.run_dir

    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["status"] == "finished"
    assert meta["n_replicates"] == 2


def test_prediction_length_mismatch_raises(tmp_path):
    plans = _make_plans(n_replicates=1)
    run = bs.init(
        tmp_path / "store",
        procedure="inbag_oob",
        dataset="synth",
        model_label="dummy",
    )
    plan = plans[0]
    n = len(plan.eval_indices)
    with pytest.raises(ValueError):
        run.log_replicate(
            plan,
            metrics={"bs": 0.1},
            y_true=np.zeros(n, dtype=int),
            p=np.full(n + 1, 0.3),
        )


def test_log_apparent_records_full_sample(tmp_path):
    run = bs.init(
        tmp_path / "store",
        procedure="optimism_bootstrap",
        dataset="synth",
        model_label="dummy",
        store_model_state=True,
    )
    y_true = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    run.log_apparent(
        y_true=y_true,
        p=p,
        arrays={"coef": np.arange(3)},
        model_state={"w": np.eye(2)},
    )

    plans = bs.optimism_bootstrap(universe_size=4, n_replicates=2, bootstrap_seed=1)
    for plan in plans:
        run.log_replicate(
            plan,
            metrics={"bs": 0.1},
            y_true=np.zeros(len(plan.eval_indices), dtype=int),
            p=np.full(len(plan.eval_indices), 0.3),
        )
    run.finish()

    apparent_path = run.run_dir / "apparent.parquet"
    assert apparent_path.exists()
    apparent = pl.read_parquet(apparent_path)
    assert apparent.height == 4
    assert set(apparent.columns) == {"sample_idx", "y_true", "p"}
    apparent = apparent.sort("sample_idx")
    np.testing.assert_array_equal(apparent["sample_idx"].to_numpy(), np.arange(4))
    np.testing.assert_array_equal(apparent["y_true"].to_numpy(), y_true)
    np.testing.assert_allclose(apparent["p"].to_numpy(), p)

    arr = np.load(run.run_dir / "apparent" / "arrays.npz")
    np.testing.assert_array_equal(arr["coef"], np.arange(3))

    state = np.load(run.run_dir / "apparent" / "model_state.npz")
    np.testing.assert_array_equal(state["w"], np.eye(2))

    queried = bs.query_apparent(tmp_path / "store", run.run_dir.name)
    queried = queried.sort("sample_idx")
    np.testing.assert_array_equal(queried["sample_idx"].to_numpy(), np.arange(4))
    np.testing.assert_array_equal(queried["y_true"].to_numpy(), y_true)
    np.testing.assert_allclose(queried["p"].to_numpy(), p)


def test_log_apparent_twice_raises(tmp_path):
    run = bs.init(
        tmp_path / "store",
        procedure="optimism_bootstrap",
        dataset="synth",
        model_label="dummy",
    )
    y_true = np.array([0, 1])
    p = np.array([0.4, 0.6])
    run.log_apparent(y_true=y_true, p=p)
    with pytest.raises(ValueError):
        run.log_apparent(y_true=y_true, p=p)


def test_log_apparent_length_mismatch_raises(tmp_path):
    run = bs.init(
        tmp_path / "store",
        procedure="optimism_bootstrap",
        dataset="synth",
        model_label="dummy",
    )
    with pytest.raises(ValueError):
        run.log_apparent(y_true=np.array([0, 1, 0]), p=np.array([0.4, 0.6]))


def test_no_apparent_call_produces_no_table(tmp_path):
    plans = _make_plans(n_replicates=1)
    run = bs.init(
        tmp_path / "store",
        procedure="inbag_oob",
        dataset="synth",
        model_label="dummy",
    )
    for plan in plans:
        run.log_replicate(plan, metrics={"bs": 0.1})
    run.finish()

    assert not (run.run_dir / "apparent.parquet").exists()
    with pytest.raises(FileNotFoundError):
        bs.query_apparent(tmp_path / "store", run.run_dir.name)


def test_log_apparent_model_state_requires_opt_in(tmp_path):
    run = bs.init(
        tmp_path / "store",
        procedure="optimism_bootstrap",
        dataset="synth",
        model_label="dummy",
        store_model_state=False,
    )
    with pytest.raises(ValueError):
        run.log_apparent(
            y_true=np.array([0, 1]),
            p=np.array([0.4, 0.6]),
            model_state={"w": np.eye(2)},
        )
