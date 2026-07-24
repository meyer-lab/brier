from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import bootstraptools as bs


def _build_store(tmp_path):
    store = tmp_path / "store"

    plans_a = bs.inbag_oob(universe_size=10, n_replicates=2, bootstrap_seed=1)
    run_a = bs.init(
        store,
        procedure="inbag_oob",
        dataset="bal",
        model_label="ULTRA",
        config={"rank": 4, "B": 2},
        tags=["fig3", "pooling"],
    )
    for plan in plans_a:
        n = len(plan.eval_indices)
        run_a.log_replicate(
            plan,
            metrics={"bs": 0.1},
            y_true=np.zeros(n, dtype=int),
            p=np.full(n, 0.3),
        )
    run_a.finish(summary={"mean_bs": 0.1})

    plans_b = bs.train_resample_holdout(
        train_idx=np.arange(8),
        val_idx=np.arange(8, 10),
        universe_size=10,
        n_replicates=2,
        bootstrap_seed=2,
    )
    run_b = bs.init(
        store,
        procedure="train_resample_holdout",
        dataset="alad",
        model_label="logreg",
        config={"rank": 1, "B": 2},
        tags=["fig3"],
    )
    for plan in plans_b:
        n = len(plan.eval_indices)
        run_b.log_replicate(
            plan,
            metrics={"bs": 0.1},
            y_true=np.zeros(n, dtype=int),
            p=np.full(n, 0.3),
        )
    run_b.finish(summary={"mean_bs": 0.1})

    return store, run_a, run_b, plans_a, plans_b


def test_query_runs_all(tmp_path):
    store, run_a, run_b, *_ = _build_store(tmp_path)
    runs = bs.query_runs(store)
    assert runs.height == 2
    for col in ["config.rank", "summary.mean_bs", "tags", "dataset"]:
        assert col in runs.columns


def test_query_runs_dataset_filter(tmp_path):
    store, run_a, run_b, *_ = _build_store(tmp_path)
    runs = bs.query_runs(store, {"dataset": "bal"})
    assert runs.height == 1
    assert runs["model_label"].to_list() == ["ULTRA"]


def test_query_runs_tags_all_required(tmp_path):
    store, run_a, run_b, *_ = _build_store(tmp_path)
    runs = bs.query_runs(store, {"tags": ["fig3", "pooling"]})
    assert runs.height == 1
    assert runs["run_id"].to_list() == [run_a.run_dir.name]


def test_query_runs_tags_single_str(tmp_path):
    store, *_ = _build_store(tmp_path)
    runs = bs.query_runs(store, {"tags": "fig3"})
    assert runs.height == 2


def test_query_runs_config_filter(tmp_path):
    store, run_a, *_ = _build_store(tmp_path)
    runs = bs.query_runs(store, {"config.rank": 4})
    assert runs.height == 1
    assert runs["run_id"].to_list() == [run_a.run_dir.name]


def test_query_runs_unknown_config_key(tmp_path):
    store, *_ = _build_store(tmp_path)
    runs = bs.query_runs(store, {"config.nonexistent": 1})
    assert runs.height == 0


def test_query_run_table_predictions(tmp_path):
    store, run_a, run_b, plans_a, plans_b = _build_store(tmp_path)
    preds = bs.query_run_table(store, run_a.run_dir.name, "predictions")
    expected = sum(len(p.eval_indices) for p in plans_a)
    assert preds.height == expected

    with pytest.raises(ValueError):
        bs.query_run_table(store, run_a.run_dir.name, "bogus")


def test_query_run_table_missing_run(tmp_path):
    store, *_ = _build_store(tmp_path)
    with pytest.raises(FileNotFoundError):
        bs.query_run_table(store, "does-not-exist", "replicates")


def test_load_runs_table(tmp_path):
    store, *_ = _build_store(tmp_path)
    table = bs.load_runs_table(store, {"tags": "fig3"}, "replicates")
    assert table.height == 4
    assert table["run_id"].n_unique() == 2
    for col in ["run_id", "dataset", "model_label"]:
        assert col in table.columns


def test_query_runs_empty_store(tmp_path):
    store = tmp_path / "empty_store"
    runs = bs.query_runs(store)
    assert runs.height == 0
