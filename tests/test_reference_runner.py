from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

import bootstraptools as bs
from reference_runner import main


def test_reference_runner_end_to_end(tmp_path):
    store = tmp_path / "store"
    result = main(str(store), n=120, n_features=10, n_replicates=25)

    runs_df = bs.query_runs(store, {"tags": "reference"})
    assert runs_df.height == 2
    assert set(runs_df["model_label"].to_list()) == {"logreg", "gnb"}

    run_ids = dict(
        zip(runs_df["model_label"].to_list(), runs_df["run_id"].to_list())
    )

    for label, run_id in run_ids.items():
        rep = bs.query_run_table(store, run_id, "replicates")
        assert rep.height == 25
        assert "brier_skill" in rep.columns

    logreg_run_id = run_ids["logreg"]
    gnb_run_id = run_ids["gnb"]

    assert (store / "runs" / logreg_run_id / "model_state" / "replicate_0000.npz").is_file()
    import numpy as np

    npz = np.load(store / "runs" / logreg_run_id / "model_state" / "replicate_0000.npz")
    assert "coef" in npz

    assert not (store / "runs" / gnb_run_id / "model_state").exists()

    for label in ("logreg", "gnb"):
        low, high = result["runs"][label]["bca"]
        assert low < high

    diff_low, diff_high = result["paired_diff"]["percentile"]
    assert diff_low < diff_high

    membership = bs.membership_matrix(store, logreg_run_id)
    assert membership.shape == (25, 120)
