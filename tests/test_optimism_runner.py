from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from optimism_runner import main


def test_optimism_runner_end_to_end(tmp_path):
    store = tmp_path / "store"
    results = main(str(store), n=160, n_features=8, n_replicates=25)

    run_dirs = list((store / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "apparent.parquet").exists()

    auroc = results["auroc"]
    assert auroc["apparent"] > auroc["corrected"]
    assert auroc["optimism"] > 0

    err632 = results["error_632plus"]
    estimate = err632["estimate"]
    assert isinstance(estimate, float)
    assert 0.0 <= estimate <= 1.0
    assert err632["weight"] >= 0.632
