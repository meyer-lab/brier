from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from double_bootstrap_runner import main


def test_double_bootstrap_runner_end_to_end(tmp_path):
    store = tmp_path / "store"
    results = main(str(store), n=150, n_features=6, n_outer=8, n_inner=8)

    run_dirs = list((store / "runs").iterdir())
    assert len(run_dirs) == 2

    auroc = results["auroc"]
    lo, hi = auroc["ci"]
    assert lo <= hi
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
    assert auroc["n_outer"] == 8
    assert isinstance(auroc["point_corrected"], float)
    assert 0.0 <= auroc["point_corrected"] <= 1.0

    brier = results["brier"]
    lo_b, hi_b = brier["ci"]
    assert lo_b <= hi_b
    assert 0.0 <= lo_b <= 1.0
    assert 0.0 <= hi_b <= 1.0
    assert brier["n_outer"] == 8
    assert isinstance(brier["point_corrected"], float)
    assert 0.0 <= brier["point_corrected"] <= 1.0
