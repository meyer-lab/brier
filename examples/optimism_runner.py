"""Reference end-to-end runner for the Efron-Gong optimism correction path."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Allow running as a plain script (`uv run python examples/optimism_runner.py`)
    # without installing `bootstraptools` as an editable package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import bootstraptools as bs

RANDOM_SEED = 13
DATASET = "synthetic"
N_REPLICATES = 100


def make_data(n: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic binary-classification dataset."""
    X, y = make_classification(n_samples=n, n_features=n_features, random_state=seed)
    return X, y


def main(
    store: str,
    n: int = 400,
    n_features: int = 10,
    n_replicates: int = N_REPLICATES,
) -> dict:
    """Fit an intentionally overfit tree, bootstrap its optimism, and correct it."""
    seeds = bs.derive_seeds(RANDOM_SEED, ["dataset", "model", "bootstrap"])
    X, y = make_data(n, n_features, seeds["dataset"])

    # Apparent (in-sample) fit on ALL n rows — unrestricted tree, so it
    # overfits and the optimism correction is visibly large.
    apparent_model = DecisionTreeClassifier(random_state=seeds["model"]).fit(X, y)
    apparent_p = apparent_model.predict_proba(X)[:, 1]

    run = bs.init(
        store,
        procedure="optimism_bootstrap",
        dataset=DATASET,
        model_label="tree",
        config={"B": n_replicates, "bootstrap_seed": seeds["bootstrap"]},
        tags=["optimism", "demo"],
        store_model_state=False,
    )
    run.log_apparent(y_true=y, p=apparent_p)

    plans = bs.optimism_bootstrap(
        universe_size=n, n_replicates=n_replicates, bootstrap_seed=seeds["bootstrap"]
    )
    fit_seeds = bs.replicate_seeds(seeds["model"], n_replicates)

    for plan in plans:
        k = plan.replicate_idx
        Xf = bs.select(X, plan.fit_indices)
        yf = y[plan.fit_indices]
        model = DecisionTreeClassifier(random_state=fit_seeds[k]).fit(Xf, yf)
        p_all = model.predict_proba(X)[:, 1]  # predict on ALL n originals

        run.log_replicate(
            plan,
            metrics={"auroc_orig": float(roc_auc_score(y, p_all))},
            y_true=y,
            p=p_all,
            model_fit_seed=fit_seeds[k],
        )

    run.finish()

    # DOWNSTREAM ANALYSIS
    def auc_metric(yt, pp, sample_weight=None):
        return roc_auc_score(yt, pp, sample_weight=sample_weight)

    auc = bs.optimism_from_run(store, run.run_dir.name, auc_metric)
    brier = bs.optimism_from_run(store, run.run_dir.name, brier_score_loss)
    err632 = bs.error_632_plus_from_run(store, run.run_dir.name, bs.zero_one_loss)

    return {
        "store": store,
        "auroc": {
            "apparent": auc["apparent"],
            "optimism": auc["optimism"],
            "corrected": auc["corrected"],
        },
        "brier": {
            "apparent": brier["apparent"],
            "optimism": brier["optimism"],
            "corrected": brier["corrected"],
        },
        "error_632plus": err632,
    }


if __name__ == "__main__":
    import pprint

    pprint.pprint(main("./optimism_store"))
