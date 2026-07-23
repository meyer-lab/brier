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
MIN_SAMPLES_LEAF = 5


def make_data(
    n: int, n_features: int, seed: int, noise: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic binary-classification dataset with label noise."""
    X, y = make_classification(n_samples=n, n_features=n_features, random_state=seed)
    rng = np.random.default_rng(seed)
    flip = rng.random(n) < noise
    y = y.copy()
    y[flip] = 1 - y[flip]
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

    # Apparent (in-sample) fit on all n rows. `min_samples_leaf` stops the
    # tree from isolating every (noisy) point into its own pure leaf, so the
    # apparent-on-resample score is < 1.0 and varies across replicates.
    apparent_model = DecisionTreeClassifier(
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seeds["model"]
    ).fit(X, y)
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
        model = DecisionTreeClassifier(
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=fit_seeds[k]
        ).fit(Xf, yf)
        p_all = model.predict_proba(X)[:, 1]  # predict on ALL n originals

        run.log_replicate(
            plan,
            metrics={"auroc_orig": float(roc_auc_score(y, p_all))},
            y_true=y,
            p=p_all,
            model_fit_seed=fit_seeds[k],
        )

    run.finish()

    def auc_metric(yt, pp, sample_weight=None):
        return roc_auc_score(yt, pp, sample_weight=sample_weight)

    auc = bs.optimism_ci_from_run(store, run.run_dir.name, auc_metric)
    brier = bs.optimism_ci_from_run(store, run.run_dir.name, brier_score_loss)
    err632 = bs.error_632_plus_from_run(store, run.run_dir.name, bs.zero_one_loss)

    return {
        "store": store,
        "auroc": {
            "apparent": auc["apparent"],
            "optimism": auc["optimism"],
            "corrected": auc["corrected"],
            "ci": auc["ci"],
        },
        "brier": {
            "apparent": brier["apparent"],
            "optimism": brier["optimism"],
            "corrected": brier["corrected"],
            "ci": brier["ci"],
        },
        "error_632plus": err632,
    }


if __name__ == "__main__":
    import pprint

    pprint.pprint(main("./optimism_store"))
