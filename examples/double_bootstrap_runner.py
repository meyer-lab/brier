from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Allow running as a plain script (`uv run python examples/double_bootstrap_runner.py`)
    # without installing `bootstraptools` as an editable package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

import bootstraptools as bs

RANDOM_SEED = 17
DATASET = "synthetic"
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


def auc_metric(yt, pp, sample_weight=None):
    return roc_auc_score(yt, pp, sample_weight=sample_weight)


def main(
    store: str,
    n: int = 300,
    n_features: int = 8,
    n_outer: int = 40,
    n_inner: int = 40,
) -> dict:
    """Run the paired point-estimate + double-bootstrap-CI pattern end to end."""
    seeds = bs.derive_seeds(RANDOM_SEED, ["dataset", "model", "bootstrap"])
    X, y = make_data(n, n_features, seeds["dataset"])

    # ------------------------------------------------------------------
    # 1) POINT-ESTIMATE run: a standard `optimism_bootstrap` run on the
    #    ORIGINAL data. This is the Efron-Gong corrected estimate that
    #    method 2 reports as the point estimate (Noma 2021).
    # ------------------------------------------------------------------
    apparent_model = DecisionTreeClassifier(
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seeds["model"]
    ).fit(X, y)
    apparent_p = apparent_model.predict_proba(X)[:, 1]

    point_run = bs.init(
        store,
        procedure="optimism_bootstrap",
        dataset=DATASET,
        model_label="tree",
        config={"n_replicates": n_inner, "bootstrap_seed": seeds["bootstrap"]},
        tags=["optimism", "double_bootstrap_point", "demo"],
        store_model_state=False,
    )
    point_run.log_apparent(y_true=y, p=apparent_p)

    point_plans = bs.optimism_bootstrap(
        universe_size=n, n_replicates=n_inner, bootstrap_seed=seeds["bootstrap"]
    )
    point_fit_seeds = bs.replicate_seeds(seeds["model"], n_inner)

    for plan in point_plans:
        k = plan.replicate_idx
        Xf = bs.select(X, plan.fit_indices)
        yf = y[plan.fit_indices]
        model = DecisionTreeClassifier(
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=point_fit_seeds[k]
        ).fit(Xf, yf)
        p_all = model.predict_proba(X)[:, 1]  # predict on ALL n originals
        point_run.log_replicate(
            plan,
            y_true=y,
            p=p_all,
            model_fit_seed=point_fit_seeds[k],
        )

    point_run.finish()
    point_run_id = point_run.run_dir.name

    point_auc = bs.optimism_from_run(store, point_run_id, auc_metric)["corrected"]
    point_brier = bs.optimism_from_run(store, point_run_id, brier_score_loss)["corrected"]

    # ------------------------------------------------------------------
    # 2) CI run: the nested `double_optimism_bootstrap` loop. For each
    #    outer replicate we re-run a full optimism correction treating
    #    the outer resample as "the data", storing only the per-outer
    #    corrected metrics (no predictions).
    # ------------------------------------------------------------------
    ci_run = bs.init(
        store,
        procedure="double_optimism_bootstrap",
        dataset=DATASET,
        model_label="tree",
        config={
            "n_outer": n_outer,
            "n_inner": n_inner,
            "bootstrap_seed": seeds["bootstrap"],
        },
        tags=["double_optimism_bootstrap", "demo"],
        store_model_state=False,
    )

    double_plans = bs.double_optimism_bootstrap(
        universe_size=n, n_outer=n_outer, n_inner=n_inner, bootstrap_seed=seeds["bootstrap"]
    )
    outer_fit_seeds = bs.replicate_seeds(seeds["model"], n_outer)

    for dp in double_plans:
        r = dp.outer_idx
        outer_idx = dp.outer_plan.fit_indices  # GLOBAL indices, length n
        X_out = bs.select(X, outer_idx)
        y_out = y[outer_idx]

        # Apparent fit on the outer resample (treated as "the data").
        outer_model = DecisionTreeClassifier(
            min_samples_leaf=MIN_SAMPLES_LEAF, random_state=outer_fit_seeds[r]
        ).fit(X_out, y_out)
        app_p = outer_model.predict_proba(X_out)[:, 1]

        inner_fit_seeds = bs.replicate_seeds(outer_fit_seeds[r], n_inner)
        inner_ps = np.empty((n_inner, n), dtype=float)
        inner_counts = np.empty((n_inner, n), dtype=int)

        for ip in dp.inner_plans:
            k = ip.replicate_idx
            # `ip.fit_indices`/`.eval_indices` are LOCAL positions [0, n)
            # into the outer resample X_out/y_out.
            Xf = bs.select(X_out, ip.fit_indices)
            yf = y_out[ip.fit_indices]
            inner_model = DecisionTreeClassifier(
                min_samples_leaf=MIN_SAMPLES_LEAF, random_state=inner_fit_seeds[k]
            ).fit(Xf, yf)
            inner_ps[k] = inner_model.predict_proba(X_out)[:, 1]  # predict on ALL of X_out
            inner_counts[k] = ip.fit_counts

        theta_corr_auc = bs.optimism_correction(
            y_out, app_p, inner_ps, inner_counts, auc_metric
        )["corrected"]
        theta_corr_brier = bs.optimism_correction(
            y_out, app_p, inner_ps, inner_counts, brier_score_loss
        )["corrected"]

        ci_run.log_replicate(
            dp.outer_plan,
            metrics={
                "theta_corr_auroc": theta_corr_auc,
                "theta_corr_brier": theta_corr_brier,
            },
        )

    ci_run.finish()
    ci_run_id = ci_run.run_dir.name

    ci_auc = bs.double_bootstrap_ci_from_run(store, ci_run_id, col="theta_corr_auroc")
    ci_brier = bs.double_bootstrap_ci_from_run(store, ci_run_id, col="theta_corr_brier")

    return {
        "store": store,
        "auroc": {
            "point_corrected": point_auc,
            "ci": ci_auc["ci"],
            "n_outer": ci_auc["n_outer"],
        },
        "brier": {
            "point_corrected": point_brier,
            "ci": ci_brier["ci"],
            "n_outer": ci_brier["n_outer"],
        },
    }


if __name__ == "__main__":
    from pprint import pprint

    pprint(main("./double_bootstrap_store"))
