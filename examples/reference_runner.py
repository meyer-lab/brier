from __future__ import annotations

import pprint
import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    # Allow running as a plain script (`uv run python examples/reference_runner.py`)
    # without installing `bootstraptools` as an editable package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import bootstraptools as bs

RANDOM_SEED = 13
DATASET = "synthetic"
N_REPLICATES = 100
RANK = 4  # config value to record


def make_data(n: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic binary-classification dataset."""
    X, y = make_classification(n_samples=n, n_features=n_features, random_state=seed)
    return X, y


def brier_score(p: np.ndarray, yv: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and outcomes."""
    return float(np.mean((p - yv) ** 2))


def main(
    store: str,
    n: int = 300,
    n_features: int = 20,
    n_replicates: int = N_REPLICATES,
) -> dict:
    """Run two paired bootstrap comparisons and return CI summaries."""
    seeds = bs.derive_seeds(RANDOM_SEED, ["dataset", "model", "bootstrap"])

    X, y = make_data(n, n_features, seeds["dataset"])

    train_idx, val_idx = train_test_split(
        np.arange(n), test_size=0.3, stratify=y, random_state=seeds["dataset"]
    )
    y_val = y[val_idx]

    plans = bs.train_resample_holdout(
        train_idx,
        val_idx,
        universe_size=n,
        n_replicates=n_replicates,
        bootstrap_seed=seeds["bootstrap"],
    )
    fit_seeds = bs.replicate_seeds(seeds["model"], n_replicates)

    shared_config = {
        "rank": RANK,
        "B": n_replicates,
        "bootstrap_seed": seeds["bootstrap"],
    }
    shared_kwargs = dict(
        procedure="train_resample_holdout",
        dataset=DATASET,
        tags=["reference", "fig3-demo"],
        config=shared_config,
    )

    run_model = bs.init(store, model_label="logreg", store_model_state=True, **shared_kwargs)
    run_base = bs.init(store, model_label="gnb", store_model_state=False, **shared_kwargs)

    runs = [
        (run_model, "logreg"),
        (run_base, "gnb"),
    ]
    skills: dict[str, list[float]] = {"logreg": [], "gnb": []}

    for plan in plans:
        Xf = bs.select(X, plan.fit_indices)
        yf = y[plan.fit_indices]
        k = plan.replicate_idx
        prevalence = float(yf.mean())
        p_ref = np.full(len(val_idx), prevalence)
        br = brier_score(p_ref, y_val)

        for run, label in runs:
            if label == "logreg":
                model_fit_seed = fit_seeds[k]
                model = LogisticRegression(max_iter=1000, random_state=model_fit_seed)
            else:
                model_fit_seed = None
                model = GaussianNB()

            model.fit(Xf, yf)
            p = model.predict_proba(X[val_idx])[:, 1]

            bm = brier_score(p, y_val)
            skill = 1.0 - bm / br if br != 0 else float("nan")
            skills[label].append(skill)

            arrays = None
            model_state = None
            if label == "logreg":
                coef = model.coef_.ravel()
                arrays = {"coef": coef}
                model_state = {"coef": coef, "intercept": model.intercept_.copy()}

            run.log_replicate(
                plan,
                metrics={"brier": bm, "brier_ref": br, "brier_skill": skill},
                y_true=y_val,
                p=p,
                arrays=arrays,
                model_state=model_state,
                model_fit_seed=model_fit_seed,
            )

    for run, label in runs:
        mean_skill = float(np.mean(skills[label]))
        run.finish(summary={"mean_brier_skill": mean_skill})

    # DOWNSTREAM ANALYSIS
    runs_df = bs.query_runs(store, {"tags": "reference"})
    tab = bs.load_runs_table(store, {"tags": "reference"}, "replicates")

    results: dict = {"runs": {}, "store": store}
    per_run_replicate_tables = {}
    per_run_membership = {}

    for run_id, model_label in zip(
        runs_df["run_id"].to_list(), runs_df["model_label"].to_list()
    ):
        run_tab = tab.filter(tab["run_id"] == run_id).sort("replicate_idx")
        per_run_replicate_tables[model_label] = run_tab
        theta_boot = run_tab["brier_skill"].to_numpy()

        # Point estimate: fit the SAME model class on the FULL train fold
        # (no resampling), evaluate skill on val.
        if model_label == "logreg":
            point_model = LogisticRegression(max_iter=1000, random_state=seeds["model"])
        else:
            point_model = GaussianNB()
        point_model.fit(X[train_idx], y[train_idx])
        p_hat = point_model.predict_proba(X[val_idx])[:, 1]
        prevalence_hat = float(y[train_idx].mean())
        p_ref_hat = np.full(len(val_idx), prevalence_hat)
        br_hat = brier_score(p_ref_hat, y_val)
        bm_hat = brier_score(p_hat, y_val)
        theta_hat = 1.0 - bm_hat / br_hat if br_hat != 0 else float("nan")

        membership = bs.membership_matrix(store, run_id)
        per_run_membership[model_label] = membership
        bca_ci = bs.bca(theta_boot, theta_hat, membership=membership)
        pct_ci = bs.percentile(theta_boot)

        results["runs"][model_label] = {
            "theta_hat": theta_hat,
            "bca": bca_ci,
            "percentile": pct_ci,
        }

    # PAIRED comparison enabled by the shared bootstrap seed.
    logreg_tab = per_run_replicate_tables["logreg"]
    gnb_tab = per_run_replicate_tables["gnb"]
    joined = logreg_tab.join(
        gnb_tab, on="replicate_idx", suffix="_gnb"
    )
    diff = (joined["brier_skill"] - joined["brier_skill_gnb"]).to_numpy()
    diff_ci = bs.percentile(diff)
    diff_bca = bs.bca(diff, float(diff.mean()), membership=per_run_membership["logreg"])

    results["paired_diff"] = {
        "mean": float(diff.mean()),
        "percentile": diff_ci,
        "bca": diff_bca,
    }

    return results


if __name__ == "__main__":
    out = main("./bootstrap_store")
    pprint.pprint(out)
