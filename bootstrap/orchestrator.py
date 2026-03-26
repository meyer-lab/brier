from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import polars as pl

from .bootstrap import generate_replicates, run_replicate, wrap_model
from .schemas import METADATA_SCHEMA, PROBABILITY_SCHEMA, BootstrapResult

if TYPE_CHECKING:
    import numpy as np

    from .types import HparamResolver


def run(
        X: Any,
        y: np.ndarray,
        model: Any,
        model_label: str,
        n_replicates: int = 100,
        random_seed: int = 13,
        hparam_resolver: HparamResolver | None = None,
        collect_inbag: bool = False,
) -> BootstrapResult:
    """
    Run bootstrap evaluation of a binary probabilistic classifier.
    Generates bootstrap replicates, wraps the model, runs each replicate sequentially, and assembles results into a Polars DataFrame.

    Parameters:
    X: Any
        Feature matrix, shape (n_samples, n_features). Can be any type accepted by the model's methods.
    y: np.ndarray
        Binary label vector, shape (n_samples,). 
    model: Any
        The model to evaluate. Can be an sklearn-like estimator with .fit and .predict_proba methods, or a tuple of (fit_fn, predict_proba_fn) callables.
    model_label: str
        A string label for the model, recorded in metadata.
    n_replicates: int
        Number of bootstrap replicates.
    random_seed: int
        Random seed for reproducibility.
    hparam_resolver: HparamResolver | None
        Optional function for hyperparameter tuning within each replicate.
    collect_inbag: bool
        Whether to collect predictions on unique in-bag samples.

    Returns:
    BootstrapResult
        Contains .metadata and .probabilities DataFrames
    """
    replicates = generate_replicates(n_samples=X.shape[0], n_replicates=n_replicates, random_seed=random_seed)
    wrapped = wrap_model(model=model, label=model_label, hparam_resolver=hparam_resolver)

    metadata_rows: list[dict] = []
    prob_rows: list[dict] = []

    for k, indices in enumerate(replicates):
        try:
            result = run_replicate(
                X=X,
                y=y,
                indices=indices,
                model=wrapped,
                seed=random_seed,
                collect_inbag=collect_inbag
                )
            metadata_rows.append(result.metadata)
            prob_rows.extend(result.probabilities)
        except Exception as exc:
            metadata_rows.append({
                "replicate_idx": indices.replicate_idx,
                "seed": random_seed,
                "model_label": model_label,
                "n_inbag_draws": len(indices.inbag),
                "n_inbag_unique": indices.n_inbag_unique,
                "n_oob": len(indices.oob),
                "n_pos_inbag": None,
                "n_pos_oob": None,
                "cv_duration_s": None,
                "fit_duration_s": None,
                "hparams": None,
                "status": "failed",
                "error_msg": f"Unhandled {type(exc).__name__}: {exc}",
            })

        if (k+1) % 10 == 0 or (k+1) == n_replicates:
            print(
                f"Replicate {k + 1}/{n_replicates} complete",
                file=sys.stderr,
                flush=True,
            )
    metadata_df = pl.DataFrame(metadata_rows, schema=METADATA_SCHEMA)
    probabilities_df = pl.DataFrame(prob_rows, schema=PROBABILITY_SCHEMA)

    return BootstrapResult(metadata=metadata_df, probabilities=probabilities_df)