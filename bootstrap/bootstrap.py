from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from .schemas import ReplicateResult, _make_failed_metadata
from .types import BootstrapIndices, HparamResolver, ModelWrapper

if TYPE_CHECKING:
    from collections.abc import Callable


def wrap_model(
        model: Any,
        label: str,
        hparam_resolver: HparamResolver | None = None,
) -> ModelWrapper:
    """Wrap a user-provided model into the bootstrap framework's expected interface.

    Accepts either:
        - An sklearn-like estimator with .fit(X, y) and .predict_proba(X) methods
        - A tuple of (fit_fn, predict_proba_fn) callables with signatures:
            fit_fn(X, y) -> fitted_model
            predict_proba_fn(fitted_model, X) -> np.ndarray of shape (n_samples, 2)
    Parameters:
    model: Any
        The model or callable pair.
    label: str
        A string label for this model to be recorded in metadata.
    hparam_resolver: HparamResolver | None
        Optional function for hyperparameter tuning within each replicate.

    Returns:
    ModelWrapper
    """
    # If model is an sklearn-like estimator/object with fit/predict_proba
    if hasattr(model, "fit") and hasattr(model, "predict_proba"):
        fit_fn: Callable[[Any, np.ndarray], Any] = lambda X, y: model.fit(X, y) or model
        predict_proba_fn: Callable[[Any, np.ndarray], np.ndarray] = lambda fitted, X: fitted.predict_proba(X)
        return ModelWrapper(label=label,fit_fn=fit_fn,predict_proba_fn=predict_proba_fn,hparam_resolver=hparam_resolver)

    # If user passes a tuple of (fit_fn, predict_proba_fn) instead of model
    if isinstance(model, tuple) and len(model) == 2:
        fit_fn, predict_proba_fn = model
        if callable(fit_fn) and callable(predict_proba_fn):
            return ModelWrapper(label=label,fit_fn=fit_fn,predict_proba_fn=predict_proba_fn,hparam_resolver=hparam_resolver)

    raise TypeError(
        f"Expected an object with .fit/.predict_proba methods or a tuple of equivalent callables"
        f"got {type(model).__name__}. If passing callables, signature must be: "
        f"fit(X, y) -> fitted, predict_proba(fitted, X) -> np.ndarray of shape (n_samples, 2)"
    )

def generate_replicates(
        n_samples: int,
        n_replicates: int,
        random_seed: int = 13,
) -> list[BootstrapIndices]:
    """Generate bootstrap replicates with in-bag and out-of-bag index splits.

    Parameters:
    n_samples: int
        Total number of samples in the dataset (patients typically).
    n_replicates: int
        Number of bootstrap replicates.
    random_seed: int
        Deterministic seed for reproducibility.

    Returns:
    list[BootstrapIndices]
        One entry per replicates with in-bag and OOB index arrays.
    """
    rng = np.random.default_rng(random_seed)
    # Draw all in-bag indices in one shot to generate (n_replicates, n_samples_) array
    # We use discrete uniform sampling for nonparametric bootstrap sampling with replacement
    inbag_matrix = rng.integers(0, n_samples, size=(n_replicates, n_samples))

    # We use a boolean seen matrix to identify OOB samples for each replicate
    seen = np.zeros((n_replicates, n_samples), dtype=bool)
    row_idx = np.arange(n_replicates)[:, np.newaxis] # (n_replicates, 1)
    seen[row_idx, inbag_matrix] = True

    replicates: list[BootstrapIndices] = []
    for k in range(n_replicates):
        oob = np.where(~seen[k])[0]
        replicates.append(
            BootstrapIndices(
                replicate_idx=k,
                inbag=inbag_matrix[k],
                oob=oob,
                n_inbag_unique=int(np.sum(seen[k])),
            )
        )
    return replicates


def run_replicate(
    X: Any,
    y: np.ndarray,
    indices: BootstrapIndices,
    model: ModelWrapper,
    seed: int,
    collect_inbag: bool = False,
) -> ReplicateResult:
    """Execute one bootstrap replicate to resolve hparams (if applicable), fit, predict, collect.
 
    Parameters:
    X: Any
        Full feature matrix/tensor. Expected to support indexing with integer arrays so may need to convert prior (e.g., np.asarray(X)).
    y: np.ndarray
        Full label vector.
    indices: BootstrapIndices
        In-bag and OOB index sets for this replicate.
    model: ModelWrapper
        Wrapped model with fit/predict callables.
    seed: int
        Root random seed (recorded in metadata, used for RNG in orchestrator).
    collect_inbag : bool
        If True, also predict on unique in-bag patients.

    Returns:
    ReplicateResult
        Metadata dict and list of probability row-dicts.
    """
    k = indices.replicate_idx

    # In case of empty OOB set
    if len(indices.oob) == 0:
        meta = _make_failed_metadata(
            indices, y, model, seed,
            status="skipped_empty_oob",
            error_msg="OOB set is empty for this replicate",
        )
        return ReplicateResult(metadata=meta, probabilities=[])

    # Slice data to obtain in-bag and OOB sets for this replicate
    unique_inbag = np.unique(indices.inbag)
    X_inbag, y_inbag = X[indices.inbag], y[indices.inbag]
    X_oob, y_oob = X[indices.oob], y[indices.oob]

    cv_duration_s: float | None = None
    fit_duration_s: float | None = None
    hparams_dict: dict | None = None

    # Optional hyperparameter tuning step via model's hparam_resolver.
    # This is expected to be a user-defined function that takes in the in-bag data.
    # Returns a new model instance with resolved hyperparameters, along with a dict of those hyperparameters for metadata recording.
    # Example shown in analysis/example_usage.py.
    if model.hparam_resolver is not None:
        try:
            t0 = time.perf_counter()
            resolved_model, hparams_dict = model.hparam_resolver(X_inbag, y_inbag, None)
            cv_duration_s = time.perf_counter() - t0
        except Exception as exc:
            meta = _make_failed_metadata(
                indices, y, model, seed,
                status="failed",
                error_msg=f"hparam_resolver raised {type(exc).__name__}: {exc}",
                cv_duration_s=time.perf_counter() - t0,
            )
            return ReplicateResult(metadata=meta, probabilities=[])
 
        # If resolver returned a new model, re-wrap fit/predict to use it
        # The resolver is expected to return a ready-to-fit object
        if resolved_model is not None and hasattr(resolved_model, "fit"):
            fit_fn = lambda Xt, yt, m=resolved_model: m.fit(Xt, yt) or m
            predict_fn = lambda fitted, Xt: fitted.predict_proba(Xt)
        else:
            fit_fn = model.fit_fn
            predict_fn = model.predict_proba_fn
    else:
        fit_fn = model.fit_fn
        predict_fn = model.predict_proba_fn

    # Fit model on in-bag data and predict on OOB data
    try:
        t0 = time.perf_counter()
        fitted_obj = fit_fn(X_inbag, y_inbag)
        fit_duration_s = time.perf_counter() - t0
    except Exception as exc:
        meta = _make_failed_metadata(
            indices, y, model, seed,
            status="failed",
            error_msg=f"fit raised {type(exc).__name__}: {exc}",
            cv_duration_s=cv_duration_s,
        )
        return ReplicateResult(metadata=meta, probabilities=[])

    try:
        oob_probs = predict_fn(fitted_obj, X_oob)  # (n_oob, 2)
    except Exception as exc:
        meta = _make_failed_metadata(
            indices, y, model, seed,
            status="failed",
            error_msg=f"predict_proba (OOB) raised {type(exc).__name__}: {exc}",
            cv_duration_s=cv_duration_s,
            fit_duration_s=fit_duration_s,
        )
        return ReplicateResult(metadata=meta, probabilities=[])

    # Validate on first success
    model.validate_proba(oob_probs, n_samples=len(indices.oob))

    # Predict in-bag (optional, on unique patients only)
    inbag_probs: np.ndarray | None = None
    if collect_inbag:
        try:
            inbag_probs = predict_fn(fitted_obj, X[unique_inbag])  # (n_unique_inbag, 2)
        except Exception as exc:
            inbag_probs = None

    metadata = {
        "replicate_idx": k,
        "seed": seed,
        "model_label": model.label,
        "n_inbag_draws": len(indices.inbag),
        "n_inbag_unique": indices.n_inbag_unique,
        "n_oob": len(indices.oob),
        "n_pos_inbag": int(np.sum(y[unique_inbag])),
        "n_pos_oob": int(np.sum(y_oob)),
        "cv_duration_s": cv_duration_s,
        "fit_duration_s": fit_duration_s,
        "hparams": json.dumps(hparams_dict) if hparams_dict is not None else None,
        "status": "success",
        "error_msg": None,
    }

    prob_rows: list[dict[str, Any]] = []

    for i in range(len(indices.oob)):
        prob_rows.append({
            "replicate_idx": k,
            "sample_idx": int(indices.oob[i]),
            "bag": "oob",
            "y_true": int(y_oob[i]),
            "prob_0": float(oob_probs[i, 0]),
            "prob_1": float(oob_probs[i, 1]),
        })

    if collect_inbag and inbag_probs is not None:
        y_unique_inbag = y[unique_inbag]
        for i in range(len(unique_inbag)):
            prob_rows.append({
                "replicate_idx": k,
                "sample_idx": int(unique_inbag[i]),
                "bag": "inbag",
                "y_true": int(y_unique_inbag[i]),
                "prob_0": float(inbag_probs[i, 0]),
                "prob_1": float(inbag_probs[i, 1]),
            })

    return ReplicateResult(metadata=metadata, probabilities=prob_rows)
