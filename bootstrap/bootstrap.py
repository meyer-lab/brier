from __future__ import annotations
 
import json
import sys
import time
import traceback
from typing import TYPE_CHECKING, Any
 
import numpy as np
 
from .schemas import ReplicateResult
from .types import AttrSpec, BootstrapIndices, CriticalExtractorError, HparamResolver, ModelWrapper, TelemetrySpec
from .util import _flatten_dict, _make_failed_metadata, _classify_return
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
        def fit_fn(X: Any, y: np.ndarray) -> Any:
            return model.fit(X, y) or model
        def predict_proba_fn(fitted: Any, X: np.ndarray) -> np.ndarray:
            return fitted.predict_proba(X)
        return ModelWrapper(
            label=label,
            fit_fn=fit_fn,
            predict_proba_fn=predict_proba_fn,
            hparam_resolver=hparam_resolver,
        )

    # If user passes a tuple of (fit_fn, predict_proba_fn) instead of model
    if isinstance(model, tuple) and len(model) == 2:
        fit_fn, predict_proba_fn = model
        if callable(fit_fn) and callable(predict_proba_fn):
            return ModelWrapper(
                label=label,
                fit_fn=fit_fn,
                predict_proba_fn=predict_proba_fn,
                hparam_resolver=hparam_resolver,
            )

    raise TypeError(
        f"Expected an object with .fit/.predict_proba methods or a tuple of "
        f"equivalent callables, got {type(model).__name__}. If passing callables, "
        f"signature must be: fit(X, y) -> fitted, predict_proba(fitted, X) -> "
        f"np.ndarray of shape (n_samples, 2)"
    )


def generate_replicates(
        n_samples: int,
        n_replicates: int,
        random_seed: int = 13,
) -> list[BootstrapIndices]:
    """Generate bootstrap replicates with in-bag and out-of-bag index splits.

    Parameters:
    n_samples: int
        Total number of samples in the dataset.
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
    attribute_extractors: dict[str, AttrSpec] | None = None,
    telemetry_extractors: dict[str, TelemetrySpec] | None = None,
) -> ReplicateResult:
    """Execute one bootstrap replicate.

    Execution order:
        1. Resolve hparams (if user-specified)
        2. Fit model
        3. Extract attributes (if user-specified)
        4. predict_proba on OOB (+ inbag if flagged)
        5. Run telemetry extractors on OOB (+ in bag if flagged, user-specified)
 
    Parameters:
    X: Any
        Full feature matrix/tensor. 
    y: np.ndarray
        Full label vector.
    indices: BootstrapIndices
        In-bag and OOB index sets for this replicate.
    model: ModelWrapper
        Wrapped model with fit/predict callables.
    seed: int
        Root random seed (recorded in metadata, used for RNG in orchestrator).
    collect_inbag : bool
        If True, also predict on unique in-bag samples.
    attribute_extractors : dict[str, AttrSpec] or None
        User-defined attribute extractors.
    telemetry_extractors : dict[str, TelemetrySpec] or None
        User-defined telemetry extractors.

    Returns:
    ReplicateResult
    """
    k = indices.replicate_idx
    attribute_extractors = attribute_extractors or {}
    telemetry_extractors = telemetry_extractors or {}

    # In case of empty OOB set
    if len(indices.oob) == 0:
        meta = _make_failed_metadata(
            indices, y, model, seed,
            status="skipped_empty_oob",
            error_msg="OOB set is empty for this replicate",
        )
        return ReplicateResult(metadata=meta, output_rows=[], attributes={}, telemetry={})

    # Slice data to obtain in-bag and OOB sets for this replicate
    unique_inbag = np.unique(indices.inbag)
    X_inbag, y_inbag = X[indices.inbag], y[indices.inbag]
    X_oob, y_oob = X[indices.oob], y[indices.oob]
 
    cv_duration_s: float | None = None
    fit_duration_s: float | None = None
    hparams_dict: dict | None = None
    all_warnings: list[str] = []


    # 1. Resolve hparams
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
            return ReplicateResult(metadata=meta, output_rows=[], attributes={}, telemetry={})
 
        if resolved_model is not None and hasattr(resolved_model, "fit"):
            def fit_fn(Xt, yt, m=resolved_model):
                return m.fit(Xt, yt) or m
            def predict_fn(fitted, Xt):
                return fitted.predict_proba(Xt)
        else:
            fit_fn = model.fit_fn
            predict_fn = model.predict_proba_fn
    else:
        fit_fn = model.fit_fn
        predict_fn = model.predict_proba_fn


    # 2. Fit model 
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
        return ReplicateResult(metadata=meta, output_rows=[], attributes={}, telemetry={})


    # 3. Extract attributes (includes hparams from resolver)
    combined_attr_extractors = dict(attribute_extractors)
    # Hparams synthetic extractor
    if hparams_dict is not None:
        combined_attr_extractors["hparams"] = AttrSpec(
            fn=lambda _fitted, _hp=hparams_dict: _hp,
            critical=False,
            ragged=False,
        )
    extracted_attributes: dict[str, Any] = {}
    if combined_attr_extractors:
        try:
            extracted_attributes, attr_warnings = _extract_attributes(
                fitted_obj, combined_attr_extractors
            )
            all_warnings.extend(attr_warnings)
        except CriticalExtractorError as exc:
            meta = _make_failed_metadata(
                indices, y, model, seed,
                status="failed",
                error_msg=str(exc),
                cv_duration_s=cv_duration_s,
                fit_duration_s=fit_duration_s,
            )
            print(
                f"Replicate {k}: critical attribute extractor failed\n"
                f"{traceback.format_exc()}",
                file=sys.stderr, flush=True,
            )
            return ReplicateResult(
                metadata=meta, output_rows=[], attributes={}, telemetry={},
            )
    

    # 4. Predict on OOB (and in-bag if flagged)
    try:
        oob_probs = predict_fn(fitted_obj, X_oob)
    except Exception as exc:
        meta = _make_failed_metadata(
            indices, y, model, seed,
            status="failed",
            error_msg=f"predict_proba (OOB) raised {type(exc).__name__}: {exc}",
            cv_duration_s=cv_duration_s,
            fit_duration_s=fit_duration_s,
        )
        return ReplicateResult(
            metadata=meta, output_rows=[], attributes=extracted_attributes, telemetry={},
        )
    model.validate_proba(oob_probs, n_samples=len(indices.oob))
    inbag_probs: np.ndarray | None = None
    if collect_inbag:
        try:
            inbag_probs = predict_fn(fitted_obj, X[unique_inbag])
        except Exception:
            inbag_probs = None


    # 5. Extract telemetry on OOB (and in-bag if flagged)
    telemetry_results: dict[str, Any] = {}
    if telemetry_extractors:
        try:
            oob_telemetry, telem_warnings = _extract_telemetry(
                fitted_obj, X_oob, y_oob, telemetry_extractors,
                n_samples=len(indices.oob), split_label="oob",
            )
            all_warnings.extend(telem_warnings)
        except CriticalExtractorError as exc:
            meta = _make_failed_metadata(
                indices, y, model, seed,
                status="failed",
                error_msg=str(exc),
                cv_duration_s=cv_duration_s,
                fit_duration_s=fit_duration_s,
            )
            print(
                f"Replicate {k}: critical telemetry extractor failed\n"
                f"{traceback.format_exc()}",
                file=sys.stderr, flush=True,
            )
            return ReplicateResult(
                metadata=meta, output_rows=[], attributes=extracted_attributes, telemetry={},
            )
        inbag_telemetry: dict[str, Any] | None = None
        if collect_inbag and inbag_probs is not None:
            try:
                y_unique_inbag = y[unique_inbag]
                inbag_telemetry, telem_warnings_ib = _extract_telemetry(
                    fitted_obj, X[unique_inbag], y_unique_inbag,
                    telemetry_extractors,
                    n_samples=len(unique_inbag), split_label="inbag",
                )
                all_warnings.extend(telem_warnings_ib)
            except CriticalExtractorError as exc:
                meta = _make_failed_metadata(
                    indices, y, model, seed,
                    status="failed",
                    error_msg=str(exc),
                    cv_duration_s=cv_duration_s,
                    fit_duration_s=fit_duration_s,
                )
                print(
                    f"Replicate {k}: critical telemetry extractor failed (inbag)\n"
                    f"{traceback.format_exc()}",
                    file=sys.stderr, flush=True,
                )
                return ReplicateResult(
                    metadata=meta, output_rows=[], attributes=extracted_attributes, telemetry={},
                )
        # Concatenate OOB + inbag telemetry in row order
        for name in oob_telemetry:
            oob_val = oob_telemetry[name]
            inbag_val = inbag_telemetry.get(name) if inbag_telemetry else None
            if oob_val is None and inbag_val is None:
                telemetry_results[name] = None
            elif oob_val is None or inbag_val is None:
                if inbag_val is None:
                    telemetry_results[name] = oob_val
                else:
                    telemetry_results[name] = inbag_val
            else:
                if isinstance(oob_val, np.ndarray) and isinstance(inbag_val, np.ndarray):
                    telemetry_results[name] = np.concatenate([oob_val, inbag_val], axis=0)
                elif isinstance(oob_val, list) and isinstance(inbag_val, list):
                    telemetry_results[name] = oob_val + inbag_val
                else:
                    telemetry_results[name] = oob_val 
    else:
        oob_telemetry = {}
        inbag_telemetry = None
    
    # 6. Construct metadata and output rows for this replicate
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
        "status": "success",
        "error_msg": "; ".join(all_warnings) if all_warnings else None,
    }
 
    output_rows: list[dict[str, Any]] = []

    for i in range(len(indices.oob)):
        output_rows.append({
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
            output_rows.append({
                "replicate_idx": k,
                "sample_idx": int(unique_inbag[i]),
                "bag": "inbag",
                "y_true": int(y_unique_inbag[i]),
                "prob_0": float(inbag_probs[i, 0]),
                "prob_1": float(inbag_probs[i, 1]),
            })

    return ReplicateResult(
        metadata=metadata,
        output_rows=output_rows,
        attributes=extracted_attributes,
        telemetry=telemetry_results,
    )


# Run all attribute extractors against a fitted model. Executes critical extractors first for fail-fast behavior.
def _extract_attributes(
    fitted_obj: Any,
    extractors: dict[str, AttrSpec],
) -> tuple[dict[str, Any], list[str]]:
    attributes: dict[str, Any] = {}
    warnings: list[str] = []
 
    # Sort by loud fails first (critical=True)
    sorted_names = sorted(extractors.keys(), key=lambda n: (not extractors[n].critical, n))
 
    for name in sorted_names:
        spec = extractors[name]
        try:
            value = spec.fn(fitted_obj)
        except Exception as exc:
            if spec.critical:
                raise CriticalExtractorError(
                    f"Critical attribute extractor '{name}' raised "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            warnings.append(
                f"Attribute extractor '{name}' raised {type(exc).__name__}: {exc}"
            )
            attributes[name] = None
            continue
 
        if isinstance(value, dict):
            flat = _flatten_dict(name, value)
            attributes.update(flat)
        else:
            attributes[name] = value
 
    return attributes, warnings


# Run all telemetry extractors on a data split. Validates output shape and sample count.
def _extract_telemetry(
    fitted_obj: Any,
    X_split: Any,
    y_split: np.ndarray,
    extractors: dict[str, TelemetrySpec],
    n_samples: int,
    split_label: str,
) -> tuple[dict[str, Any], list[str]]:
    results: dict[str, Any] = {}
    warnings: list[str] = []
 
    # Sort by loud fails first (critical=True)
    sorted_names = sorted(extractors.keys(), key=lambda n: (not extractors[n].critical, n))
 
    for name in sorted_names:
        spec = extractors[name]
        try:
            value = spec.fn(fitted_obj, X_split, y_split)
        except Exception as exc:
            if spec.critical:
                raise CriticalExtractorError(
                    f"Critical telemetry extractor '{name}' raised "
                    f"{type(exc).__name__}: {exc} (split={split_label})"
                ) from exc
            warnings.append(
                f"Telemetry extractor '{name}' raised {type(exc).__name__}: "
                f"{exc} (split={split_label})"
            )
            results[name] = None
            continue
 
        # Validate first axis matches sample count
        if value is not None:
            ret_type = _classify_return(value)
 
            if ret_type == "ragged":
                # Object array or list of arrays
                if isinstance(value, np.ndarray) and not spec.ragged:
                    raise CriticalExtractorError(
                        f"Telemetry extractor '{name}' returned ragged output "
                        f"(object array) but ragged=False. Set ragged=True in "
                        f"TelemetrySpec to allow variable-length returns."
                    )
                length = len(value)
            elif isinstance(value, np.ndarray):
                length = value.shape[0]
            else:
                raise CriticalExtractorError(
                    f"Telemetry extractor '{name}' must return an ndarray, "
                    f"got {type(value).__name__}"
                )
 
            if length != n_samples:
                raise CriticalExtractorError(
                    f"Telemetry extractor '{name}' returned {length} rows, "
                    f"expected {n_samples} (split={split_label})"
                )
 
        results[name] = value
 
    return results, warnings