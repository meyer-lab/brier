from __future__ import annotations
 
import json
from typing import TYPE_CHECKING, Any
 
import numpy as np
import polars as pl
 
if TYPE_CHECKING:
    from .types import BootstrapIndices, ModelWrapper


# Construct a metadata row for a failed replicate.
def _make_failed_metadata(
    indices: BootstrapIndices,
    y: np.ndarray,
    model: ModelWrapper,
    seed: int,
    status: str,
    error_msg: str,
    cv_duration_s: float | None = None,
    fit_duration_s: float | None = None,
) -> dict[str, Any]:
    unique_inbag = np.unique(indices.inbag)
    return {
        "replicate_idx": indices.replicate_idx,
        "seed": seed,
        "model_label": model.label,
        "n_inbag_draws": len(indices.inbag),
        "n_inbag_unique": indices.n_inbag_unique,
        "n_oob": len(indices.oob),
        "n_pos_inbag": int(np.sum(y[unique_inbag])),
        "n_pos_oob": int(np.sum(y[indices.oob])) if len(indices.oob) > 0 else 0,
        "cv_duration_s": cv_duration_s,
        "fit_duration_s": fit_duration_s,
        "status": status,
        "error_msg": error_msg,
    }


# Infer the Polars dtype for a scalar value.
def _infer_dtype(value: Any) -> pl.DataType:
    if isinstance(value, (bool, np.bool_)):
        return pl.Boolean
    if isinstance(value, (int, np.integer)):
        return pl.Int64
    if isinstance(value, (float, np.floating)):
        return pl.Float64
    if isinstance(value, str):
        return pl.Utf8
    return pl.Utf8


# Classify an extractor return value for routing to the appropriate Polars column type and storage format.
def _classify_return(value: Any) -> str:
    if value is None:
        return "scalar"
 
    if isinstance(value, dict):
        return "dict"
 
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return "ragged"
        if value.ndim == 1:
            return "array_1d"
        if value.ndim == 2:
            return "array_2d"
        return "json"
 
    if isinstance(value, (list, tuple)):
        # Check if it's a list of arrays (ragged)
        if len(value) > 0 and isinstance(value[0], np.ndarray):
            return "ragged"
        return "json"
 
    if isinstance(value, (bool, np.bool_, int, np.integer, float, np.floating, str)):
        return "scalar"
 
    return "json"

# Map numpy array dtype to the appropriate Polars List dtype for ragged storage.
def _ragged_to_polars(arr: np.ndarray) -> pl.DataType:
    if np.issubdtype(arr.dtype, np.floating) or arr.dtype == object:
        return pl.Float64
    if np.issubdtype(arr.dtype, np.integer):
        return pl.Int64
    if np.issubdtype(arr.dtype, np.bool_):
        return pl.Boolean
    return pl.Float64  # default fallback

# Flatten a dict one level deep with dot-notation keys. Nested dicts become JSON strings.
def _flatten_dict(prefix: str, d: dict) -> dict[str, Any]:
    flat = {}
    for key, value in d.items():
        flat_key = f"{prefix}.{key}"
        if isinstance(value, dict):
            # One level only — nested dicts become JSON
            flat[flat_key] = json.dumps(value)
        else:
            flat[flat_key] = value
    return flat


