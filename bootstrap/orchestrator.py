from __future__ import annotations
 
import json
import sys
import traceback
from typing import TYPE_CHECKING, Any
 
import numpy as np
import polars as pl
 
from .bootstrap import CriticalExtractorError, generate_replicates, run_replicate, wrap_model
from .schemas import (
    METADATA_SCHEMA,
    OUTPUT_PROBABILITY_SCHEMA,
    BootstrapResult,
)
from .util import _classify_return, _infer_dtype
if TYPE_CHECKING:
    from .types import AttrSpec, HparamResolver, TelemetrySpec


def run(
    X: Any,
    y: np.ndarray,
    model: Any,
    model_label: str,
    n_replicates: int,
    random_seed: int = 13,
    hparam_resolver: HparamResolver | None = None,
    collect_inbag: bool = False,
    attribute_extractors: dict[str, AttrSpec] | None = None,
    telemetry_extractors: dict[str, TelemetrySpec] | None = None,
) -> BootstrapResult:
    """
    Run bootstrap evaluation of a binary probabilistic classifier.

    Parameters:
    X : Any
        Feature matrix, shape (n_samples, n_features).
    y : np.ndarray
        Binary label vector, shape (n_samples,).
    model : Any
        The model to evaluate. sklearn-like estimator or tuple of
        (fit_fn, predict_proba_fn) callables.
    model_label : str
        A string label for the model.
    n_replicates : int
        Number of bootstrap replicates.
    random_seed : int
        Random seed for reproducibility.
    hparam_resolver : HparamResolver or None
        Optional hyperparameter tuning within each replicate.
    collect_inbag : bool
        Whether to collect predictions on unique in-bag samples.
    attribute_extractors : dict[str, AttrSpec] or None
        Named attribute extractors applied to fitted model each replicate.
    telemetry_extractors : dict[str, TelemetrySpec] or None
        Named telemetry extractors applied per data split each replicate.

    Returns:
    BootstrapResult
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


# Describes how a single extractor output is stored.
class _ColumnRoute:
    def __init__(
        self,
        name: str,
        target: str,            # 'df' or 'array'
        polars_dtype: pl.DataType | None = None,
        array_shape_tail: tuple[int, ...] | None = None,
        numpy_dtype: np.dtype | None = None,
    ):
        self.name = name
        self.target = target                    # 'df' -> DataFrame column, 'array' -> numpy dict
        self.polars_dtype = polars_dtype         # set if target == 'df'
        self.array_shape_tail = array_shape_tail # set if target == 'array', shape excluding first axis
        self.numpy_dtype = numpy_dtype           # set if target == 'array'
 
    def __repr__(self) -> str:
        if self.target == "df":
            return f"_ColumnRoute({self.name!r}, df, dtype={self.polars_dtype})"
        return f"_ColumnRoute({self.name!r}, array, shape=(...,{self.array_shape_tail}), dtype={self.numpy_dtype})"

# Infer storage routes from the first replicate's attribute values.
def _build_attribute_routes(
    first_attrs: dict[str, Any],
    extractors: dict[str, AttrSpec],
) -> list[_ColumnRoute]:
    routes = []
    for key, value in first_attrs.items():
        if value is None:
            routes.append(_ColumnRoute(key, target="df", polars_dtype=pl.Utf8))
            continue
 
        base_name = key.split(".")[0] if "." in key else key
        ragged = extractors[base_name].ragged if base_name in extractors else False
        ret_type = _classify_return(value)
 
        if ret_type == "scalar":
            routes.append(_ColumnRoute(key, target="df", polars_dtype=_infer_dtype(value)))
 
        elif ret_type in ("array_1d", "array_2d") and not ragged:
            routes.append(_ColumnRoute(
                key, target="array",
                array_shape_tail=value.shape,
                numpy_dtype=value.dtype,
            ))
 
        elif ret_type in ("array_1d", "array_2d") and ragged:
            if value.ndim == 1:
                routes.append(_ColumnRoute(key, target="df", polars_dtype=pl.List(pl.Float64)))
            else:
                routes.append(_ColumnRoute(key, target="df", polars_dtype=pl.List(pl.List(pl.Float64))))
 
        elif ret_type == "ragged":
            routes.append(_ColumnRoute(key, target="df", polars_dtype=pl.List(pl.Float64)))
 
        else:
            routes.append(_ColumnRoute(key, target="df", polars_dtype=pl.Utf8))
    return routes

# Infer storage routes from the first replicate's telemetry values.
def _build_telemetry_routes(
    first_telemetry: dict[str, Any],
    extractors: dict[str, TelemetrySpec],
) -> list[_ColumnRoute]:
    routes = []
    for name, value in first_telemetry.items():
        if value is None:
            routes.append(_ColumnRoute(name, target="df", polars_dtype=pl.Float64))
            continue
 
        spec = extractors[name]
        ret_type = _classify_return(value)
 
        if ret_type == "ragged":
            if not spec.ragged:
                raise ValueError(
                    f"Telemetry extractor '{name}' returned ragged output but "
                    f"ragged=False. Set ragged=True in TelemetrySpec."
                )
            routes.append(_ColumnRoute(name, target="df", polars_dtype=pl.List(pl.Float64)))
 
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                dtype = _infer_dtype(value.flat[0]) if value.size > 0 else pl.Float64
                routes.append(_ColumnRoute(name, target="df", polars_dtype=dtype))
            elif value.ndim == 2:
                if spec.ragged:
                    routes.append(_ColumnRoute(name, target="df", polars_dtype=pl.List(pl.Float64)))
                else:
                    d = value.shape[1]
                    routes.append(_ColumnRoute(
                        name, target="array",
                        array_shape_tail=(d,),
                        numpy_dtype=value.dtype,
                    ))
            else:
                raise ValueError(
                    f"Telemetry extractor '{name}' returned {value.ndim}D array, "
                    f"expected 1D or 2D."
                )
        else:
            raise ValueError(
                f"Telemetry extractor '{name}' returned {type(value).__name__}, "
                f"expected ndarray."
            )
    return routes

#  Convert an extractor return value to a DataFrame-compatible cell value.
def _to_df_cell(value: Any, route: _ColumnRoute) -> Any:
    if value is None:
        return None
 
    if route.polars_dtype == pl.Utf8 and not isinstance(value, str):
        if isinstance(value, np.ndarray):
            return json.dumps(value.tolist())
        return json.dumps(value)
 
    if isinstance(route.polars_dtype, pl.List):
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                return value.tolist()
            elif value.ndim == 2:
                return [row.tolist() for row in value]
        if isinstance(value, list):
            return [v.tolist() if isinstance(v, np.ndarray) else v for v in value]
        return value
 
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value

# Validate that a fixed-shape array matches the expected shape.
def _validate_shape(
    name: str,
    value: Any,
    route: _ColumnRoute,
    replicate_idx: int,
) -> None:
    if route.target != "array" or value is None:
        return
    if not isinstance(value, np.ndarray):
        raise ValueError(
            f"Replicate {replicate_idx}: extractor '{name}' returned "
            f"{type(value).__name__}, expected ndarray (shape inferred from "
            f"first replicate as {route.array_shape_tail})"
        )
    if value.shape != route.array_shape_tail:
        raise ValueError(
            f"Replicate {replicate_idx}: extractor '{name}' returned shape "
            f"{value.shape}, expected {route.array_shape_tail} "
            f"(set ragged=True if variable shapes are intentional)"
        )

# Validate telemetry array shape for this replicate.
def _validate_telemetry_shape(
    name: str,
    value: Any,
    route: _ColumnRoute,
    replicate_idx: int,
    n_rows: int,
) -> None:
    if value is None:
        return
    if route.target == "array" and isinstance(value, np.ndarray):
        expected_d = route.array_shape_tail[0] if route.array_shape_tail else None
        if value.ndim == 2 and expected_d is not None and value.shape[1] != expected_d:
            raise ValueError(
                f"Replicate {replicate_idx}: telemetry '{name}' returned "
                f"{value.shape[1]} columns, expected {expected_d} "
                f"(set ragged=True if variable widths are intentional)"
            )