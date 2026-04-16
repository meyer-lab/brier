from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


class FitModel(Protocol):
    # X should be anything that supports __getitem__ with integer array indexing (e.g., np.ndarray, np.object, AnnData, etc.)
    # This is intentionally flexible to allow for a custom "sample" data representation without enforcing a specific type.
    def fit(self, X: Any, y: np.ndarray) -> Any: ...
    def predict_proba(self, X: Any) -> np.ndarray: ...

# (X_train, y_train, model) -> (configured_model, hparams_dict)
# The resolver is responsible for cloning/re-instantiating the model internally
# No function to deepcopy on users behalf
HparamResolver = Callable[[Any, np.ndarray, Any], tuple[Any, dict]]

@dataclass(frozen=True)
class BootstrapIndices:
    """Index sets for a single bootstrap rpelicate/iteration."""
    replicate_idx: int
    inbag: np.ndarray # (n_samples,) sampled with replacement
    n_inbag_unique: int
    oob: np.ndarray # (n_samples - n_inbag_unique,)

@dataclass
class ModelWrapper:
    """Normalizes sklearn-like models and (fit, predict_proba) callable pairs into a uniform interface so the 
    orchestrator can run any model that follows either convention."""
    label: str
    fit_fn: Callable[[Any, np.ndarray], Any]
    predict_proba_fn: Callable[[Any, np.ndarray], np.ndarray]
    hparam_resolver: HparamResolver | None = None
    _validated: bool = field(default=False, repr=False)

    def validate_proba(self, proba: np.ndarray, n_samples: int) -> None:
        """Checks shape, dtype, value range, and row normalization -> Raises ValueError on failure."""
        if self._validated:
            return

        if not isinstance(proba, np.ndarray):
            raise ValueError(f"predict_proba_fn must return a numpy array, got {type(proba).__name__}")

        if proba.ndim != 2 or proba.shape[1] != 2:
            raise ValueError(f"predict_proba_fn must return a 2D array with shape (n_samples, 2), got {proba.shape}")

        if proba.shape[0] != n_samples:
            raise ValueError(f"predict_proba_fn returned {proba.shape[0]} samples, expected {n_samples}")

        if np.any(proba < 0.0) or np.any(proba > 1.0):
            out_of_range = proba[(proba < 0.0) | (proba > 1.0)]
            raise ValueError(
                f"predict_proba values must be in [0, 1], "
                f"found out-of-range values: {out_of_range[:5]}"
            )

        row_sums = proba.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            bad_idx = np.where(~np.isclose(row_sums, 1.0, atol=1e-6))[0]
            raise ValueError(
                f"predict_proba rows must sum to ~1.0, "
                f"replicates {bad_idx[:5]} have sums {row_sums[bad_idx[:5]]}"
            )
        self._validated = True

@dataclass
class AttrSpec:
    """Specification for a user-defined attribute extractor. Model-specific."""
    fn: Callable[[Any], Any]
    critical: bool = False
    ragged: bool = False

@dataclass
class TelemetrySpec:
    """Specification for a user-defined telemetry extractor. Model- and Sample- specific."""
    fn: Callable[[Any, Any, np.ndarray], np.ndarray]
    critical: bool = False
    ragged: bool = False

class CriticalExtractorError(Exception):
    pass