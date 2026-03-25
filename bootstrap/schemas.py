from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from .types import BootstrapIndices, ModelWrapper

METADATA_SCHEMA = {
    "replicate_idx": pl.UInt32,
    "seed": pl.UInt64,
    "model_label": pl.Utf8,
    "n_inbag_draws": pl.UInt32,
    "n_inbag_unique": pl.UInt32,
    "n_oob": pl.UInt32,
    "n_pos_inbag": pl.UInt32,
    "n_pos_oob": pl.UInt32,
    "cv_duration_s": pl.Float64,
    "fit_duration_s": pl.Float64,
    "hparams": pl.Utf8,
    "status": pl.Utf8,
    "error_msg": pl.Utf8,
}

PROBABILITY_SCHEMA = {
    "replicate_idx": pl.UInt32,
    "sample_idx": pl.UInt32,
    "bag": pl.Utf8,
    "y_true": pl.UInt8,
    "prob_0": pl.Float64,
    "prob_1": pl.Float64,
}

@dataclass
class BootstrapResult:
    metadata: pl.DataFrame
    probabilities: pl.DataFrame

    @property
    def n_replicates(self) -> int:
        return self.metadata.height

    @property
    def n_successful(self) -> int:
        return self.metadata.filter(pl.col("status") == "success").height

    @property
    def oob_probabilities(self) -> pl.DataFrame:
        return self.probabilities.filter(pl.col("bag") == "oob")

    @property
    def inbag_probabilities(self) -> pl.DataFrame:
        return self.probabilities.filter(pl.col("bag") == "inbag")


@dataclass
class ReplicateResult:
    metadata: dict[str, Any]
    probabilities: list[dict[str, Any]]

def _make_failed_metadata(
    indices: BootstrapIndices,
    y: np.ndarray,
    model: ModelWrapper,
    seed: int,
    status: str,
    error_msg: str,
    cv_duration_s: float | None = None,
    fit_duration_s: float | None = None,
    hparams_dict: dict | None = None,
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
        "hparams": json.dumps(hparams_dict) if hparams_dict is not None else None,
        "status": status,
        "error_msg": error_msg,
    }
