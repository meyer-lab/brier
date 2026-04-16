from __future__ import annotations
 
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
 
import numpy as np
import polars as pl
 

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
    "status": pl.Utf8,
    "error_msg": pl.Utf8,
}

# Fixed columns always present in output table
OUTPUT_PROBABILITY_SCHEMA = {
    "replicate_idx": pl.UInt32,
    "sample_idx": pl.UInt32,
    "bag": pl.Utf8,
    "y_true": pl.UInt8,
    "prob_0": pl.Float64,
    "prob_1": pl.Float64,
}


@dataclass
class ReplicateResult:
    """Result from a single bootstrap replicate."""
    metadata: dict[str, Any]
    output_rows: list[dict[str, Any]]
    attributes: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)


@dataclass
class BootstrapResult:
    """Complete bootstrap evaluation result with hybrid storage.
 
    Attributes
    ----------
    metadata : pl.DataFrame
        Information about each replicate run.
    attributes : pl.DataFrame
        If user provides additional attributes to extract, scalar + ragged 
        array as List columns appear here per replicate.
    array_attributes : dict[str, np.ndarray]
        If user provides additional attributes to extract, fixed-shape array 
        attributes appear here. Keyed by extractor name, arrays have shape 
        (n_replicates, ...).
    output : pl.DataFrame
        Contains returned probability estimates for each sample. If user
        provides additional telemetry functions to extract/compute,
        scalar + ragged array (as List columns) returns will appear here.
    array_telemetry : dict[str, np.ndarray]
        If user provides additional fixed-shape array telemetry to extract,
        will appear here. Keyed by extractor name, arrays have shape 
        (total_rows, d). Row-aligned with output DF.
    """
    metadata: pl.DataFrame
    attributes: pl.DataFrame
    array_attributes: dict[str, np.ndarray] = field(default_factory=dict)
    output: pl.DataFrame = field(default_factory=lambda: pl.DataFrame())
    array_telemetry: dict[str, np.ndarray] = field(default_factory=dict)
 
    @property
    def n_replicates(self) -> int:
        return self.metadata.height
 
    @property
    def n_successful(self) -> int:
        return self.metadata.filter(pl.col("status") == "success").height
 
    @property
    def oob_output(self) -> pl.DataFrame:
        return self.output.filter(pl.col("bag") == "oob")
 
    @property
    def inbag_output(self) -> pl.DataFrame:
        return self.output.filter(pl.col("bag") == "inbag")
 
    def save(self, path: str | Path) -> None:
        """Save all result components to a directory.
 
        Structure:
            path/
            ├── metadata.parquet
            ├── attributes.parquet
            ├── output.parquet
            ├── array_attributes.npz
            └── array_telemetry.npz
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
 
        self.metadata.write_parquet(path / "metadata.parquet")
        self.attributes.write_parquet(path / "attributes.parquet")
        self.output.write_parquet(path / "output.parquet")
 
        if self.array_attributes:
            np.savez(path / "array_attributes.npz", **self.array_attributes)
        if self.array_telemetry:
            np.savez(path / "array_telemetry.npz", **self.array_telemetry)
 
    @classmethod
    def load(cls, path: str | Path) -> BootstrapResult:
        """Load a saved BootstrapResult from a directory."""
        path = Path(path)
 
        metadata = pl.read_parquet(path / "metadata.parquet")
        attributes = pl.read_parquet(path / "attributes.parquet")
        output = pl.read_parquet(path / "output.parquet")
 
        array_attributes: dict[str, np.ndarray] = {}
        npz_path = path / "array_attributes.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                array_attributes = {k: data[k] for k in data.files}
 
        array_telemetry: dict[str, np.ndarray] = {}
        npz_path = path / "array_telemetry.npz"
        if npz_path.exists():
            with np.load(npz_path) as data:
                array_telemetry = {k: data[k] for k in data.files}
 
        return cls(
            metadata=metadata,
            attributes=attributes,
            array_attributes=array_attributes,
            output=output,
            array_telemetry=array_telemetry,
        )

