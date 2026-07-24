"""Store QUERY side

Reads the plain `<store>/runs/<run_id>/{meta.json,replicates.parquet,predictions.parquet,
membership.parquet}` layout written by `bootstraptools.store`
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

_TOP_LEVEL_COLUMNS = [
    "run_id",
    "procedure",
    "dataset",
    "model_label",
    "status",
    "n_replicates",
    "created",
]

_VALID_TABLES = {"replicates", "predictions", "membership"}


def _load_all_meta(store: str | Path) -> list[dict[str, Any]]:
    """Read and parse every `<store>/runs/*/meta.json`."""
    runs_dir = Path(store) / "runs"
    if not runs_dir.is_dir():
        return []
    metas = []
    for run_dir in sorted(runs_dir.iterdir()):
        meta_path = run_dir / "meta.json"
        if meta_path.is_file():
            metas.append(json.loads(meta_path.read_text()))
    return metas


def _matches(meta: dict[str, Any], filters: dict[str, Any]) -> bool:
    """Check whether one run's meta dict satisfies all AND-ed filters."""
    for key, want in filters.items():
        if key == "tags":
            want_tags = [want] if isinstance(want, str) else list(want)
            have_tags = meta.get("tags") or []
            if not all(t in have_tags for t in want_tags):
                return False
        elif key in _TOP_LEVEL_COLUMNS:
            if meta.get(key) != want:
                return False
        elif key.startswith("config."):
            config_key = key[len("config.") :]
            config = meta.get("config") or {}
            if config_key not in config or config[config_key] != want:
                return False
        else:
            # Unknown filter key: matches nothing.
            return False
    return True


def query_runs(
    store: str | Path, filters: dict[str, Any] | None = None
) -> pl.DataFrame:
    """List runs in `store`, one row per run, sorted by `created` ascending.

    Columns: `run_id, procedure, dataset, model_label, status, n_replicates,
    created`, flattened `config.<key>` and `summary.<key>` columns, and a
    `tags` column (`List(Utf8)`).

    `filters` (all AND-ed): exact-match on `procedure`, `dataset`,
    `model_label`, `status`, `run_id`; `tags` (str or list[str], run must
    have ALL requested tags); `config.<key>` exact-match. Unknown filter
    keys match no rows.
    """
    metas = _load_all_meta(store)
    if filters:
        metas = [m for m in metas if _matches(m, filters)]

    if not metas:
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "procedure": pl.Utf8,
                "dataset": pl.Utf8,
                "model_label": pl.Utf8,
                "status": pl.Utf8,
                "n_replicates": pl.Int64,
                "created": pl.Utf8,
                "tags": pl.List(pl.Utf8),
            }
        )

    rows = []
    for meta in metas:
        row: dict[str, Any] = {col: meta.get(col) for col in _TOP_LEVEL_COLUMNS}
        row["tags"] = list(meta.get("tags") or [])
        for key, value in (meta.get("config") or {}).items():
            row[f"config.{key}"] = value
        for key, value in (meta.get("summary") or {}).items():
            row[f"summary.{key}"] = value
        rows.append(row)

    df = pl.DataFrame(rows, infer_schema_length=None)
    return df.sort("created")


def query_run_table(
    store: str | Path, run_id: str, table: str = "replicates"
) -> pl.DataFrame:
    """Read one run's `<table>.parquet`.

    `table` must be one of "replicates", "predictions", "membership".
    """
    if table not in _VALID_TABLES:
        raise ValueError(
            f"Unknown table {table!r}; expected one of {sorted(_VALID_TABLES)}"
        )
    path = Path(store) / "runs" / run_id / f"{table}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"No {table} table found for run {run_id!r}: {path}")
    return pl.read_parquet(path)


def query_apparent(store: str | Path, run_id: str) -> pl.DataFrame:
    """Read one run's `apparent.parquet` (sample_idx, y_true, p).

    Raises `FileNotFoundError` if the run has no apparent record (i.e. it is
    not an optimism run, or `log_apparent` was never called).
    """
    path = Path(store) / "runs" / run_id / "apparent.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"No apparent table found for run {run_id!r}: {path}"
        )
    return pl.read_parquet(path)


def membership_matrix(store: str | Path, run_id: str) -> np.ndarray:
    """Reconstruct the dense (n_replicates, universe_size) fit-count matrix.

    Reads `meta.json` for `n_replicates`/`universe_size` and
    `membership.parquet` for the sparse `(replicate_idx, sample_idx, count)`
    rows, and fills a dense int array (0 where a unit is absent from a
    replicate's fit set).
    """
    meta_path = Path(store) / "runs" / run_id / "meta.json"
    meta = json.loads(meta_path.read_text())
    n_replicates = int(meta["n_replicates"])
    universe_size = int(meta["universe_size"])

    membership = query_run_table(store, run_id, "membership")
    dense = np.zeros((n_replicates, universe_size), dtype=int)
    if membership.height > 0:
        replicate_idx = membership["replicate_idx"].to_numpy()
        sample_idx = membership["sample_idx"].to_numpy()
        count = membership["count"].to_numpy()
        dense[replicate_idx, sample_idx] = count
    return dense


def load_runs_table(
    store: str | Path,
    filters: dict[str, Any] | None = None,
    table: str = "replicates",
) -> pl.DataFrame:
    """List runs by `filters`, pull one `table` per run, concatenate.

    Each run's table is prefixed with `run_id`, `dataset`, `model_label`
    columns for downstream grouping. Uses diagonal-relaxed concatenation so
    runs with differing metric columns still stack (missing -> null).
    """
    runs = query_runs(store, filters)
    if runs.height == 0:
        return pl.DataFrame()

    frames = []
    for run_id, dataset, model_label in runs.select(
        "run_id", "dataset", "model_label"
    ).iter_rows():
        table_df = query_run_table(store, run_id, table)
        table_df = table_df.select(
            pl.lit(run_id).alias("run_id"),
            pl.lit(dataset).alias("dataset"),
            pl.lit(model_label).alias("model_label"),
            pl.all(),
        )
        frames.append(table_df)

    return pl.concat(frames, how="diagonal_relaxed")
