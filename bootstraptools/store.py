"""Store WRITE side:

A `Run` is a filesystem sink under `<store>/runs/<run_id>/` that a bootstrap
runner writes to once per replicate via `log_replicate`, then finalizes via
`finish`.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from bootstraptools.procedures import ResamplePlan

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

_SPLIT_BY_PROCEDURE = {
    "train_resample_holdout": "holdout",
    "inbag_oob": "oob",
    "optimism_bootstrap": "original",
}


def _sanitize(part: str) -> str:
    """Replace os-unsafe characters with underscores."""
    return _UNSAFE_CHARS.sub("_", part)


def _bootstraptools_version() -> str:
    try:
        return version("bootstraptools")
    except PackageNotFoundError:
        return "unknown"


def _to_scalar(value: Any) -> Any:
    """Coerce numpy scalars to native Python scalars and pass through others."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def init(
    store: str | Path,
    *,
    procedure: str,
    dataset: str,
    model_label: str,
    config: dict[str, Any] | None = None,
    tags: list[str] | None = None,
    store_model_state: bool = False,
    run_id: str | None = None,
) -> Run:
    """Create a new run directory under `store` and return a `Run` handle.

    Parameters
    ----------
    store : str | Path
        Path to the store ROOT. `<store>/runs/<run_id>/` is created.
    procedure, dataset, model_label : str
        Identifying metadata written to `meta.json`.
    config : dict, optional
        Arbitrary run configuration, defaults to `{}`.
    tags : list[str], optional
        Free-form tags, defaults to `[]`.
    store_model_state : bool, default False
        Opt-in required for `log_replicate(..., model_state=...)`.
    run_id : str, optional
        Explicit run id. If None, generated as
        `f"{model_label}-{dataset}-{uuid4().hex[:8]}"` with unsafe chars
        replaced by `_`.

    Returns
    -------
    Run
    """
    store = Path(store)
    if run_id is None:
        run_id = f"{_sanitize(model_label)}-{_sanitize(dataset)}-{uuid4().hex[:8]}"

    run_dir = store / "runs" / run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)

    meta = {
        "run_id": run_id,
        "procedure": procedure,
        "dataset": dataset,
        "model_label": model_label,
        "config": config or {},
        "tags": tags or [],
        "store_model_state": store_model_state,
        "created": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "n_replicates": 0,
        "summary": {},
        "bootstraptools_version": _bootstraptools_version(),
    }
    run = Run(run_dir=run_dir, meta=meta)
    run._write_meta()
    return run


class Run:
    """A write handle for one bootstrap run's outputs.

    Constructed by `init` not intended to be instantiated directly.
    """

    def __init__(self, run_dir: Path, meta: dict[str, Any]) -> None:
        self._run_dir = run_dir
        self._meta = meta
        self._finished = False
        self._replicate_rows: list[dict[str, Any]] = []
        self._prediction_rows: list[dict[str, Any]] = []
        self._membership_rows: list[dict[str, Any]] = []
        self._universe_size = 0
        self._apparent_rows: list[dict[str, Any]] = []
        self._apparent_logged = False

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def store_model_state(self) -> bool:
        return bool(self._meta["store_model_state"])

    def _meta_path(self) -> Path:
        return self.run_dir / "meta.json"

    def _write_meta(self) -> None:
        self._meta_path().write_text(json.dumps(self._meta, indent=2))

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._finished:
            return
        if exc_type is not None:
            self._meta["status"] = "failed"
            self._write_meta()
            return
        self.finish()

    def log_replicate(
        self,
        plan: ResamplePlan,
        *,
        metrics: dict[str, Any] | None = None,
        y_true: np.ndarray | None = None,
        p: np.ndarray | None = None,
        arrays: dict[str, np.ndarray] | None = None,
        model_state: dict[str, np.ndarray] | None = None,
        model_fit_seed: int | None = None,
        status: str = "success",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record one replicate's outcome.

        Parameters
        ----------
        plan : ResamplePlan
            The resample plan this replicate was fit/evaluated on.
        metrics : dict, optional
            Scalar metric values (float/int/bool/str/None or numpy scalars).
        y_true, p : np.ndarray, optional
            Aligned to `plan.eval_indices`; if both are given, per-sample
            prediction rows are recorded.
        arrays : dict[str, np.ndarray], optional
            Written immediately to `arrays/replicate_{idx:04d}.npz`.
        model_state : dict[str, np.ndarray], optional
            Written immediately to `model_state/replicate_{idx:04d}.npz`.
            Requires `store_model_state=True` at `init` time.
        model_fit_seed : int, optional
            Seed used by the runner to fit the model (distinct from
            `plan.resample_seed`).
        status : str, default "success"
            Replicate outcome status.
        extra : dict, optional
            Additional scalar columns merged into the replicate row.
        """
        metrics = metrics or {}
        extra = extra or {}
        self._universe_size = max(self._universe_size, len(plan.fit_counts))

        row: dict[str, Any] = {
            "replicate_idx": plan.replicate_idx,
            "resample_seed": plan.resample_seed,
            "procedure": plan.procedure,
            "n_fit": len(plan.fit_indices),
            "n_fit_unique": int((plan.fit_counts > 0).sum()),
            "n_eval": len(plan.eval_indices),
            "model_fit_seed": model_fit_seed,
            "status": status,
        }
        for key, value in {**metrics, **extra}.items():
            row[key] = _to_scalar(value)
        self._replicate_rows.append(row)

        if y_true is not None and p is not None:
            y_true = np.asarray(y_true)
            p = np.asarray(p)
            if not (len(y_true) == len(p) == len(plan.eval_indices)):
                raise ValueError(
                    "y_true, p, and plan.eval_indices must have equal length: "
                    f"got len(y_true)={len(y_true)}, len(p)={len(p)}, "
                    f"len(eval_indices)={len(plan.eval_indices)}"
                )
            split = _SPLIT_BY_PROCEDURE.get(plan.procedure, plan.procedure)
            for j, sample_idx in enumerate(plan.eval_indices):
                self._prediction_rows.append(
                    {
                        "replicate_idx": plan.replicate_idx,
                        "sample_idx": int(sample_idx),
                        "y_true": int(y_true[j]),
                        "p": float(p[j]),
                        "split": split,
                    }
                )

        nonzero = np.nonzero(plan.fit_counts)[0]
        for i in nonzero:
            self._membership_rows.append(
                {
                    "replicate_idx": plan.replicate_idx,
                    "sample_idx": int(i),
                    "count": int(plan.fit_counts[i]),
                }
            )

        if arrays:
            arrays_dir = self.run_dir / "arrays"
            arrays_dir.mkdir(exist_ok=True)
            np.savez(
                arrays_dir / f"replicate_{plan.replicate_idx:04d}.npz", **arrays
            )

        if model_state:
            if not self.store_model_state:
                raise ValueError(
                    "log_replicate() received model_state but this run was "
                    "not opted in via init(..., store_model_state=True)."
                )
            model_state_dir = self.run_dir / "model_state"
            model_state_dir.mkdir(exist_ok=True)
            np.savez(
                model_state_dir / f"replicate_{plan.replicate_idx:04d}.npz",
                **model_state,
            )

    def log_apparent(
        self,
        *,
        y_true: np.ndarray,
        p: np.ndarray,
        arrays: dict[str, np.ndarray] | None = None,
        model_state: dict[str, np.ndarray] | None = None,
    ) -> None:
        """Record the apparent fit's per-sample predictions.

        The apparent fit is the model fit once on the full original dataset
        (no resampling), evaluated on all n original samples. This is the
        P_app term in the Efron–Gong optimism correction.

        Parameters
        ----------
        y_true, p : np.ndarray
            Equal-length 1D arrays over the full original sample
            (length n = universe_size).
        arrays : dict[str, np.ndarray], optional
            Written immediately to `apparent/arrays.npz`.
        model_state : dict[str, np.ndarray], optional
            Written immediately to `apparent/model_state.npz`. Requires
            `store_model_state=True` at `init` time.

        Raises
        ------
        ValueError
            If `y_true` and `p` have mismatched lengths, if this run already
            has an apparent record, or if `model_state` is given without
            `store_model_state=True`.
        """
        if self._apparent_logged:
            raise ValueError(
                "log_apparent() was already called for this run; a run has "
                "exactly one apparent fit."
            )

        y_true = np.asarray(y_true)
        p = np.asarray(p)
        if len(y_true) != len(p):
            raise ValueError(
                "y_true and p must have equal length: "
                f"got len(y_true)={len(y_true)}, len(p)={len(p)}"
            )

        for i in range(len(y_true)):
            self._apparent_rows.append(
                {
                    "sample_idx": int(i),
                    "y_true": int(y_true[i]),
                    "p": float(p[i]),
                }
            )

        if arrays:
            apparent_dir = self.run_dir / "apparent"
            apparent_dir.mkdir(exist_ok=True)
            np.savez(apparent_dir / "arrays.npz", **arrays)

        if model_state:
            if not self.store_model_state:
                raise ValueError(
                    "log_apparent() received model_state but this run was "
                    "not opted in via init(..., store_model_state=True)."
                )
            apparent_dir = self.run_dir / "apparent"
            apparent_dir.mkdir(exist_ok=True)
            np.savez(apparent_dir / "model_state.npz", **model_state)

        self._apparent_logged = True

    def finish(self, summary: dict[str, Any] | None = None) -> None:
        """Flush buffered rows to parquet and finalize `meta.json`.

        Idempotent: calling twice has no additional effect after the first.
        """
        if self._finished:
            return

        pl.DataFrame(self._replicate_rows).write_parquet(
            self.run_dir / "replicates.parquet"
        )
        pl.DataFrame(self._prediction_rows).write_parquet(
            self.run_dir / "predictions.parquet"
        )
        pl.DataFrame(self._membership_rows).write_parquet(
            self.run_dir / "membership.parquet"
        )
        if self._apparent_logged:
            pl.DataFrame(self._apparent_rows).write_parquet(
                self.run_dir / "apparent.parquet"
            )

        self._meta["status"] = "finished"
        self._meta["n_replicates"] = len(self._replicate_rows)
        self._meta["summary"] = summary or {}
        self._meta["universe_size"] = self._universe_size
        self._write_meta()

        self._finished = True
