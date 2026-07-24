from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bootstraptools.resample import draw, draw_counts, out_of_bag


@dataclass(frozen=True)
class ResamplePlan:
    """One replicate's fit/eval resample plan over the shared patient universe."""

    replicate_idx: int
    fit_indices: np.ndarray
    eval_indices: np.ndarray
    fit_counts: np.ndarray
    resample_seed: int
    procedure: str


def _resample_seeds(bootstrap_seed: int, n_replicates: int) -> list[int]:
    """Derive per-replicate resample seeds from one bootstrap seed."""
    seed_rng = np.random.default_rng(bootstrap_seed)
    return [int(x) for x in seed_rng.integers(0, 2**31, size=n_replicates)]


def inbag_oob(
    universe_size: int,
    n_replicates: int,
    bootstrap_seed: int,
    size: int | None = None,
    replace: bool = True,
) -> list[ResamplePlan]:
    """Fit on an in-bag resample, evaluate on its out-of-bag complement."""
    source = np.arange(universe_size)
    seeds = _resample_seeds(bootstrap_seed, n_replicates)
    plans = []
    for k, resample_seed in enumerate(seeds):
        rng = np.random.default_rng(resample_seed)
        fit = draw(rng, source, size=size, replace=replace)
        eval_ = out_of_bag(fit, source)
        fit_counts = draw_counts(fit, universe_size)
        plans.append(
            ResamplePlan(
                replicate_idx=k,
                fit_indices=fit,
                eval_indices=eval_,
                fit_counts=fit_counts,
                resample_seed=resample_seed,
                procedure="inbag_oob",
            )
        )
    return plans


def optimism_bootstrap(
    universe_size: int,
    n_replicates: int,
    bootstrap_seed: int,
    size: int | None = None,
    replace: bool = True,
) -> list[ResamplePlan]:
    """Efron–Gong optimism bootstrap: fit on a bootstrap resample of the n
    rows, evaluate on the full original sample. Per-replicate optimism =
    (apparent-on-resample) − (test-on-original) is computed downstream from
    the stored predictions + membership counts; corrected = apparent −
    mean(optimism).
    """
    source = np.arange(universe_size)
    if size is None:
        size = universe_size
    seeds = _resample_seeds(bootstrap_seed, n_replicates)
    plans = []
    for k, resample_seed in enumerate(seeds):
        rng = np.random.default_rng(resample_seed)
        fit = draw(rng, source, size=size, replace=replace)
        eval_ = np.arange(universe_size)
        fit_counts = draw_counts(fit, universe_size)
        plans.append(
            ResamplePlan(
                replicate_idx=k,
                fit_indices=fit,
                eval_indices=eval_,
                fit_counts=fit_counts,
                resample_seed=resample_seed,
                procedure="optimism_bootstrap",
            )
        )
    return plans


@dataclass(frozen=True)
class DoubleResamplePlan:
    """One outer replicate of Noma method 2 (double/two-stage bootstrap), 
    paired with its nested inner resample plans."""

    outer_idx: int
    outer_plan: ResamplePlan
    inner_plans: list[ResamplePlan]


def double_optimism_bootstrap(
    universe_size: int,
    n_outer: int,
    n_inner: int,
    bootstrap_seed: int,
    size: int | None = None,
    replace: bool = True,
) -> list[DoubleResamplePlan]:
    """Noma method 2: nested (double) optimism bootstrap.

    All seeds are derived deterministically from one `bootstrap_seed`: the
    outer seeds via `_resample_seeds(bootstrap_seed, n_outer)`, and each
    outer replicate's inner seeds via `_resample_seeds(outer_seed, n_inner)`,
    so the entire R x B tree is reproducible from a single top-level seed.
    """
    source = np.arange(universe_size)
    if size is None:
        size = universe_size
    outer_seeds = _resample_seeds(bootstrap_seed, n_outer)
    double_plans = []
    for r, outer_seed in enumerate(outer_seeds):
        outer_rng = np.random.default_rng(outer_seed)
        outer_fit = draw(outer_rng, source, size=size, replace=replace)
        outer_eval = np.arange(universe_size)
        outer_fit_counts = draw_counts(outer_fit, universe_size)
        outer_plan = ResamplePlan(
            replicate_idx=r,
            fit_indices=outer_fit,
            eval_indices=outer_eval,
            fit_counts=outer_fit_counts,
            resample_seed=outer_seed,
            procedure="double_optimism_bootstrap",
        )

        inner_seeds = _resample_seeds(outer_seed, n_inner)
        inner_plans = []
        for b, inner_seed in enumerate(inner_seeds):
            inner_rng = np.random.default_rng(inner_seed)
            inner_fit = draw(inner_rng, source, size=size, replace=replace)
            inner_eval = np.arange(universe_size)
            inner_fit_counts = draw_counts(inner_fit, universe_size)
            inner_plans.append(
                ResamplePlan(
                    replicate_idx=b,
                    fit_indices=inner_fit,
                    eval_indices=inner_eval,
                    fit_counts=inner_fit_counts,
                    resample_seed=inner_seed,
                    procedure="double_optimism_bootstrap",
                )
            )

        double_plans.append(
            DoubleResamplePlan(
                outer_idx=r,
                outer_plan=outer_plan,
                inner_plans=inner_plans,
            )
        )
    return double_plans


def train_resample_holdout(
    train_idx,
    val_idx,
    universe_size: int,
    n_replicates: int,
    bootstrap_seed: int,
    size: int | None = None,
    replace: bool = True,
) -> list[ResamplePlan]:
    """Resample the training fold with replacement, refit, evaluate every
    replicate on the same fixed held-out validation fold.
    """
    train_idx = np.asarray(train_idx)
    val_idx = np.asarray(val_idx)
    if size is None:
        size = len(train_idx)
    eval_ = np.unique(val_idx)
    seeds = _resample_seeds(bootstrap_seed, n_replicates)
    plans = []
    for k, resample_seed in enumerate(seeds):
        rng = np.random.default_rng(resample_seed)
        fit = draw(rng, train_idx, size=size, replace=replace)
        fit_counts = draw_counts(fit, universe_size)
        plans.append(
            ResamplePlan(
                replicate_idx=k,
                fit_indices=fit,
                eval_indices=eval_,
                fit_counts=fit_counts,
                resample_seed=resample_seed,
                procedure="train_resample_holdout",
            )
        )
    return plans
