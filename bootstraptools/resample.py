from __future__ import annotations

import numpy as np


def draw(
    rng: np.random.Generator,
    source_idx,
    size: int | None = None,
    replace: bool = True,
):
    """Draw indices from `source_idx` with a given (or default) resample size.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator used for the draw.
    source_idx : array-like of int
        1D array-like of integer indices to sample from.
    size : int, optional
        Number of draws. Defaults to `len(source_idx)` when None.
    replace : bool, default True
        Whether to sample with replacement. If False and `size` exceeds
        `len(source_idx)`, raises ValueError.

    Returns
    -------
    numpy.ndarray
        1D int array of drawn indices (values are elements of `source_idx`).
    """
    source_idx = np.asarray(source_idx)
    if size is None:
        size = len(source_idx)
    if not replace and size > len(source_idx):
        raise ValueError(
            f"Cannot draw {size} samples without replacement from "
            f"{len(source_idx)} available indices."
        )
    return rng.choice(source_idx, size=size, replace=replace)


def draw_counts(drawn_idx, universe_size: int):
    """Count how many times each global index was drawn.

    Parameters
    ----------
    drawn_idx : array-like of int
        Indices drawn from a universe of size `universe_size`.
    universe_size : int
        Size of the universe of indices.

    Returns
    -------
    numpy.ndarray
        1D int array of length `universe_size`; entry i is the number of
        times global index i was drawn.
    """
    return np.bincount(drawn_idx, minlength=universe_size)


def out_of_bag(drawn_idx, source_idx):
    """Return the sorted, unique complement of `drawn_idx` within `source_idx`.

    Parameters
    ----------
    drawn_idx : array-like of int
        Indices that were drawn.
    source_idx : array-like of int
        Indices that were available to draw from.

    Returns
    -------
    numpy.ndarray
        Sorted unique indices in `source_idx` that were not drawn.
    """
    return np.setdiff1d(source_idx, drawn_idx)


def select(X, idx):
    """Select elements of `X` at positions `idx`.

    Parameters
    ----------
    X : numpy.ndarray or sequence
        Data to select from. If a numpy array, fancy indexing is used.
        Otherwise, `X` is treated as a sequence of bags and a list is built.
    idx : array-like of int
        Indices to select.

    Returns
    -------
    numpy.ndarray or list
        `X[idx]` if `X` is a numpy array, else `[X[i] for i in idx]`.
    """
    if isinstance(X, np.ndarray):
        return X[idx]
    return [X[i] for i in idx]
