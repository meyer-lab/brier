
from __future__ import annotations

import numpy as np


def derive_seeds(master_seed: int, names: list[str]) -> dict[str, int]:
    """Derive one independent named process seed per entry in `names` from a single master seed."""
    rng = np.random.default_rng(master_seed)
    return {n: int(rng.integers(0, 2**31)) for n in names}


def replicate_seeds(seed: int, n_replicates: int) -> list[int]:
    """Derive `n_replicates` per-replicate seeds from one process seed (e.g. for model fits)."""
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(0, 2**31, size=n_replicates)]
