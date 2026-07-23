# bootstraptools

> The population is to the sample as the sample is to the bootstrap samples

Local storage, query, and uncertainty-quantification infrastructure for
**bootstrap experiments** to evaluate regression and classification model performance. A bootstrap runner writes one run (B replicates) to a local store and downstream processing scripts query it back and compute confidence intervals on demand.

`bootstraptools` is **model-agnostic**. It sees a model as a generic `fit` / `predict_proba` object and stores whatever scalars and named arrays a runner hands it. It is inspired by plain sklearn architectures meaning other types of models may require gentle massaging to work smoothly.

Relies on [uv](https://github.com/astral-sh/uv) as the dependency/package manager.

```bash
uv sync
uv run pytest -q                           # full test suite
uv run python examples/reference_runner.py # end-to-end demo, prints CIs
```

---

## Data model

| Concept | Meaning | On disk |
|---|---|---|
| **Store** | A root directory holding many runs. | `<store>/runs/` |
| **Run** | One bootstrap experiment: B replicates for a `(dataset, model, procedure)` config, plus tags/config/summary. | `<store>/runs/<run_id>/` |
| **Replicate** | One refit + evaluation: scalar metrics, per-sample predictions, resample membership, seeds, optional rich arrays, optional full model state. | rows/files within a run |

Per-run layout:

```
<store>/runs/<run_id>/
  meta.json            # tags, config, procedure, dataset, model_label, seeds,
                       # universe_size, status, n_replicates, summary
  replicates.parquet   # 1 row / replicate: seeds, fit metadata, scalar metrics
  predictions.parquet  # 1 row / (replicate, sample): sample_idx, y_true, p, split
  membership.parquet   # 1 row / (replicate, sample, count>0) 
  arrays/replicate_XXXX.npz       # opt-in rich per-replicate arrays
  model_state/replicate_XXXX.npz  # opt-in full fitted params (store_model_state=True)
```

Tables are [polars](https://github.com/pola-rs/polars)/[parquet](https://github.com/apache/parquet-format/) and arrays are [npz](https://note.nkmk.me/en/python-numpy-load-save-savez-npy-npz/). `arrays/` and `model_state/` npz are written **immediately** per replicate (durable if a long run crashes) and the three
parquet tables are flushed on `finish()`.

---

## Bootstrap procedures

Two procedures are currently supported, but they answer different questions. Both are first-class and a runner **picks one per run**. Each yields a list of
`ResamplePlan`s over a shared membership schema (`fit_counts` keyed to the full
patient universe, so downstream code treats both identically).

```python
import bootstraptools as bs

# (1) in-bag / out-of-bag: fit on a resample, evaluate on its OOB complement.
#     Eval set VARIES per replicate. Asks: how sensitive are predictions to
#     which patients land in vs. out of the resample?
plans = bs.inbag_oob(universe_size=n, n_replicates=100, bootstrap_seed=seed)

# (2) train-resample / refit / fixed holdout: resample the training fold with
#     replacement, refit, evaluate every replicate on the SAME held-out fold.
#     Asks: how sensitive is performance on a fixed target population to
#     training-sample composition?
plans = bs.train_resample_holdout(
    train_idx, val_idx, universe_size=n, n_replicates=100, bootstrap_seed=seed
)
```

Each `ResamplePlan` carries `replicate_idx`, `fit_indices` (with multiplicity),
`eval_indices`, `fit_counts` (membership vector), `resample_seed`, and
`procedure`. Every replicate is reproducible in isolation from its
`resample_seed`.

The core primitive is exposed directly and supports **variable resample size**
(m ≠ n, with or without replacement):

```python
rng = np.random.default_rng(seed)
drawn  = bs.draw(rng, source_idx, size=None, replace=True)  # size defaults to len(source_idx)
counts = bs.draw_counts(drawn, universe_size)
oob    = bs.out_of_bag(drawn, source_idx)
subset = bs.select(X, idx)   # X[idx] for ndarrays, [X[i] for i in idx] for a list of bags
```

### Seeds

`bootstraptools` uses one master seed which spawns deterministic independent named process seeds. For per-replicate model-fit seeds (what models consume internally), it is convention to use the derived model seed (separate from the resample seeds the procedures derive from the bootstrap seed).

```python
seeds = bs.derive_seeds(RANDOM_SEED, ["dataset", "model", "bootstrap"])
fit_seeds = bs.replicate_seeds(seeds["model"], n_replicates)
```

---

## Writing a run 

```python
run = bs.init(
    store, procedure="train_resample_holdout", dataset="bal", model_label="ULTRA",
    config={"rank": 4, "B": 100, "bootstrap_seed": seeds["bootstrap"]},
    tags=["fig3", "pooling"],
    store_model_state=False,        # opt-in: default off 
)

for plan in plans:
    Xf = bs.select(X, plan.fit_indices)
    model.fit(Xf, y[plan.fit_indices])
    p = model.predict_proba(bs.select(X, plan.eval_indices))[:, 1]
    run.log_replicate(
        plan,
        metrics={"brier": ..., "brier_skill": ..., "auroc": ...},
        y_true=y[plan.eval_indices], p=p,   # makes predictions.parquet
        arrays={"attn_entropy": ...},       # opt-in rich artifacts (npz)
        model_state={"W_k": ..., "M": ..., "b": ...},  # requires store_model_state=True
        model_fit_seed=fit_seeds[k],
    )

run.finish(summary={"mean_brier_skill": ...})   # or use `with bs.init(...) as run:`
```

`log_replicate` takes a `ResamplePlan` and pulls replicate index / seed / membership from it. There is no fixed schema so metric column names are whatever you pass. `bootstraptools` stores the named arrays and the runner/processing script knows what they mean.

---

## Querying results

```python
runs = bs.query_runs(store, {"tags": "fig3", "dataset": "bal"})   # 1 row / run
tbl  = bs.query_run_table(store, run_id, table="replicates")      # one run's table
all_ = bs.load_runs_table(store, {"tags": "fig3"}, table="replicates")  # concat + run_id col
```

`query_runs` filters (AND-ed) on `procedure`/`dataset`/`model_label`/`status`/
`run_id`, on `tags` (string or list), and on any `config.<key>`. Config and summary are flattened into `config.*` / `summary.*` columns.

To reconstruct per-replicate resample membership (needed for BCa):

```python
membership = bs.membership_matrix(store, run_id)   # dense (B, universe_size) counts
```

---

## Confidence intervals (compute-on-demand)

Currently there are six interchangeable methods, all **pure functions over a stored replicate distribution** `theta_boot`. BCa is the recommended default, but the choice is each analysis's to make.

```python
from bootstraptools import percentile, basic, normal, bca, studentized, bayesian_bootstrap, ci

percentile(theta_boot)                                  # empirical quantiles
basic(theta_boot, theta_hat)                            # reverse-percentile
normal(theta_boot, theta_hat)                           # theta_hat ± z·SD
bca(theta_boot, theta_hat, jackknife=jack_values)       # BCa, exact leave-one-out jackknife
bca(theta_boot, theta_hat, membership=membership)       # BCa, jackknife-after-bootstrap (no extra fits)
studentized(theta_boot, theta_hat, se_boot, se_hat)     # bootstrap-t
bayesian_bootstrap(per_obs_values)                      # Dirichlet-weighted (data-level input)

ci(theta_boot, theta_hat, method="bca", membership=membership, alpha=0.05)  # dispatcher
```

All return a `(low, high)` tuple. `alpha` is the total two-sided level (default 0.05 → 95%). **BCa** can use the jackknife-after-bootstrap acceleration (`membership=`), avoiding the *n* extra refits an exact leave-one-out jackknife
(`jackknife=`) would need. The acceleration is estimated from the existing B
replicates' resample composition. `bayesian_bootstrap` takes **per-observation values** (e.g. per-patient squared errors for Brier), not `theta_boot`, so it is called directly rather than through `ci`.

---

## Internal validation: optimism & .632 / .632+ bootstrap

Overfitting-corrected performance à la Harrell's `rms::validate`. Use the
`optimism_bootstrap` procedure: each replicate fits on a bootstrap resample of
the *n* rows and predicts on the **full original sample**; you also log the
apparent model (fit once on all *n* rows). The correction is then computed
downstream — the stored per-replicate predictions-on-originals plus the
membership counts recover both the apparent-on-resample and test-on-original
performance for any metric, so nothing is baked in at write time.

```python
# --- write ---
run = bs.init(store, procedure="optimism_bootstrap", dataset="bal",
              model_label="ULTRA", config={"B": 200, "bootstrap_seed": s})
run.log_apparent(y_true=y, p=apparent_p)          # full-data model on all n rows
for plan in bs.optimism_bootstrap(universe_size=n, n_replicates=200, bootstrap_seed=s):
    model.fit(bs.select(X, plan.fit_indices), y[plan.fit_indices])
    p_all = model.predict_proba(X)[:, 1]          # predict on ALL n originals
    run.log_replicate(plan, y_true=y, p=p_all)
run.finish()

# --- correct (downstream, compute-on-demand) ---
from sklearn.metrics import roc_auc_score, brier_score_loss
res = bs.optimism_from_run(store, run_id, brier_score_loss)
# -> {"corrected", "apparent", "optimism", "optimism_per_replicate"}
```

`corrected = apparent − mean(optimism)` is one uniform formula — no
higher/lower-is-better branch (the optimism term is self-signed). The whole
modelling procedure (variable selection, HP tuning, preprocessing that learns
from data) must be repeated inside each replicate's fit, or the optimism is
understated.

**Which estimator — chosen by your primary metric:**

| Primary metric | Use | Why |
|---|---|---|
| **Proper scoring rule** (Brier, log-loss) | `optimism_from_run` (Efron–Gong) | Metric-agnostic; the standard choice for proper scores, applies directly to Brier. |
| **Discontinuous / improper** (accuracy, 0/1 error) | `error_632_plus_from_run` (+ `error_632_from_run`) | The family `.632/.632+` was designed for; its no-information-rate / relative-overfitting machinery is cleanest for per-sample 0/1-type losses. |

```python
bs.error_632_plus_from_run(store, run_id, bs.zero_one_loss)
# -> {"estimate", "apparent", "oob", "no_information_rate",
#     "relative_overfitting_rate", "weight"}
```

`.632/.632+` take a **per-sample loss** `loss_fn(y, p) -> array` (helpers:
`bs.squared_error_loss`, `bs.zero_one_loss`); `.632+` uses the canonical
Efron–Tibshirani (1997) weight `w = 0.632/(1 − 0.368·R)`. The split is a
practical guide, not an absolute law (both *can* apply to any metric); it
reflects theoretical grounding plus small-sample bias — Efron–Gong/.632 run
slightly optimistic, `.632+` slightly pessimistic.

> **Uncertainty.** These return a **point** correction plus
> `optimism_per_replicate` as a **diagnostic** — the spread of that array is the
> Monte-Carlo noise of the bias-correction term, **not** a confidence interval on
> model performance, and is not exposed as one. A proper CI on the corrected
> estimate (location-shifted / double bootstrap, per Noma et al. 2021) is planned
> as a follow-up. (`rms`'s ABCLOC heuristic is deliberately not used — it
> under-covers rank-based indices.)

See `examples/optimism_runner.py` for the full path end-to-end.

---

## Comparing across runs (paired analyses)

Some downstream analyses need *extra* fits layered on a base replicate set. It is recommended to represent each such axis as a **separate run that shares the `bootstrap_seed`** with the base run. Identical `bootstrap_seed` gives identical
per-replicate `fit_indices` which makes the runs' replicates row-joinable on `replicate_idx` for paired comparisons. See `examples/reference_runner.py`,
which runs two models on a shared seed and forms a paired brier-skill
difference with its own CI.

---
