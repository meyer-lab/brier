# bootstraptools

> The population is to the sample as the sample is to the bootstrap samples

Local storage, query, and uncertainty-quantification infrastructure for
**bootstrap experiments** that evaluate regression and classification model
performance. A bootstrap runner writes one run (B replicates) to a local
store, and downstream processing scripts query it back and compute point
estimates and confidence intervals **on demand**.

`bootstraptools` is **model-agnostic**: it sees a model as a generic `fit` /
`predict_proba` object and stores whatever scalars and named arrays a runner
hands it. It is shaped around plain sklearn architectures, so other model
types may need gentle massaging (a `fit(X, y)` / `predict_proba(X) -> (n,
2)` shim and a way to slice examples) to plug in.

```bash
uv sync                                    # uv manages deps/venv
uv run pytest -q                           # full test suite
uv run python examples/reference_runner.py # end-to-end demo, prints CIs
```

## Data model

| Concept | Meaning | On disk |
|---|---|---|
| **Store** | A root directory holding many runs. | `<store>/runs/` |
| **Run** | One bootstrap experiment: B replicates for a `(dataset, model, procedure)` config, plus tags/config/summary. | `<store>/runs/<run_id>/` |
| **Replicate** | One refit + evaluation: scalar metrics, per-sample predictions, resample membership, seeds, optional rich arrays / model state. | rows/files within a run |

Per-run layout:

```
<store>/runs/<run_id>/
  meta.json            # tags, config, procedure, dataset, model_label, seeds,
                       # universe_size, status, n_replicates, summary
  replicates.parquet   # 1 row / replicate: seeds, fit metadata, scalar metrics
  predictions.parquet  # 1 row / (replicate, sample): sample_idx, y_true, p, split
  membership.parquet   # 1 row / (replicate, sample, count>0)
  apparent.parquet     # opt-in: the full-data model's predictions on all n rows
  arrays/replicate_XXXX.npz       # opt-in rich per-replicate arrays
  model_state/replicate_XXXX.npz  # opt-in full fitted params (store_model_state=True)
```

Tables are [polars](https://github.com/pola-rs/polars)/parquet, arrays are
npz. `arrays/` and `model_state/` are written **immediately** per replicate
(durable if a long run crashes).

## Writing a run

`init` -> per-replicate `log_replicate` -> `finish`. The runner owns the
fit/predict loop.

```python
import bootstraptools as bs

seeds = bs.derive_seeds(RANDOM_SEED, ["dataset", "model", "bootstrap"])
plans = bs.train_resample_holdout(
    train_idx, val_idx, universe_size=n, n_replicates=100,
    bootstrap_seed=seeds["bootstrap"],
)
fit_seeds = bs.replicate_seeds(seeds["model"], len(plans))

with bs.init(store, procedure="train_resample_holdout", dataset="bal",
             model_label="ULTRA",
             config={"rank": 4, "B": 100, "bootstrap_seed": seeds["bootstrap"]},
             tags=["fig3", "pooling"]) as run:
    for plan in plans:
        model.fit(bs.select(X, plan.fit_indices), y[plan.fit_indices])
        p = model.predict_proba(bs.select(X, plan.eval_indices))[:, 1]
        run.log_replicate(
            plan,
            metrics={"brier": ..., "auroc": ...},
            y_true=y[plan.eval_indices], p=p,   # -> predictions.parquet
            arrays={"attn_entropy": ...},       # opt-in rich artifacts
            model_fit_seed=fit_seeds[plan.replicate_idx],
        )
```

`log_replicate` pulls the replicate index / seed / membership from the
`ResamplePlan`. There is no fixed metric schema; column names are whatever
you pass. `bs.select(X, idx)` slices ndarrays (`X[idx]`) or lists of bags
(`[X[i] for i in idx]`). See the [procedures reference](reference/procedures.md)
for which procedure to pick, and the [resample reference](reference/resample.md)
for the variable-size draw primitives beneath them.

## Querying results

```python
runs = bs.query_runs(store, {"tags": "fig3", "dataset": "bal"})   # 1 row / run
tbl  = bs.query_run_table(store, run_id, table="replicates")      # one run's table
all_ = bs.load_runs_table(store, {"tags": "fig3"}, table="replicates")  # concat + run_id col
membership = bs.membership_matrix(store, run_id)                   # dense (B, n) counts (for BCa)
```

`query_runs` filters (AND-ed) on `procedure`/`dataset`/`model_label`/`status`/
`run_id`, on `tags`, and on any `config.<key>`. config/summary flatten into
`config.*` / `summary.*` columns.

## Comparing across runs (paired analyses)

Represent any "same resample, different fit" axis as **separate runs
sharing the `bootstrap_seed`**: identical seeds give identical per-replicate
`fit_indices`, so the runs' replicates are row-joinable on `replicate_idx`
for paired comparisons.

## Where to go next

Browse the [API reference](reference/resample.md) for details on every
public function and class, or read `DOCS.md` in the repository for a deeper
walkthrough of the bootstrap procedures and reproducibility model.
