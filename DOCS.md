# bootstraptools — methods reference

Detailed, documentation of every bootstrap **procedure**, every **point estimator**, and every **confidence-interval** method in the package, with guidance on *which pipeline to use when* (sample size, model class, metric, compute budget) and the statistical guarantees and pitfalls of each.

For motivation, install, the data model, and the write/query API, see [README.md](README.md). This document assumes you already know how to write and query a run.

Contents:

1. [Why bootstrap (vs cross-validation)](#1-why-bootstrap-vs-cross-validation)
2. [Bootstrap procedures](#2-bootstrap-procedures)
3. [Point estimators of corrected performance](#3-point-estimators-of-corrected-performance)
4. [Confidence intervals over a replicate distribution](#4-confidence-intervals-over-a-replicate-distribution)
5. [Confidence intervals on the optimism-corrected estimate](#5-confidence-intervals-on-the-optimism-corrected-estimate)
6. [Choosing a pipeline](#6-choosing-a-pipeline)
7. [Reproducibility, paired runs, storage](#7-reproducibility-paired-runs-storage)
8. [References](#8-references)

---

## 1. Why bootstrap (vs cross-validation)

The bootstrap treats the observed sample as a stand-in for the population and resamples *with replacement* to imitate drawing fresh datasets.

> The population is to the sample as the sample is to the bootstrap samples. 

Compared to k-fold cross-validation and single train/test splits, this yields two significant things that matter for model evaluation:

- **A full uncertainty distribution, not just a point estimate.** Each replicate yields a statistic. The collection `theta_boot` is an estimate of that statistic's sampling distribution. Confidence intervals, standard errors, and paired differences all fall out of it. k-fold CV gives one number per fold with no principled interval (fold estimates are strongly dependent, so their spread is not a valid CI).
- **No permanent loss of data to a holdout.** Every model is fit on ~*n* rows and every observation participates. For small *n* (where a 20% holdout is painful) this is a large efficiency gain, and it makes the overfitting-corrected estimators of #3 possible. In effect, they use all the data yet still estimate out-of-sample performance.

Nevertheless, there are some trade-offs to keep in mind when using the bootstrap. The ordinary (0.632-style) bootstrap reuses each resample's own rows for evaluation, so naive in-sample scoring is optimistic. This is exactly what the corrections in #3/#5 exist to remove; however, they can often be more computationally expensive and/or subject to their own bias. Moreover, as bootstrap CIs are asymptotic, small samples need more careful methods.

References: Efron & Tibshirani, *An Introduction to the Bootstrap* (1993);
Harrell, *Regression Modeling Strategies* (RMS) #5.

---

## 2. Bootstrap procedures

A **procedure** turns the shared observation universe (`arange(n)`) plus a seed into a list of per-replicate `ResamplePlan`s (`replicate_idx`, `fit_indices`, `eval_indices`, `fit_counts`, `resample_seed`, `procedure`). All procedures key `fit_counts` to the full universe, so every run shares one membership schema regardless of procedure. A runner picks **one procedure per run**.

### `inbag_oob(universe_size, n_replicates, bootstrap_seed, size=None, replace=True)`

Fit on an in-bag resample of *n* rows; evaluate on that draw's **out-of-bag complement**. The eval set *varies* per replicate. Answers: *how sensitive are predictions to which observations land in vs. out of the resample?* This is the substrate for `.632`/`.632+` (#3), whose OOB error is exactly the leave-one-out bootstrap error over these complements.

### `train_resample_holdout(train_idx, val_idx, universe_size, n_replicates, bootstrap_seed, size=None, replace=True)`

Resample the **training fold** with replacement, refit, and evaluate every replicate on the **same fixed validation fold**. The eval set is *constant* across replicates. Answers: *how sensitive is performance on a fixed target population to training-sample composition?* Use when you have a designated holdout/target set and want its performance's sampling distribution.

### `optimism_bootstrap(universe_size, n_replicates, bootstrap_seed, size=None, replace=True)`

Efron–Gong optimism bootstrap. Each replicate fits on a bootstrap resample of the *n* rows and predicts on the **full original sample** (`eval = arange(n)`, constant). Pair it with `Run.log_apparent(y, p)` (the full-data model's predictions on all *n* rows). Downstream, the stored per-replicate predictions-on-originals plus the membership counts recover **both** the apparent-on-resample and test-on-original performance for *any* metric. This enables the optimism correction (#3) to be computed on demand so that nothing is baked in at write time.

Sources: Efron (1983); Gong (1986); Efron & Tibshirani (1993) #17.6; Harrell RMS #5.3.5.

### `double_optimism_bootstrap(universe_size, n_outer, n_inner, bootstrap_seed, size=None, replace=True)`

Nested (two-stage) resampling for the method-2 CI (#5). Yields `DoubleResamplePlan(outer_idx, outer_plan, inner_plans)`. The **outer** plan draws from the original universe and each **inner** plan draws from the outer resample's *local* positions `[0, n)` and evaluates on the full outer resample. In effect, it re-runs the optimism procedure *treating each outer resample as its own "original" data*. The whole R×B seed tree derives from one `bootstrap_seed`. Architecturally, because it needs R×B fits, results are stored **compact** (one corrected estimate per outer replicate, see #5). 

Source: Noma et al. (2021),`predboot::pred.ML2`.

> **Repeat the whole modelling procedure inside each replicate.** Variable
> selection, hyper-parameter tuning, and any preprocessing that learns from the
> data must be refit *within* each replicate's fit, or the estimated optimism is
> understated (Steyerberg et al.; Harrell RMS). Selecting a final feature set on
> the full data and only bootstrapping the coefficients silently omits the
> selection component of overfitting.

The variable-size primitive under all procedures (`bs.draw` / `draw_counts` / `out_of_bag` / `select`) supports `m ≠ n` and with/without replacement (although should be done with replacement for optimism correction).

---

## 3. Point estimators of corrected performance

All three of the following methods remove the in-sample optimism of apparent performance. They differ in their mechanism and small-sample bias. All are pure functions over an `optimism_bootstrap` run's artifacts and are **metric-agnostic** (you pass the metric or per-sample loss).

### Efron–Gong optimism — `optimism_correction` / `optimism_from_run`

```
optimism_b = metric(y, p_b, sample_weight=counts_b)  −  metric(y, p_b)
           = P_boot,b  −  P_orig,b
corrected  = metric(y, apparent_p)  −  mean_b(optimism_b)
```

One **uniform, self-signed** formula (the optimism term carries its own sign, so it applies identically to Brier and to AUROC). `metric` has signature `metric(y_true, p, sample_weight=None) -> float` (sklearn's `brier_score_loss`, `roc_auc_score`, `log_loss`, … all qualify). Returns `{corrected, apparent, optimism, optimism_per_replicate}`.

> `optimism_per_replicate` is a **diagnostic**, never a CI: its spread is the
> Monte-Carlo noise of the bias-correction *term*, not the sampling uncertainty
> of model performance. For an interval, use #5.

### `.632` and `.632+` — `error_632[_plus][_from_run]`

For **per-sample-loss** metrics `loss_fn(y, p) -> (n,)` (helpers:
`bs.squared_error_loss`, `bs.zero_one_loss`):

```
.632 :  err = 0.368·err_app + 0.632·eps0
.632+:  R   = (eps0 − err_app) / (gamma − err_app)  clamped to [0, 1]
        w   = 0.632 / (1 − 0.368·R)          # canonical Efron–Tibshirani 1997 weight
        err = (1 − w)·err_app + w·min(eps0, gamma)
```

`eps0` is the leave-one-out bootstrap (OOB) error reconstructed from membership;
`gamma` is the no-information rate via the all-pairs formula
`gamma = mean_{i,j} loss(y_i, p_j)` (computed from the apparent predictions, so `.632/.632+` need **no** artifact beyond the optimism run). 

Source: Efron & Tibshirani (1997). Note this uses the canonical weight, not the mlxtend variant.

### Which point estimator?

| Primary metric | Estimator | Why |
|---|---|---|
| **Proper scoring rule** (Brier, log-loss) | Efron–Gong (`optimism_from_run`) | Standard for proper scores and applies directly / metric-agnostically. |
| **Discontinuous / improper** (accuracy, 0/1 error) | `.632+` (`error_632_plus_from_run`) | Designed for this scoring family as the no-information-rate / relative-overfitting machinery is cleanest for 0/1-type losses. |

A useful practical guide, but explicitly not a law (both can be used with any metric, just have statistical divergences related to over- and under- bias). Small-sample bias runs in opposite directions where the Efron–Gong/`.632` is slightly **optimistic**, but the `.632+` is slightly **pessimistic**.

---

## 4. Confidence intervals over a replicate distribution

Six interchangeable methods in `bootstraptools.uq`, of which they are all **pure functions over a stored `theta_boot` (B,)** (and, where noted, the point estimate / membership). All return a `(low, high)` tuple representing the statistic coverage (but not guaranteed to always be 95%). `alpha` is the total two-sided level (default 0.05 -> 95%). These are for an *ordinary* replicate statistic (e.g. a `train_resample_holdout` run's per-replicate AUROC), **not** for any optimism-*corrected* estimate (#5).

| Method | Call | Notes |
|---|---|---|
| Percentile | `percentile(theta_boot)` | Empirical quantiles. |
| Basic | `basic(theta_boot, theta_hat)` | Reverse-percentile (reflects bias). |
| Normal | `normal(theta_boot, theta_hat)` | `theta_hat ± z·SD`, `ddof=1`; plain (not bias-corrected). |
| **BCa** *(default)* | `bca(theta_boot, theta_hat, membership=…)` | Bias-corrected + accelerated. |
| Studentized | `studentized(theta_boot, theta_hat, se_boot, se_hat)` | Bootstrap-*t*; needs per-replicate SEs. |
| Bayesian bootstrap | `bayesian_bootstrap(per_obs_values)` | Rubin (1981), Dirichlet(1,…,1) over **per-observation** values. |

`ci(theta_boot, theta_hat, method=…, **kwargs)` dispatches all but `bayesian_bootstrap` (whose input is data-level, not `theta_boot`).

**Conventions are pinned to SciPy** (`scipy.stats.bootstrap`) and validated numerically against it: `z0 = Φ⁻¹((#{θ*<θ̂}+#{θ*≤θ̂})/2B)` (finite at the edges), acceleration sign `d_i = θ̄_(·) − θ̂_(i)`, linear-interpolation quantiles.

**BCa acceleration without refits.** `bca(..., membership=membership)` uses the **jackknife-after-bootstrap** (JAB): for each observation *i*, θ̃_(i) is the mean of `theta_boot` over replicates where *i* was absent (`count == 0`), restricted to observations resampled ≥1× and absent ≥1×. This avoids the *n* extra refits an exact leave-one-out jackknife (`bca(..., jackknife=…)`) needs.
`bs.membership_matrix(store, run_id)` rebuilds the dense `(B, n)` counts that feed it. 

Source: Efron (1992); Efron & Tibshirani (1993).

---

## 5. Confidence intervals on the optimism-corrected estimate

The #4 methods do **not** give a CI on the *corrected* estimate of #3. In other words, the corrected value is not a percentile of `theta_boot`. The following three methods do, forming a regime-selected menu. All operate on an `optimism_bootstrap` run (methods 1 and ABCLOC are pure downstream functions over its stored artifacts whereas method 2 needs its own nested `double_optimism_bootstrap` run).

Notation: `p_boot,b = metric(y, p_b, sample_weight=counts_b)` (apparent on resample), `p_orig,b = metric(y, p_b)` (bootstrap model on original), `optimism = mean_b(p_boot,b − p_orig,b)`, `corrected = apparent − optimism`.

### Method 1 — location-shifted — `optimism_location_shifted_ci` / `optimism_ci_from_run`

```
q_lo, q_hi = percentile({p_boot,b}, alpha)      # plain percentile of the apparent-on-resample dist.
ci         = (q_lo − optimism, q_hi − optimism) # shift the whole interval by −optimism
```

Faithful to Noma et al. (2021) method 1 (`predboot::pred.ML`, lines 55/57–58/89). Returns `{corrected, apparent, optimism, ci, apparent_ci, alpha}` where `apparent_ci` is the unshifted interval. 

- **Cost:** O(B) (the base optimism run).
- **Coverage:** good in large samples but **under-covers small samples** (~70–80% at nominal 95%) because it treats `optimism` as a fixed shift and ignores that term's own variability.
- **Caveat:** the interval is `(percentile of p_boot) − optimism`, formed separately from the point estimate, so it is **not** guaranteed to be centered on or even to contain `corrected`. 

### ABCLOC ("sd2rev wtd4") — `optimism_abcloc_ci` / `optimism_abcloc_ci_from_run`

Harrell's fast asymmetric interval around the ordinary corrected estimate.

```
x_b       = p_boot,b − 1.25·p_orig,b            # "wtd4": down-weight the (low-variance) train term
m         = mean(x);  bottom = {x_b ≤ m}, top = {x_b ≥ m}   # dualSD, equal-to-mean in both
sd_bottom = rms(bottom − m, ddof=1);  sd_top = rms(top − m, ddof=1)
ci        = (corrected − sd_top·z,  corrected + sd_bottom·z)   # sd2rev: reversed SDs, z = Φ⁻¹(1−alpha/2)
```

Verified against Harrell's `bootcal` post and Hmisc `dualSD` (per-side RMS from the *overall* mean, Bessel denominator, `nmin=10` fallback to the ordinary SD). Returns `{corrected, apparent, optimism, ci, sd_bottom, sd_top, alpha}`. Unlike methods 1 and 2, ABCLOC **always contains** the point estimate.

- **Cost:** O(B) (the base optimism run).
- **Coverage:** Brier ~94.7%, calibration slope ~95.5% (good). **Rank-based indices (Somers' Dxy / AUROC) under-cover (~85% at small n).**
- **Scope / caveats:** use for **proper scoring rules only**. The 1.25 constant is **empirically tuned** (proper-score / regularized-logistic flavored) which we should treat as a heuristic outside that regime. It was the winner of Harrell's 27-method comparison which decisively did **not** include either Noma method.

### Method 2 — double (two-stage) bootstrap — `double_bootstrap_ci` / `double_bootstrap_ci_from_run`

The rigorous but expensive CI. Run a `double_optimism_bootstrap` run: for each outer replicate *r*, compute a full Efron–Gong `corrected` treating the outer resample as the data (`theta_corr_r`), and store it as a metrics-only row. Then:

```
ci = percentile({theta_corr_r}, alpha)          # plain percentile, no shift
```

Faithful to Noma et al. (2021) method 2 (`predboot::pred.ML2`). Returns `{ci, outer_mean, outer_median, n_outer, alpha}`.

- **Cost:** O(R×B) model fits.
- **Coverage:** ~95% across small and large samples as it captures **both** model-refit variability (outer loop) and optimism-estimation variability (inner loop), which is why it fixes method 1's small-sample under-coverage. The naive single-stage variant under-covers precisely because it omits the model-refit component.
- **Point estimate:** report the ordinary corrected value from a **paired** `optimism_bootstrap` run on the same `bootstrap_seed` (see #7). `outer_mean`/`outer_median` summarize the outer distribution and are *not* the point estimate.
- **Caveat:** the CI need not contain that point estimate. Inner evaluation is on the duplicate-containing outer resample, so `theta_corr_r` runs mildly optimistic relative to the original-data corrected. 

### The regime table

| Sample size | Primary metric | Recommended CI | Cost |
|---|---|---|---|
| Large (n large / EPV ≥ ~10) | any | **Method 1** (location-shifted) | O(B) |
| Small | **proper score** (Brier, calibration) | **ABCLOC** (or Method 2 if you can afford it) | O(B) |
| Small | **rank-based** (AUROC, Dxy) | **Method 2** — ABCLOC *and* method 1 both under-cover here | O(R×B) |


---

## 6. Choosing a pipeline

A decision path from a modelling question to a concrete set of calls.

1. **Do you need overfitting correction at all?**
   - *Comparing/plotting the sampling distribution of a statistic on a fixed target population* (e.g. a held-out fold) -> **no correction**. Use `train_resample_holdout` + a #4 CI (BCa default) on the per-replicate statistic.
   - *Reporting a single model's out-of-sample performance using all the data* -> **yes, correct**. Use `optimism_bootstrap` + #3 + #5 CI.

2. **Which corrected point estimate?** Proper score should use Efron–Gong. 0/1-type should use `.632+` (#3 table).

3. **Which CI on it?** Read #5's regime table as it is a matter driven by sample size, metric type, and your compute budget. When in doubt for a rank-based metric in a small sample, pay for **method 2** as it is the only one that confidently covers there.

4. **Model class / parameterization regime.** The corrections assume the *entire* fitting procedure is repeated per replicate (#2 box). Heavily regularized/tuned pipelines have larger optimism and *must* refit the tuning inside each replicate. ABCLOC's tuned constant was calibrated on proper-score / regularized-logistic settings which means for very different model classes (trees, deep nets), it is preferred to use method 1 (large n) or method 2 (small n) and treat ABCLOC as a fast sanity check only.

5. **Compute budget.** Methods 1/ABCLOC and all of #3/#4 are O(B) and reuse one run's artifacts (often computed after the fact). Method 2 is O(R×B) fits and its own run. It should really only be used for the cases where the cheaper methods are known to under-cover.

**Guarantees & pitfalls at a glance:**

- Bootstrap CIs are asymptotic which means small samples degrade coverage (method 2 is the small-sample-honest option as method 1 and ABCLOC have documented small-sample weaknesses).
- We can never present `optimism_per_replicate`'s spread as a CI (#3).
- Methods 1 and 2 can place the interval off the point estimate but ABCLOC always brackets it.
- There will be understated optimism if the modelling pipeline is not fully refit per replicate (#2 box).

---

## 7. Reproducibility, paired runs, storage

**Seeds.** One master seed spawns independent named process seeds. Procedures derive per-replicate `resample_seed`s from the `bootstrap_seed`, and model-fit seeds come from the derived `model` seed. Every replicate is reproducible in isolation.

```python
seeds = bs.derive_seeds(RANDOM_SEED, ["dataset", "model", "bootstrap"])
fit_seeds = bs.replicate_seeds(seeds["model"], n_replicates)
```

**Paired runs (shared `bootstrap_seed`).** Two runs given the same `bootstrap_seed` draw identical per-replicate `fit_indices`, so their replicates are row-joinable on `replicate_idx` for paired comparisons (e.g., model A vs. B at each resample, a method-2 CI paired with its point-estimate run etc.). This is the recommended representation for any "same resample, different fit" axis. See `examples/reference_runner.py` (paired brier-skill difference with its own CI) and `examples/double_bootstrap_runner.py` (method-2 CI paired with an `optimism_bootstrap` point estimate).

**Opt-in storage.** `store_model_state=True` at `init` enables `log_replicate(model_state=…)`. Nevertheless, rich per-replicate `arrays=` are always available. Both are written per-replicate (durable across crashes) as the parquet tables flush on `finish()`. Method-2 runs are deliberately **compact** (metrics-only outer rows) as the full metric-agnostic storage would be O(R×B×n).

**Examples.**

| Example | Shows |
|---|---|
| `examples/reference_runner.py` | `train_resample_holdout` + #4 BCa/percentile + paired difference |
| `examples/optimism_runner.py` | `optimism_bootstrap` + #3 + method-1 CI + `.632+` |
| `examples/double_bootstrap_runner.py` | `double_optimism_bootstrap` + method-2 CI, paired point estimate |

---

## 8. References

- Efron, B. (1983). Estimating the error rate of a prediction rule. *JASA*.
- Gong, G. (1986). Cross-validation, the jackknife, and the bootstrap. *JASA*.
- Efron, B. (1992). Jackknife-after-bootstrap standard errors. *JRSS-B*.
- Efron, B. & Tibshirani, R. (1993). *An Introduction to the Bootstrap*.
- Efron, B. & Tibshirani, R. (1997). Improvements on cross-validation: the .632+
  bootstrap method. *JASA*.
- Rubin, D. (1981). The Bayesian bootstrap. *Annals of Statistics*.
- Harrell, F. *Regression Modeling Strategies* (RMS), §5; `rms::validate`;
  "bootcal" post (ABCLOC / sd2rev wtd4); Hmisc `dualSD`.
- Steyerberg, E. et al. Internal validation of prediction models.
- Noma, H. et al. (2021). Confidence intervals of prediction accuracy measures …
  based on bootstrap-based optimism correction. *Statistics in Medicine*
  40(26):5691–5701, doi:10.1002/sim.9148; `predboot` (`pred.ML`, `pred.ML2`).
- SciPy `scipy.stats.bootstrap` (BCa/percentile conventions this package pins to).
