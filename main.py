# %% Imports
import json

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

from bootstrap.orchestrator import run

# %% Import datasets

np.random.seed(42)
n = 200

age = np.random.normal(55, 12, n)
bmi = np.random.normal(27, 5, n)
bp = np.random.normal(130, 20, n)
cholesterol = np.random.normal(200, 40, n)
glucose = np.random.normal(100, 25, n)

logit = (
    -6
    + 0.03 * age
    + 0.05 * bmi
    + 0.02 * bp
    + 0.005 * cholesterol
    + 0.01 * glucose
)
prob = 1 / (1 + np.exp(-logit))
disease = (np.random.rand(n) < prob).astype(int)

patient_ids = [f"Patient{i+1}" for i in range(n)]

df = pl.DataFrame({
    "patient_id": patient_ids,
    "age": age,
    "bmi": bmi,
    "bp": bp,
    "cholesterol": cholesterol,
    "glucose": glucose,
    "disease": disease,
})

# %% Perform data processing for linear baselines

feature_cols = ["age", "bmi", "bp", "cholesterol", "glucose"]

df = df.with_columns([
    ((pl.col(c) - pl.col(c).mean()) / pl.col(c).std()).alias(c)
    for c in feature_cols
])

X = df.select(feature_cols).to_numpy()
y_series = df["disease"]
y = y_series.to_numpy()

# %% Define a hyperparameter resolver for logistic regression

def lr_cv_resolver(X_train, y_train, _model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_cv = LogisticRegressionCV(
        Cs=10,
        cv=cv,
        solver="lbfgs",
        scoring="neg_log_loss",
        max_iter=1000,
        random_state=42,
    )
    lr_cv.fit(X_train, y_train)

    best_C = float(lr_cv.C_[0])

    model = LogisticRegression(
        C=best_C,
        solver="lbfgs",
        max_iter=1000,
    )
    hparams = {"C": best_C}

    return model, hparams

# %% Run bootstrap with CV resolver

# It is important to note that dataset must be able to be integer array indexed
# The orchestrator will slice it according to the bootstrap indices rather than label lookup.
model = LogisticRegression()
result = run(X=X, y=y,
    model=model,
    model_label="LogisticRegressionCV",
    n_replicates=10,
    random_seed=13,
    hparam_resolver=lr_cv_resolver,
    collect_inbag=True,
)

# %% Inspect metadata results

# Lookup table for mapping sample_idx back to patient_id
patient_ids = df["patient_id"].to_numpy()
pid_lookup = pl.DataFrame({
    "sample_idx": np.arange(len(patient_ids), dtype=np.uint32),
    "patient_id": patient_ids,
})

print(result.metadata.select(
    "replicate_idx", "status", "cv_duration_s", "fit_duration_s", "hparams"
).head(10))

# Inspect traceback error (if any)
#print(result.metadata["error_msg"][0])

# %% Inspect hyperparameter results

c_values = (
    result.metadata
    .filter(pl.col("status") == "success")
    .with_columns(
        pl.col("hparams")
        .map_elements(lambda s: json.loads(s)["C"], return_dtype=pl.Float64)
        .alias("C")
    )
)

print("\nSelected C across replicates:")
print(f"  mean:   {c_values['C'].mean():.4f}")
print(f"  median: {c_values['C'].median():.4f}")
print(f"  std:    {c_values['C'].std():.4f}")
print(f"  range:  [{c_values['C'].min():.4f}, {c_values['C'].max():.4f}]")

# %% OOB performance results

oob = result.oob_probabilities.join(pid_lookup, on="sample_idx").drop("sample_idx")
inbag = result.inbag_probabilities.join(pid_lookup, on="sample_idx").drop("sample_idx")

oob_summary = (
    oob
    .group_by("patient_id")
    .agg(
        pl.col("y_true").first(),
        pl.col("prob_1").mean().alias("mean_prob"),
        pl.col("prob_1").count().alias("n_appearances"),
    )
    .sort("patient_id")
)

y_true = oob_summary["y_true"].to_numpy()
p_hat = oob_summary["mean_prob"].to_numpy()

brier = np.mean((y_true - p_hat) ** 2)
print(f"\nBrier score (OOB): {brier:.4f}")
print(f"Mean prob separation: {np.mean(p_hat[y_true == 1]) - np.mean(p_hat[y_true == 0]):.4f}")

# %% Inbag performance results (if collected)
inbag_summary = (
    inbag
    .group_by("patient_id")
    .agg(
        pl.col("y_true").first(),
        pl.col("prob_1").mean().alias("mean_prob"),
        pl.col("prob_1").count().alias("n_appearances"),
    )
    .sort("patient_id")
)

y_true = inbag_summary["y_true"].to_numpy()
p_hat = inbag_summary["mean_prob"].to_numpy()

brier = np.mean((y_true - p_hat) ** 2)
print(f"\nBrier score (In-bag): {brier:.4f}")
print(f"Mean prob separation: {np.mean(p_hat[y_true == 1]) - np.mean(p_hat[y_true == 0]):.4f}")


# %% Inspect naive optimism (inbag - oob)

combined = (
    oob_summary.select("patient_id", "y_true", pl.col("mean_prob").alias("oob_prob"))
    .join(
        inbag_summary.select("patient_id", pl.col("mean_prob").alias("inbag_prob")),
        on="patient_id",
    )
    .with_columns(
        (pl.col("inbag_prob") - pl.col("oob_prob")).alias("optimism")
    )
)

print(combined)
# %%
