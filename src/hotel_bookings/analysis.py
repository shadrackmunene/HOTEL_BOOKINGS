"""Cancellation cost estimate, association analysis and a logistic model."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

# Assumptions carried over from the original analysis (44,224 cancellations x $200 x 3.4)
COST_PER_NIGHT = 200
NIGHTS_PER_CANCELLATION = 3.4


def cancellation_counts(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["is_canceled"].value_counts().rename_axis("is_canceled").reset_index(name="n")
    counts["prop"] = counts["n"] / counts["n"].sum()
    return counts


def cancellation_cost(
    n_cancellations: int,
    cost_per_night: float = COST_PER_NIGHT,
    nights: float = NIGHTS_PER_CANCELLATION,
) -> float:
    return n_cancellations * cost_per_night * nights


def binarize(df: pd.DataFrame, target: str = "is_canceled", bins: int = 4) -> pd.DataFrame:
    """One-hot encode categoricals and quantile-bin numerics (target kept as 0/1)."""
    features = df.drop(columns=[target])
    parts = []
    for col in features.columns:
        s = features[col]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > bins:
            s = pd.qcut(s, q=bins, duplicates="drop").astype(str)
        elif pd.api.types.is_numeric_dtype(s):
            s = s.astype(str)
        parts.append(pd.get_dummies(s, prefix=col).astype(int))
    return pd.concat([df[[target]], *parts], axis=1)


def correlate_with_target(df: pd.DataFrame, target: str = "is_canceled") -> pd.Series:
    """Correlation of each binarized feature with the target, strongest first."""
    b = binarize(df, target)
    corr = b.corr()[target].drop(target).dropna()
    return corr.reindex(corr.abs().sort_values(ascending=False).index)


def auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via the rank-sum (Mann-Whitney) formula."""
    ranks = pd.Series(score).rank().to_numpy()
    n_pos = int(np.sum(y_true == 1))
    n_neg = len(y_true) - n_pos
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_logit(df: pd.DataFrame, target: str = "is_canceled"):
    """Logistic regression (GLM, Binomial) with in-sample AUC."""
    # reservation_status leaks the outcome, so it is excluded.
    X = pd.get_dummies(df.drop(columns=[target, "reservation_status"]), drop_first=True)
    X = sm.add_constant(X.astype(float))
    model = sm.GLM(df[target], X, family=sm.families.Binomial()).fit()
    return model, auc(df[target].to_numpy(), model.predict(X).to_numpy())
