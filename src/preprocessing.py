import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix invalid categorical codes and handle missing/duplicate data."""
    df = df.copy()

    # Drop exact duplicate rows
    df = df.drop_duplicates()

    # EDUCATION: fold undocumented codes (0, 5, 6) into "other" (4)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # MARRIAGE: fold unknown code (0) into "other" (3)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # AGE: clip unrealistic values
    df = df[(df["AGE"] >= 18) & (df["AGE"] <= 100)]

    # BILL_AMT / PAY_AMT: negative bill amounts are valid (credit balance),
    # but guard against absurd outliers using the 99.5th percentile
    for col in [c for c in df.columns if c.startswith("BILL_AMT") or c.startswith("PAY_AMT")]:
        upper = df[col].quantile(0.995)
        df[col] = df[col].clip(upper=upper)

    # Drop rows with missing target
    df = df.dropna(subset=["default"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional predictive features from the raw columns."""
    df = df.copy()

    pay_cols = [f"PAY_{i}" if i != 1 else "PAY_0" for i in range(0, 7) if f"PAY_{i}" in df.columns or i == 0]
    pay_cols = [c for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in df.columns]
    bill_cols = [c for c in df.columns if c.startswith("BILL_AMT")]
    pay_amt_cols = [c for c in df.columns if c.startswith("PAY_AMT")]

    # Average / max delay across the 6-month history
    df["AVG_PAY_DELAY"] = df[pay_cols].mean(axis=1)
    df["MAX_PAY_DELAY"] = df[pay_cols].max(axis=1)
    df["MONTHS_DELAYED"] = (df[pay_cols] > 0).sum(axis=1)

    # Credit utilization ratio (most recent bill / credit limit)
    df["UTILIZATION_RATIO"] = df["BILL_AMT1"] / df["LIMIT_BAL"].replace(0, np.nan)
    df["UTILIZATION_RATIO"] = df["UTILIZATION_RATIO"].fillna(0).clip(0, 3)

    # Average bill and payment amounts
    df["AVG_BILL_AMT"] = df[bill_cols].mean(axis=1)
    df["AVG_PAY_AMT"] = df[pay_amt_cols].mean(axis=1)

    # Payment-to-bill ratio: how much of the bill customers tend to pay back
    df["PAY_TO_BILL_RATIO"] = df["AVG_PAY_AMT"] / df["AVG_BILL_AMT"].replace(0, np.nan)
    df["PAY_TO_BILL_RATIO"] = df["PAY_TO_BILL_RATIO"].fillna(0).clip(0, 5)

    # Trend: is the customer's bill growing or shrinking over the 6 months?
    df["BILL_TREND"] = df["BILL_AMT1"] - df["BILL_AMT6"]

    return df


def get_feature_target_split(df: pd.DataFrame, target_col: str = "default"):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
