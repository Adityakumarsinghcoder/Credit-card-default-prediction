import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid")


def run_eda(df: pd.DataFrame, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Dataset overview ---")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nDefault rate: {df['default'].mean():.2%}")

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="default", data=df, palette=["#4C72B0", "#C44E52"])
    plt.title("Default vs Non-Default (Class Balance)")
    plt.xlabel("Default (1 = Yes, 0 = No)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_class_balance.png"), dpi=150)
    plt.close()

    # 2. Age distribution by default status
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="AGE", hue="default", bins=30, kde=True, element="step")
    plt.title("Age Distribution by Default Status")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_age_distribution.png"), dpi=150)
    plt.close()

    # 3. Credit limit vs default
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="default", y="LIMIT_BAL", data=df, palette=["#4C72B0", "#C44E52"])
    plt.title("Credit Limit by Default Status")
    plt.xlabel("Default (1 = Yes, 0 = No)")
    plt.ylabel("Credit Limit (NT$)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_limit_bal_by_default.png"), dpi=150)
    plt.close()

    # 4. Correlation heatmap (numeric columns)
    plt.figure(figsize=(12, 9))
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.3)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_correlation_heatmap.png"), dpi=150)
    plt.close()

    # 5. Repayment status (PAY_0) vs default rate
    plt.figure(figsize=(6, 4))
    rate_by_pay = df.groupby("PAY_0")["default"].mean().reset_index()
    sns.barplot(x="PAY_0", y="default", data=rate_by_pay, color="#C44E52")
    plt.title("Default Rate by Most Recent Repayment Status (PAY_0)")
    plt.xlabel("PAY_0 (repayment delay in months)")
    plt.ylabel("Default Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_default_rate_by_pay_status.png"), dpi=150)
    plt.close()

    print(f"\nEDA plots saved to: {output_dir}")
