import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42


def generate_dataset(n_samples: int = 30000, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    # --- Demographics -----------------------------------------------------
    limit_bal = rng.choice(
        [10000, 20000, 30000, 50000, 80000, 100000, 150000,
         200000, 300000, 500000, 1000000],
        size=n_samples,
        p=[0.05, 0.08, 0.10, 0.15, 0.13, 0.12, 0.12, 0.10, 0.08, 0.05, 0.02],
    )
    sex = rng.choice([1, 2], size=n_samples, p=[0.4, 0.6])
    education = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.47, 0.16, 0.02])
    marriage = rng.choice([1, 2, 3], size=n_samples, p=[0.45, 0.52, 0.03])
    age = rng.integers(21, 70, size=n_samples)

    # --- Repayment status (PAY_0 is most recent month) --------------------
    # Skewed toward on-time/paid-duly (-1, 0) with a tail of delayed payments
    pay_cols = {}
    base_risk = rng.normal(0, 1, size=n_samples)  # latent "riskiness" per customer
    for i, col in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        noise = rng.normal(0, 1, size=n_samples)
        raw = base_risk * 1.4 + noise
        # map continuous latent value into the discrete PAY_x coding
        pay_status = np.digitize(raw, bins=[-1.5, -0.5, 0.5, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2]) - 2
        pay_status = np.clip(pay_status, -1, 8)
        pay_cols[col] = pay_status

    # --- Bill and payment amounts ------------------------------------------
    bill_cols = {}
    pay_amt_cols = {}
    prev_bill = limit_bal * rng.uniform(0.05, 0.6, size=n_samples)
    for i in range(1, 7):
        drift = rng.normal(1.0, 0.15, size=n_samples)
        bill = np.clip(prev_bill * drift, 0, limit_bal * 1.2)
        bill_cols[f"BILL_AMT{i}"] = bill.astype(int)
        # customers with worse repayment status pay back less of their bill
        risk_factor = np.clip(1 - (base_risk * 0.15), 0.05, 1.0)
        payment = np.clip(bill * rng.uniform(0.1, 0.9, size=n_samples) * risk_factor, 0, None)
        pay_amt_cols[f"PAY_AMT{i}"] = payment.astype(int)
        prev_bill = bill

    # --- Target: default next month ----------------------------------------
    # Combine latent risk + high utilization + late payments into a default probability
    utilization = bill_cols["BILL_AMT1"] / np.maximum(limit_bal, 1)
    recent_delay = np.clip(pay_cols["PAY_0"], -1, 8)

    logit = (
        -2.2
        + 0.9 * base_risk
        + 0.55 * recent_delay
        + 1.3 * utilization
        - 0.000002 * limit_bal
        + 0.01 * (age - 35) / 10
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, np.clip(prob_default, 0.02, 0.95))

    # --- Assemble dataframe --------------------------------------------
    df = pd.DataFrame({
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        **pay_cols,
        **bill_cols,
        **pay_amt_cols,
        "default": default,
    })

    # Inject a small amount of realistic missing/noisy data for cleaning practice
    noise_idx = rng.choice(df.index, size=int(0.01 * n_samples), replace=False)
    df.loc[noise_idx, "MARRIAGE"] = 0  # "unknown" category, as in the real dataset

    return df


def main():
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "credit_card_data.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} records -> {out_path}")
    print(f"Default rate: {df['default'].mean():.2%}")


if __name__ == "__main__":
    main()
