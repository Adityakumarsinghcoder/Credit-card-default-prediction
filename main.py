import os
import pandas as pd

from src.generate_data import generate_dataset
from src.preprocessing import clean_data, engineer_features, get_feature_target_split
from src.eda import run_eda
from src.train_model import run_training_pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_card_data.csv")
EDA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eda_plots")
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "evaluation_plots")
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_data() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset from {DATA_PATH}")
        return pd.read_csv(DATA_PATH)
    print("No dataset found — generating a synthetic 30,000-record dataset...")
    df = generate_dataset()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df


def main():
    print("=" * 60)
    print("CREDIT CARD DEFAULT PREDICTION SYSTEM")
    print("=" * 60)

    # 1. Load data
    df = load_data()
    print(f"\nLoaded {len(df):,} records with {df.shape[1]} columns")

    # 2. Clean + engineer features
    df_clean = clean_data(df)
    df_features = engineer_features(df_clean)
    print(f"After cleaning + feature engineering: {df_features.shape[1]} columns, {len(df_features):,} rows")

    # 3. EDA
    run_eda(df_features, EDA_OUTPUT_DIR)

    # 4. + 5. Train, tune, evaluate
    X, y = get_feature_target_split(df_features, target_col="default")
    results = run_training_pipeline(X, y, EVAL_OUTPUT_DIR, MODEL_DIR)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(results.round(4))


if __name__ == "__main__":
    main()
