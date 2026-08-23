# Credit Card Default Prediction System

An end-to-end machine learning classification project that predicts whether a
credit card customer will default on their next payment, using demographic
and financial history data.

## Tech Stack
- Python 3
- Pandas / NumPy
- Scikit-learn
- Matplotlib / Seaborn

## Project Structure
```
credit_card_default_prediction/
├── data/
│   └── credit_card_data.csv      # dataset (generated or your own)
├── models/
│   └── best_model.pkl            # saved trained model (after running)
├── outputs/
│   ├── eda_plots/                # exploratory data analysis charts
│   └── evaluation_plots/         # confusion matrix, ROC curve, etc.
├── src/
│   ├── generate_data.py          # creates a synthetic dataset (skip if you have real data)
│   ├── preprocessing.py          # cleaning + feature engineering
│   ├── eda.py                    # exploratory data analysis + visualizations
│   └── train_model.py            # training, tuning, evaluation
├── main.py                       # runs the full pipeline end-to-end
├── requirements.txt
└── README.md
```

## How to Run in VS Code

1. Open this folder in VS Code (`File > Open Folder`).
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Mac/Linux
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the full pipeline:
   ```bash
   python main.py
   ```

This will:
- Generate/load the dataset (30,000 customer records)
- Run EDA and save charts to `outputs/eda_plots/`
- Clean data and engineer features
- Train Logistic Regression and Random Forest models
- Run hyperparameter tuning (GridSearchCV)
- Evaluate both models (accuracy, precision, recall, F1, ROC-AUC)
- Save the best model to `models/best_model.pkl`
- Save evaluation charts (confusion matrix, ROC curve, feature importance) to `outputs/evaluation_plots/`

## Using Your Own Data

If you have the real UCI "Default of Credit Card Clients" dataset (or similar),
drop it into `data/credit_card_data.csv` with the same column names used in
`src/generate_data.py`, and skip the generation step in `main.py`.

## Results

On the test split, the pipeline typically reports:
- Logistic Regression: ~80-82% accuracy
- Random Forest (tuned): ~82-84% accuracy, better recall on the minority
  (default) class

Exact numbers will vary slightly by random seed and data source.
