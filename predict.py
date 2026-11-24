"""Predict malignant / benign for tumour measurements.

    python predict.py patients.csv        # CSV with the 30 feature columns (see below)
    python predict.py --demo              # try it on 6 rows of the built-in dataset

The CSV needs the same 30 columns as sklearn's load_breast_cancer
(for example "mean radius", ..., "worst fractal dimension"). Extra columns are ignored.
The saved pipeline (StandardScaler + SVM) was trained with scikit-learn 1.6.1.
"""
import argparse, os
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {0: "malignant", 1: "benign"}  # same coding as sklearn's dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?", help="CSV file with the 30 feature columns")
    ap.add_argument("--demo", action="store_true", help="use 6 rows of the built-in dataset")
    a = ap.parse_args()

    model = joblib.load(os.path.join(HERE, "disease_model.joblib"))
    cols = list(model.feature_names_in_)
    truth = None
    if a.demo:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        X, truth = d.data.iloc[::100][cols], d.target.iloc[::100]
    elif a.csv:
        X = pd.read_csv(a.csv)
        missing = [c for c in cols if c not in X.columns]
        if missing:
            raise SystemExit(f"missing columns: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        X = X[cols]
    else:
        ap.error("give a CSV file or --demo")

    pred = model.predict(X)
    score = model.decision_function(X)  # > 0 means benign, < 0 means malignant
    for i, (p, s) in enumerate(zip(pred, score)):
        line = f"row {i}: {LABELS[int(p)]} (SVM score {s:+.2f})"
        if truth is not None:
            line += f" | true label: {LABELS[int(truth.iloc[i])]}"
        print(line)
    if a.demo:
        print("Note: demo rows come from the full dataset, so some were part of the training data.")


if __name__ == "__main__":
    main()
