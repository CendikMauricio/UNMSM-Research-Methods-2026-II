"""Reproducible pipeline for Wine Dataset classification.

Usage:
    PYTHONHASHSEED=0 python src/train.py --seed 42
"""

import argparse
import hashlib
import json
import random

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int) -> None:
    """Set the random sources used in this project."""
    random.seed(seed)
    np.random.seed(seed)


def file_md5(path: str) -> str:
    """Calculate the MD5 fingerprint of the dataset file."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def main(seed: int = 42, data_path: str = "data/wine.csv") -> dict:
    set_seed(seed)

    df = pd.read_csv(data_path)
    X = df.drop(columns="target")
    y = df["target"].astype(int)

    # Everything that learns from the data stays inside the Pipeline.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])

    # The seed controls how the folds are shuffled.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
        n_jobs=None
    )

    return {
        "seed": seed,
        "data_md5": file_md5(data_path),
        "accuracy_mean": round(float(scores["test_accuracy"].mean()), 5),
        "accuracy_sd": round(float(scores["test_accuracy"].std(ddof=1)), 5),
        "f1_macro_mean": round(float(scores["test_f1_macro"].mean()), 5),
        "f1_macro_sd": round(float(scores["test_f1_macro"].std(ddof=1)), 5)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(main(args.seed), ensure_ascii=False))
