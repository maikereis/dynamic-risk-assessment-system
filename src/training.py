from pathlib import Path
from typing import List

import pandas as pd
import typer
import yaml
from joblib import dump
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model(
    dataset_csv_path: Path,
    model_path: Path,
):

    # create model folder
    Path(model_path).mkdir(exist_ok=True)

    with open("params.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
        drop_col = params["train"]["drop"]
        features = params["train"]["features"]
        target = params["train"]["target"]

    train_data = pd.read_csv(dataset_csv_path / "finaldata.csv")

    # drop unnused column
    train_data = train_data.drop(drop_col, axis=1)

    X = train_data[features]
    y = train_data[target]

    # use this logistic regression for training
    lr = LogisticRegression(
        C=1.0,
        class_weight=None,
        dual=False,
        fit_intercept=True,
        intercept_scaling=1,
        l1_ratio=None,
        max_iter=100,
        multi_class="auto",
        n_jobs=None,
        penalty="l2",
        random_state=0,
        solver="liblinear",
        tol=0.0001,
        verbose=0,
        warm_start=False,
    )

    # fit the logistic regression to your data
    lr.fit(X, y)

    # write the trained model to your workspace in a file called trainedmodel.pkl
    dump(lr, model_path / "trainedmodel.pkl")


if __name__ == "__main__":
    typer.run(train_model)
