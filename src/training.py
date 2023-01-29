from pathlib import Path
from typing import List
import json
import pandas as pd
import typer
import yaml
from joblib import dump
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils import preprocess_data

with open("config.json", "r") as f:
    config = json.load(f)

data_folder_path = Path(config["output_folder_path"])
model_folder_path = Path(config["output_model_path"])

data_file_path = data_folder_path / "finaldata.csv"
model_file_path = model_folder_path / "trainedmodel.pkl"


def train_model(
    data_file_path: Path = data_file_path,
    model_file_path: Path = model_file_path,
):

    # create model folder
    Path(model_file_path).parent.mkdir(exist_ok=True)

    with open("params.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
        drop_col = params["train"]["drop"]
        features = params["train"]["features"]
        target = params["train"]["target"]

    X, y = preprocess_data(data_file_path)

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
    print(model_file_path)
    dump(lr, model_file_path)


if __name__ == "__main__":
    typer.run(train_model)
