from pathlib import Path

import pandas as pd
import typer
import yaml
from joblib import load
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def score_model(model_path: Path, dataset_csv_path: Path):

    # use the same parameter as in training
    with open("params.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
        drop_col = params["train"]["drop"]
        features = params["train"]["features"]
        target = params["train"]["target"]

    model = load(model_path / "trainedmodel.pkl")

    test_data = pd.read_csv(dataset_csv_path / "testdata.csv")

    # drop unnused column
    test_data = test_data.drop(drop_col, axis=1)

    # split into features and target
    X = test_data[features]
    y_true = test_data[target]

    # make predictions
    y_pred = model.predict(X)

    # score model
    f1_score_value = metrics.f1_score(y_true, y_pred)

    # saving score
    with open(model_path / "latestscore.txt", "a") as f:
        f.write(f"{f1_score_value}")


if __name__ == "__main__":
    typer.run(score_model)
