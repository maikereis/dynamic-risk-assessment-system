from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
import yaml
from sklearn.metrics import confusion_matrix

from diagnostics import model_predictions


def score_model(dataset_csv_path: Path, output_model_path: Path):

    with open("params.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
        drop_col = params["train"]["drop"]
        features = params["train"]["features"]
        target = params["train"]["target"]

    data = pd.read_csv(dataset_csv_path)

    data = data.drop(drop_col, axis=1, errors="ignore")

    X = data[features]
    y_true = data[target]

    y_pred = model_predictions(X)

    fig, ax = plt.subplots(figsize=(4, 4))

    conf_mat = confusion_matrix(y_true, y_pred)

    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["no-exited", "exited"])
    ax.yaxis.set_ticklabels(["no-exited", "exited"])
    plt.savefig(output_model_path / "confusionmatrix.png")


if __name__ == "__main__":
    typer.run(score_model)
