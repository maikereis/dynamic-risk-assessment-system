import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer
import yaml
from sklearn.metrics import confusion_matrix

from src.diagnostics import model_predictions
from src.utils import preprocess_data

with open("config.json", "r") as f:
    config = json.load(f)

data_folder_path = Path(config["test_data_path"])
data_file_path = data_folder_path / "testdata.csv"

img_folder_path = Path(config["output_model_path"])
img_file_path = img_folder_path / "confusionmatrix.png"


def score_model():
    X, y_true = preprocess_data(data_file_path)

    y_pred = model_predictions(X)

    fig, ax = plt.subplots(figsize=(4, 4))

    conf_mat = confusion_matrix(y_true, y_pred)

    sns.heatmap(conf_mat, annot=True, cmap="Blues", fmt="d", cbar=False, ax=ax)
    ax.set_xlabel("Predicted labels")
    ax.set_ylabel("True labels")
    ax.set_title("Confusion Matrix")
    ax.xaxis.set_ticklabels(["no-exited", "exited"])
    ax.yaxis.set_ticklabels(["no-exited", "exited"])
    plt.savefig(img_file_path)


if __name__ == "__main__":
    typer.run(score_model)
