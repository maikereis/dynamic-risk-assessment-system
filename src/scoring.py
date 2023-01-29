import json
from pathlib import Path

import typer
from joblib import load
from sklearn import metrics

from src.utils import preprocess_data

with open("config.json", "r") as f:
    config = json.load(f)

data_folder_path = Path(config["test_data_path"])
model_folder_path = Path(config["output_model_path"])

data_file_path = data_folder_path / "testdata.csv"
model_file_path = model_folder_path / "trainedmodel.pkl"
score_file_path = model_folder_path / "latestscore.txt"


def score_model(
    model_file_path: Path = model_file_path,
    data_file_path: Path = data_file_path,
    score_file_path: Path = score_file_path,
):

    model = load(model_file_path)

    X_test, y_test = preprocess_data(data_file_path)

    # make predictions
    y_pred = model.predict(X_test)

    # score model
    f1_score_value = metrics.f1_score(y_test, y_pred)

    if score_file_path != None:
        # saving score
        with open(score_file_path, "w") as f:
            f.write(f"{f1_score_value}")

    return f1_score_value


if __name__ == "__main__":
    try:
        typer.run(score_model)
    except ValueError as e:
        print(f"An error occurred: {e}")
