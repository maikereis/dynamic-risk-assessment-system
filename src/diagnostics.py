import json
import subprocess
import timeit
from pathlib import Path

import pandas as pd
import pkg_resources
import typer
from joblib import load
from src.utils import preprocess_data

app = typer.Typer()

with open("config.json", "r") as f:
    config = json.load(f)

data_folder_path = Path(config["output_folder_path"])
model_folder_path = Path(config["output_model_path"])

data_file_path = data_folder_path / "finaldata.csv"
model_file_path = model_folder_path / "trainedmodel.pkl"


@app.command()
def model_predictions(X):
    model = load(model_file_path)
    y_pred = model.predict(X)
    return y_pred


@app.command()
def dataframe_summary():
    data = pd.read_csv(data_file_path)
    # calculate statistics
    stats_df = data.describe().loc[["mean", "50%", "std"]]
    # list is not the best data structure do return, i will return a python dict
    return stats_df.to_dict()


@app.command()
def count_nan():
    data = pd.read_csv(data_file_path)
    percent_nan = data.isnull().mean()
    return percent_nan.to_dict()


@app.command()
def execution_time():
    times = []
    for file in ["ingest", "train"]:
        starttime = timeit.default_timer()
        _ = subprocess.run(["dvc", "repro", file])
        timing = timeit.default_timer() - starttime
        times.append({file: timing})
    return times


@app.command()
def outdated_packages():
    # run the command and get the output
    output = subprocess.run(["pip", "list", "--outdated"], capture_output=True)
    output = output.stdout.decode()
    # check if there are any outdated packages
    if "Package" in output:
        lines = output.strip().split("\n")[2:]
        outdated_package_list = []
        for line in lines:
            package_info = line.strip().split()
            outdated_package_list.append(
                {
                    "name": package_info[0],
                    "installed": package_info[1],
                    "latest": package_info[2],
                }
            )
    else:
        outdated_package_list = []

    return outdated_package_list


if __name__ == "__main__":
    app()
