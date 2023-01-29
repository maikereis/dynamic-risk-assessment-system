import json
import subprocess
import timeit
from pathlib import Path

import numpy as np
import pandas as pd
import pkg_resources
import typer
import yaml
from joblib import load

app = typer.Typer()


with open("config.json", "r") as f:
    config = json.load(f)
    model_filepath = Path(config["output_model_path"]) / "trainedmodel.pkl"


@app.command()
def model_predictions(X):
    model = load(model_filepath)
    y_pred = model.predict(X)
    return y_pred


@app.command()
def dataframe_summary(dataset_csv_path: Path):

    data = pd.read_csv(dataset_csv_path / "finaldata.csv")

    # calculate statistics
    stats_df = data.describe().loc[["mean", "50%", "std"]]

    # list is not the best data structure do return, i will return a python dict
    return stats_df.to_dict()


@app.command()
def count_nan(dataset_csv_path: Path):

    data = pd.read_csv(dataset_csv_path / "finaldata.csv")

    percent_nan = data.isnull().mean()

    return percent_nan.to_list()


@app.command()
def execution_time():

    times = []
    for file in ["ingest", "train"]:
        starttime = timeit.default_timer()
        _ = subprocess.run(["dvc", "repro", "ingest", file])
        timing = timeit.default_timer() - starttime
        times.append(timing)

    print(times)
    return times


@app.command()
def outdated_packages_list():
    dists = [d for d in pkg_resources.working_set]

    packages_info_list = []

    for dist in dists:
        current_version = dist.version
        try:
            latest_version = pkg_resources.get_distribution(dist.project_name).version
        except pkg_resources.DistributionNotFound:
            latest_version = "N/A"

        packages_info_list.append(
            (
                dist.project_name,
                current_version,
                latest_version,
            )
        )

    outdated_modules = [
        (name, installed, latest)
        for name, installed, latest in packages_info_list
        if installed != latest
    ]

    return outdated_modules


if __name__ == "__main__":
    app()
