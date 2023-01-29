import json
import subprocess
import timeit
from pathlib import Path
import pkg_resources

import pandas as pd
import typer
from joblib import load

app = typer.Typer()

with open("config.json", "r") as f:
    config = json.load(f)

data_folder_path = Path(config["output_folder_path"])
model_folder_path = Path(config["output_model_path"])

data_file_path = data_folder_path / "finaldata.csv"
model_file_path = model_folder_path / "trainedmodel.pkl"


@app.command()
def model_predictions(data_file_path: Path):
    model = load(model_file_path)
    X, _ = preprocess_data(data_file_path)
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
    output = subprocess.run(
        ["pip", "list", "--outdated", "--format", "json"], capture_output=True
    )
    # parse the json output
    packages = json.loads(output.stdout)
    # filter only outdated packages
    outdated_packages = [
        package
        for package in packages
        if package["latest_version"] != package["version"]
    ]
    # extract the needed information
    package_list = [
        {
            "name": package["name"],
            "installed": package["version"],
            "latest": package["latest_version"],
        }
        for package in outdated_packages
    ]
    return package_list


if __name__ == "__main__":
    app()
