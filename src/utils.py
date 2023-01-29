import pandas as pd
import yaml


def preprocess_data(data_file_path):
    with open("params.yaml", "r") as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
        drop_col = params["train"]["drop"]
        features = params["train"]["features"]
        target = params["train"]["target"]

    data = pd.read_csv(data_file_path)
    data = data.drop(drop_col, axis=1)

    X = data[features]

    try:
        y = data[target]
    except KeyError:
        print(f"{target} not found in data")
        return (X, None)
    return (X, y)
