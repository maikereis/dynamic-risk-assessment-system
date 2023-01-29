import shutil
import json
from pathlib import Path

import typer

with open("config.json", "r") as f:
    config = json.load(f)

ingested_folder_path = Path(config["output_folder_path"])
model_folder_path = Path(config["output_model_path"])
deployment_folder_path = Path(config["prod_deployment_path"])


def store_model_into_pickle(
    ingested_folder_path: Path = ingested_folder_path,
    model_path: Path = model_folder_path,
    prod_deployment_path: Path = deployment_folder_path,
):
    prod_deployment_path.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        ("ingesteddata.txt", "ingesteddata.txt"),
        ("latestscore.txt", "latestscore.txt"),
        ("trainedmodel.pkl", "trainedmodel.pkl"),
    ]

    for src_file, dest_file in files_to_copy:
        src = (
            (ingested_folder_path / src_file)
            if "ingesteddata" in src_file
            else (model_path / src_file)
        )
        dest = prod_deployment_path / dest_file

        shutil.copy(src, dest)


if __name__ == "__main__":
    typer.run(store_model_into_pickle)
