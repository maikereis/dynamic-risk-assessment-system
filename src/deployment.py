import shutil
from pathlib import Path

import typer


def store_model_into_pickle(
    dataset_csv_path: Path, model_path: Path, prod_deployment_path: Path
):
    prod_deployment_path.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        ("ingesteddata.txt", "ingesteddata.txt"),
        ("latestscore.txt", "latestscore.txt"),
        ("trainedmodel.pkl", "trainedmodel.pkl"),
    ]

    for src_file, dest_file in files_to_copy:
        src = (
            (dataset_csv_path / src_file)
            if "ingesteddata" in src_file
            else (model_path / src_file)
        )
        dest = prod_deployment_path / dest_file

        shutil.copy(src, dest)


if __name__ == "__main__":
    typer.run(store_model_into_pickle)
