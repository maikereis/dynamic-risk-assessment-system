import typer
import pandas as pd
from pathlib import Path
from itertools import chain


def merge_multiple_dataframe(input_folder_path: Path, output_folder_path: Path):

    # create output folder
    Path(output_folder_path).mkdir(exist_ok=True)

    # search for csv and txt files
    paths = chain(input_folder_path.glob("*.csv"), input_folder_path.glob("*.txt"))

    parts = []

    # read each file and append in a list
    for file_path in paths:
        if file_path.suffix == ".json":
            data_part = pd.read_json(file_path)
        elif file_path.suffix == ".csv":
            data_part = pd.read_csv(file_path)
        parts.append(data_part)
        with open(output_folder_path / "ingesteddata.txt", "a") as f:
            f.write(file_path.name + "\n")

    data = pd.concat(parts)
    data = data.drop_duplicates()

    data.to_csv(output_folder_path / "finaldata.csv", index=False)


if __name__ == "__main__":
    typer.run(merge_multiple_dataframe)
