import json
import pandas as pd
from pathlib import Path
from itertools import chain

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = Path(config["input_folder_path"])
output_folder_path = Path(config["output_folder_path"])
ingested_file_path = output_folder_path / "ingesteddata.txt"
output_file_path = output_folder_path / "finaldata.csv"


def merge_multiple_dataframe():
    # Create the output folder if it doesn't exist
    output_folder_path.mkdir(exist_ok=True)

    # Search for CSV and TXT files
    file_paths = chain(
        input_folder_path.glob("*.csv"), input_folder_path.glob("*.json")
    )

    parts = []
    file_names = []
    for file_path in file_paths:
        if file_path.suffix in [".json", ".csv"]:
            data_part = (
                pd.read_json(file_path)
                if file_path.suffix == ".json"
                else pd.read_csv(file_path)
            )
            parts.append(data_part)
            file_names.append(file_path.name)

    # Write the file names to the ingested file
    with open(ingested_file_path, "w") as f:
        f.write("\n".join(file_names))

    # Concatenate all the data parts and drop duplicates
    data = pd.concat(parts).drop_duplicates()

    # Write the final data to a CSV file
    data.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    merge_multiple_dataframe()
