import json
import sys
from pathlib import Path
from src.ingestion import merge_multiple_dataframe
from src.scoring import score_model
from src.training import train_model
from src.deployment import store_model_into_pickle
from src.reporting import score_model as reporting

with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = Path(config["input_folder_path"])
ingested_folder_path = Path(config["output_folder_path"])
model_folder_path = Path(config["prod_deployment_path"])
deployment_folder_path = Path(config["prod_deployment_path"])


##################Check and read new data
# first, read ingestedfiles.txt

ingested_file_path = deployment_folder_path / "ingesteddata.txt"
with open(ingested_file_path, "r") as f:
    ingested_file_list = [line.rstrip("\n") for line in f]


available_file_paths = input_folder_path.glob("*.csv")
available_file_list = [file_path.name for file_path in available_file_paths]

# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt

if set(available_file_list) != set(ingested_file_list):
    ##################Deciding whether to proceed, part 1
    # if you found new data, you should proceed. otherwise, do end the process here
    merge_multiple_dataframe()
else:
    print("All files are updated!")
    sys.exit()


##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
score_file_path = deployment_folder_path / "latestscore.txt"
with open(score_file_path, "r") as f:
    last_score = float(f.read())


data_file_path = ingested_folder_path / "finaldata.csv"
model_file_path = model_folder_path / "trainedmodel.pkl"

new_score = score_model(model_file_path, data_file_path, None)

print(f"The model last score = {last_score:.2f} and new score = {new_score:.2f}")

if abs((new_score - last_score) / last_score) >= 0.2:
    print("The difference is greater than 20% or less than -20%\nModel Drift detect!")
    print("Retraining the model....")
    train_model()
    print("Scoring the model...")
    score_model()
    print("Storing the model..")
    store_model_into_pickle()
    print("Reporting the model.")
    reporting()
else:
    print("Nothing to do!")
