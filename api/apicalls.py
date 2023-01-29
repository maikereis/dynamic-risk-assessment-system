import json
import requests
from pathlib import Path

DEBUG = True

if DEBUG:
    URL = "http://127.0.0.1:8080"
else:
    URL = "https://udacity-maike-app.herokuapp.com"

with open("config.json", "r") as f:
    config = json.load(f)

responses_folder_path = Path(config["output_model_path"])
responses_file_path = responses_folder_path / "apireturns.txt"

# Specify a URL that resolves to your workspace
header = {"Content-Type": "application/json"}
data = {"data_file_path": "data/testdata/testdata.csv"}

# Call each API endpoint and store the responses
response1 = requests.post(
    "http://127.0.0.1:8080/prediction", headers=header, params=data
)
response2 = requests.post("http://127.0.0.1:8080/scoring")
response3 = requests.post("http://127.0.0.1:8080/summarystats")
response4 = requests.post("http://127.0.0.1:8080/diagnostics")

responses = []

responses.append(response1.text)
responses.append(response2.text)
responses.append(response3.text)
responses.append(response4.text)

for response in responses:
    with open(responses_file_path, "w") as f:
        f.write(response + "\n")
