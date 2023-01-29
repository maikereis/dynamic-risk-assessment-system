import secrets
from pathlib import Path

from fastapi import FastAPI

from src.diagnostics import count_nan, dataframe_summary, model_predictions
from src.scoring import score_model
from src.utils import preprocess_data

secret_key = secrets.token_hex(32)
app = FastAPI(secret_key=secret_key)


@app.get("/")
def read_root():
    return {"CashBack API": "Hello World!"}


@app.post("/post")
def post_post(q):
    print("aaa")
    return None


#######################Prediction Endpoint
@app.post("/prediction")
def predict(data_file_path: Path):
    X, _ = preprocess_data(data_file_path)
    y_pred = model_predictions(X)
    return {"model predicted": y_pred.tolist()}


@app.post("/scoring")
def stats():
    return {"model last f1-score": score_model()}


@app.post("/summarystats")
def stats():
    return {"summaries": dataframe_summary()}


@app.post("/diagnostics")
def stats():
    return {"nan count": count_nan()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", reload=True, port=8080)
