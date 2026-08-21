import os

import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI()

ARTIFACT_BUCKET = os.getenv("ARTIFACT_BUCKET")
MODEL_KEY = "artifacts/current/model.joblib"
MODEL_PATH = os.path.expanduser(os.getenv("MODEL_PATH", "~/models/model.joblib"))


def download_model() -> None:
    """Download the current model from object storage at server startup."""
    model_directory = os.path.dirname(MODEL_PATH)
    if model_directory:
        os.makedirs(model_directory, exist_ok=True)

    if ARTIFACT_BUCKET:
        region = os.getenv("AWS_DEFAULT_REGION") or os.getenv("AWS_REGION")
        client = boto3.client("s3", region_name=region)
        client.download_file(ARTIFACT_BUCKET, MODEL_KEY, MODEL_PATH)
        print("Model da duoc tai xuong tu Amazon S3.")
        return

    # This fallback makes local API smoke tests possible. Production sets the bucket.
    if os.path.exists(MODEL_PATH):
        print("ARTIFACT_BUCKET chua duoc dat; su dung model local.")
        return

    raise RuntimeError(
        "ARTIFACT_BUCKET is required unless MODEL_PATH points to an existing model"
    )


download_model()
model = joblib.load(MODEL_PATH)


class ScoreRequest(BaseModel):
    features: list[float]


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/score")
def score(req: ScoreRequest) -> dict[str, int | str]:
    if len(req.features) != 10:
        raise HTTPException(status_code=400, detail="Expected 10 features (adult income)")

    prediction = int(model.predict([req.features])[0])
    label = "thu_nhap_cao" if prediction == 1 else "thu_nhap_thap"
    return {"prediction": prediction, "label": label}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
