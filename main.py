from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import shutil
import os
import gdown

app = FastAPI()

MODEL_ID = "1872AyBlvYlYMi6ZHL7doXNlZziMhG6Qv"
MODEL_PATH = "best.pt"

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = None


def download_model():
    global model

    if not os.path.exists(MODEL_PATH):
        print("Downloading model with gdown...")

        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)


download_model()


@app.get("/")
def home():
    return {"status": "API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(file_path)

    detections = []

    for r in results:
        for box in r.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0]),
                "xyxy": box.xyxy[0].tolist()
            })

    return {"detections": detections}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
