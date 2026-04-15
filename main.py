from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import shutil
import os
import urllib.request

app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?export=download&id=1872AyBlvYlYMi6ZHL7doXNlZziMhG6Qv"

# download model at startup
urllib.request.urlretrieve(MODEL_URL, "best.pt")

model = YOLO("best.pt")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

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
