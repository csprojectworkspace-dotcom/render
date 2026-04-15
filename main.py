from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import uvicorn
import shutil
import os

app = FastAPI()

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
    uvicorn.run(app, host="127.0.0.1", port=8000)