from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import urllib.request

app = FastAPI()

# -------------------------
# MODEL DOWNLOAD (Render safe)
# -------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1872AyBlvYlYMi6ZHL7doXNlZziMhG6Qv"
MODEL_PATH = "best.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

download_model()

model = YOLO(MODEL_PATH)

# -------------------------
# UPLOAD FOLDER
# -------------------------
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# ROUTE
# -------------------------
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
