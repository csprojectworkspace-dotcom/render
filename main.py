from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import requests

app = FastAPI()

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "best.pt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1872AyBlvYlYMi6ZHL7doXNlZziMhG6Qv"

model = None  # IMPORTANT: avoid crash on startup

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------
# DOWNLOAD MODEL SAFELY
# -------------------------
def download_model():
    global model

    try:
        if not os.path.exists(MODEL_PATH):
            print("Downloading model...")

            r = requests.get(MODEL_URL, stream=True, timeout=60)
            r.raise_for_status()

            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print("Loading YOLO model...")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully")

    except Exception as e:
        print("MODEL LOAD FAILED:", e)
        model = None

# Run at startup safely
download_model()

# -------------------------
# ROUTE
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model

    # If model failed, don't crash API
    if model is None:
        return {"error": "Model not loaded"}

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
