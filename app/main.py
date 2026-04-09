from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import os

app = FastAPI()

# ✅ CORS (frontend connect ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sab allow (dev ke liye best)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Paths set
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "../model/plant_disease_model.h5")
class_path = os.path.join(BASE_DIR, "../data/class_names.json")

# ✅ Load model
model = tf.keras.models.load_model(model_path, compile=False)

# ✅ Load class names
with open(class_path) as f:
    class_names = json.load(f)


@app.get("/")
def home():
    return {"message": "Plant Disease Detection API 🚀"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # 📥 Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))

        # 🔄 Preprocess
        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        # 🤖 Prediction
        pred = model.predict(img)
        
        pred_index = int(np.argmax(pred))
        pred_class = class_names[pred_index]
        confidence = float(np.max(pred))

        return {
            "prediction": pred_class,
            "confidence": round(confidence, 3)
        }

    except Exception as e:
        return {"error": str(e)}