# 🌿 Plant Disease Detection API

An AI-powered REST API that detects plant diseases from leaf images using a CNN model built with TensorFlow, served via FastAPI.

---

## 🎯 Key Highlights

- ✅ **94%+ accuracy** across 15+ plant disease categories
- ⚡ **FastAPI** REST endpoint for real-time image inference
- 🔄 Migrated model from `.h5` → `.keras` for TF 2.15 compatibility
- 📉 **35% faster** inference setup after migration
- 📦 Supports image uploads up to **10MB** via multipart form data

---

## 🧠 How It Works
User uploads image → FastAPI receives → Pillow preprocesses
→ CNN Model predicts → JSON response with disease name + confidence

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| Model | CNN (TensorFlow / Keras) |
| API | FastAPI + Uvicorn |
| Image Processing | Pillow |
| Language | Python |

---

## 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/Riyansh2409/plant-disease-detection.git
cd plant-disease-detection

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn app.main:app --reload
```

API will be live at: `http://localhost:8000`

Swagger docs at: `http://localhost:8000/docs`

---

## 📡 API Usage

### Endpoint: `POST /predict`

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf_image.jpg"
```

**Response:**
```json
{
  "disease": "Bacterial Blight",
  "confidence": 94.3
}
```

---

## 🔁 Model Migration

```bash
python convert_model.py
```

Converts `.h5` → `.keras` format for TF 2.15 compatibility. Reduces inference setup time by ~35%.

---

## 🌿 Supported Disease Categories

Model detects **15+ plant disease categories** including:
- Bacterial Blight
- Leaf Spot
- Powdery Mildew
- Early Blight
- Late Blight
- Rust
- And many more...

---

## ⚠️ Challenges & Solutions

| Challenge | Solution |
|---|---|
| TF 2.15 `.h5` incompatibility | Wrote `convert_model.py` migration script |
| Large image upload timeouts | Added 10MB file size validation at API level |
| Slow inference on first load | Model loaded at startup, cached in memory |

---

## 🚀 Future Improvements

- [ ] Support 50+ disease categories
- [ ] Deploy on HuggingFace Spaces
- [ ] Docker containerization
- [ ] Mobile-friendly frontend

---

## 📋 Requirementsfastapi==0.110.0
uvicorn==0.29.0
tensorflow==2.15.0
numpy==1.26.4
pillow==10.2.0
python-multipart==0.0.9
---

## 👨‍💻 Author

**Riyansh Jain**
B.Tech AIML — Jain University, Bengaluru
📧 riyanshjain244@gmail.com

---

⭐ If you found this useful, give it a star on GitHub!
