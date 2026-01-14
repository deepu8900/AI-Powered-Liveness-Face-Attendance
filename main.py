from fastapi import FastAPI, UploadFile
from .model import FaceRecognizer
from .utils import read_image
app = FastAPI()
recognizer = FaceRecognizer(model_path='models/face_model.h5', threshold=0.5)

@app.post("/register-face")
def register_face(name: str, image: UploadFile):
    img = read_image(image)
    recognizer.register_face(name, img)
    return {"status": "registered", "name": name}

@app.post("/verify-face")
def verify_face(image: UploadFile):
    img = read_image(image)
    name, score = recognizer.predict(img)
    return {"name": name, "confidence": float(score) if score else 0.0}
