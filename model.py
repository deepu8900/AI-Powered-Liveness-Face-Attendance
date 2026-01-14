import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from utils import preprocess_face

class FaceRecognizer:
    def __init__(self, model_path='models/face_model.h5', threshold=0.5):
        self.model = load_model(model_path)  
        self.threshold = threshold
        self.known_embeddings = {}  

    def register_face(self, name, face_img):
        embedding = self.get_embedding(face_img)
        self.known_embeddings[name] = embedding
        return True

    def get_embedding(self, face_img):
        face = preprocess_face(face_img)
        face = np.expand_dims(face, axis=0)
        embedding = self.model.predict(face)[0]
        return embedding / np.linalg.norm(embedding)

    def compare(self, embedding):
        best_match = None
        best_score = 0
        for name, known_emb in self.known_embeddings.items():
            score = np.dot(embedding, known_emb)  
            if score > best_score:
                best_score = score
                best_match = name
        if best_score >= self.threshold:
            return best_match, best_score
        return None, best_score

    def predict(self, face_img):
        embedding = self.get_embedding(face_img)
        return self.compare(embedding)
