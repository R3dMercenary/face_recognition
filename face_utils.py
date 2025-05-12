import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.yolo_model = YOLO("yolov8n-face-lindevs.pt")
        self.known_face_names = []
        self.known_face_info = {}
        self.model = self.load_model_with_tfsmlayer()
        self.load_known_faces()
        
    def detect_faces_yolo(self, image):
        """Detect faces using YOLO and return list of cropped face images"""
        results = self.yolo_model(image)
        faces = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size > 0:
                    faces.append(face_crop)
        return faces
    # ------ Model and Embedding -------- #
    def load_model_with_tfsmlayer(self):
        """Load FaceNet model using TFSMLayer"""
        model_path = os.path.abspath(os.path.join('static', 'model'))
        
        if not os.path.exists(model_path):
            logger.error(f"Model directory not found at: {model_path}")
            return None

        try:
            logger.info(f"Loading FaceNet model from: {model_path}")
            
            # Create model with TFSMLayer
            input_layer = tf.keras.layers.Input(shape=(160, 160, 3), name='input_layer', dtype=tf.float32)
            tfsm_layer = tf.keras.layers.TFSMLayer(
                model_path,
                call_endpoint='serving_default',
                name='tfsm_layer'
            )
            output_layer = tfsm_layer(input_layer)
            model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
            
            # Verify model works
            test_img = np.random.rand(160, 160, 3) * 255
            test_embedding = self.get_embedding(test_img, model)
            logger.info(f"Model test successful. Embedding shape: {test_embedding.shape}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load FaceNet model: {str(e)}")
            return None

    def get_embedding(self, image, model=None):
        """Convert image to FaceNet embedding with proper preprocessing"""
        model = model or self.model
        if model is None:
            raise RuntimeError("Model not loaded - cannot extract embeddings")
        
        try:
            # Convert to array and preprocess
            if isinstance(image, str):  # If input is file path
                img = tf.keras.utils.load_img(image, target_size=(160, 160))
                img = tf.keras.utils.img_to_array(img)
            else:  # If input is numpy array (from webcam)
                img = cv2.resize(image, (160, 160))
                if len(img.shape) == 2:  # Grayscale
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                img = img.astype('float32')
            
            # FaceNet specific preprocessing
            img = np.expand_dims(img, axis=0)
            img = (img - 127.5) / 128.0
            
            # Get embedding and normalize
            embedding = model(img)["Bottleneck_BatchNorm"].numpy()[0]
            return embedding / np.linalg.norm(embedding)
            
        except Exception as e:
            logger.error(f"Error during embedding extraction: {str(e)}")
            raise RuntimeError(f"Embedding extraction failed: {str(e)}")

    def register_new_face(self, image, name, additional_info=""):
        try:
            faces = self.detect_faces_yolo(image)
            if not faces:
                return False, "No face detected"
            embedding = self.get_embedding(faces[0])  # Only use the first face

            
            # Store in memory
            self.known_face_encodings.append(embedding)
            self.known_face_names.append(name)
            self.known_face_info[name] = {
                "additional_info": additional_info,
                "registration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            
            self.save_known_faces()
            
            return True, "Face registered successfully"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
        
    # ------ Face Recognition -------- #
    def recognize_face(self, image):
        """Recognize face(s) using FaceNet after detecting with YOLO"""
        try:
            faces = self.detect_faces_yolo(image)
            if not faces:
                return False, "No face detected", 0, None

            best_match = None
            best_confidence = 0
            best_name = "Unknown"
            best_info = None

            for face in faces:
                query_embedding = self.get_embedding(face)

                # Compare with all known embeddings
                for known_embedding, name in zip(self.known_face_encodings, self.known_face_names):
                    distance = np.linalg.norm(query_embedding - known_embedding)
                    confidence = max(0, 1 - distance) * 100

                    if distance < 0.6 and confidence > best_confidence:
                        best_match = name
                        best_confidence = confidence
                        best_info = self.known_face_info.get(name, {}).get("additional_info", "")

            if best_match:
                return True, best_match, best_confidence, best_info
            else:
                return False, "Unknown", 0, None

        except Exception as e:
            return False, f"Recognition error: {str(e)}", None, None


    def load_known_faces(self):
        """Load existing .pkl files from known_faces directory (supports multiple embeddings per person)"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_info = {}

        try:
            pkl_files = [f for f in os.listdir("known_faces") if f.endswith(".pkl")]

            for pkl_file in pkl_files:
                with open(os.path.join("known_faces", pkl_file), "rb") as f:
                    data = pickle.load(f)
                    
                    name = data['name']
                    encodings = data.get('encodings') or [data['encoding']]  # backward compatibility
                    
                    for encoding in encodings:
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(name)

                    self.known_face_info[name] = data.get('info', {})

            print(f"Loaded {len(pkl_files)} known identities from .pkl files")

        except Exception as e:
            print(f"Error loading known faces: {str(e)}")
            self.known_face_encodings = []
            self.known_face_names = []
            self.known_face_info = {}

    def save_known_faces(self):
        """Save each face as individual .pkl file"""
        os.makedirs("known_faces", exist_ok=True)
        
        try:
            # First delete old files
            for f in os.listdir("known_faces"):
                if f.endswith(".pkl"):
                    os.remove(os.path.join("known_faces", f))
                    
            # Save new files
            for name, encoding, info in zip(self.known_face_names, 
                                        self.known_face_encodings,
                                        [self.known_face_info[n] for n in self.known_face_names]):
                filename = f"known_faces/{name.lower().replace(' ', '_')}.pkl"
                with open(filename, "wb") as f:
                    pickle.dump({
                        'name': name,
                        'encoding': encoding,
                        'info': info
                    }, f)
                    
            print(f"Saved {len(self.known_face_names)} faces to .pkl files")
            
        except Exception as e:
            print(f"Error saving known faces: {str(e)}")


    pass


face_recognition_system = FaceRecognitionSystem()

__all__ = ['FaceRecognitionSystem']

face_system = FaceRecognitionSystem()