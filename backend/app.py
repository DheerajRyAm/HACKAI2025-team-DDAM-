import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification

# =================== 1️⃣ INITIALIZE FLASK =================== #
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# =================== 2️⃣ LOAD HUGGING FACE MODEL =================== #
print("🔄 Loading Hugging Face ViT model...")
model_name = "nateraw/face-emotion-recognition"
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
vit_model = ViTForImageClassification.from_pretrained(model_name)
vit_model.eval()
print("✅ Model loaded successfully.")

# Emotion labels (from model config)
emotion_labels = vit_model.config.id2label

# =================== 3️⃣ EMOTION DETECTION FUNCTION =================== #
def detect_emotion(frame):
    """Processes an image and predicts emotion using Hugging Face ViT model."""
    try:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        image = Image.fromarray(image)  # Convert NumPy array to PIL Image

        # Preprocess the image
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Run inference
        with torch.no_grad():
            outputs = vit_model(**inputs)

        # Get predicted emotion
        predicted_class = outputs.logits.argmax(-1).item()
        detected_emotion = emotion_labels[predicted_class]

        return detected_emotion

    except Exception as e:
        print(f"❌ Error in emotion detection: {e}")
        return "error"

# =================== 4️⃣ FLASK API ENDPOINT =================== #
@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives an image from API, processes it, and returns the detected emotion."""
    try:
        file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        detected_emotion = detect_emotion(frame)
        return jsonify({"emotion": detected_emotion})

    except Exception as e:
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)})

# =================== 5️⃣ RUN FLASK SERVER =================== #
if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)
