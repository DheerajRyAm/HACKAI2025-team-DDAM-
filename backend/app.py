import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from torchvision import transforms

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Image preprocessing for AI model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gesture Labels
gesture_labels = ["Neutral", "Jump", "Spin", "Crouch", "Step Left", "Step Right"]

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives video frame, processes it, and returns detected gesture."""
    try:
        # Read frame from request
        file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Preprocess image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = transform(img_rgb).unsqueeze(0)

        # Run AI model (PyTorch)
        with torch.no_grad():
            output = pytorch_model(tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        detected_gesture = gesture_labels[predicted_class]
        print(f"Detected Move: {detected_gesture}")

        return jsonify({"gesture": detected_gesture})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)
