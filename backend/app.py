from collections import OrderedDict
import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
from flask_cors import CORS
from flask_socketio import SocketIO
from torchvision import transforms
import torch.nn as nn
from PIL import Image

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
IMG_SIZE = (64, 64) 

# Define possible emotions and gestures
emotion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "suprise"]

# Dummy AI model and label mapping (replace with your actual model)
emotion_labels = {i: label for i, label in enumerate(emotion)}

class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 7)  # 7 emotions
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Image preprocessing for the AI model
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def load_model(model_path):
    model = EmotionCNN()  # Ensure architecture matches

    # Load state_dict but rename keys
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_key = k.replace("model.", "").replace("fc.", "")  # Adjust key names
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=False)  # Allow missing/excess keys
    model.eval()
    return model

eval_model = load_model("emotion_model_epoch_30.pth")

@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Receives a video frame, processes it, and returns the detected emotion."""
    try:
        file = request.files['frame']
        frame = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image data")
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)
        tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = eval_model(tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        detected_emotion = emotion_labels.get(predicted_class, "unknown")
        
        return jsonify({"emotion": detected_emotion})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "file" not in request.files:
#         return jsonify({"error": "No file provided"}), 400
#     file = request.files["file"]
#     file_path = "temp.jpg"  # Temporary file storage
#     file.save(file_path)
#     emotion = predict_emotion(model, file_path)
#     return jsonify({"emotion": emotion})


# def load_model(model_path):
#     """Load the pre-trained model from the given path."""
#     return tf.keras.models.load_model(model_path)

# def preprocess_image(img_path, target_size=(224, 224)):
#     """Load and preprocess the image for the model."""
#     img = image.load_img(img_path, target_size=target_size)
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize the image
#     return img_array

# def predict_emotion(model, img_path, class_labels):
#     """Predict the emotion of the given image using the loaded model."""
#     img_array = preprocess_image(img_path)
#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions)
#     return class_labels[predicted_class]


# def game_loop():
#     global game_status, current_score, high_score, num_correct, current_emotion, current_gesture
#     while game_status:
#         # Select a random target challenge
#         random_emotion = random.choice(emotion)
#         random_gesture = random.choice(gesture)




#         target_emotion = random_emotion
#         target_gesture = random_gesture
#         challenge_time = time.time()

#         # Simulate AI values (replace these with actual inputs)
#         #time.sleep(1)  # Simulate delay for receiving input
#         emotion_value_from_ai = model()
#         gesture_value_from_ai = model()
#         response_time = time.time()

#         if emotion_value_from_ai == random_emotion and gesture_value_from_ai == random_gesture:
#             num_correct += 1
#         else:
#             num_correct = 0





#         # Check if response came within 2 seconds
#         #if response_time - challenge_time <= 2:




#         # Update current_score based on num_correct (combo scoring system)
#         if num_correct <= 3:
#             current_score += 5
#         elif num_correct <= 7:
#             current_score *= 5
#         elif num_correct <= 10:
#             current_score *= 10
#         elif num_correct <= 20:
#             current_score *= 25
#         else:
#             current_score *= 100

#         if current_score > high_score:
#             high_score = current_score

#         time.sleep(1)  # Wait before next challenge


# @app.route('/start_game', methods=['POST'])
# def start_game():
#     global game_status, high_score, current_score, num_correct, combo, current_emotion, current_gesture
#     # Reset game state
#     high_score = 0
#     current_score = 0
#     num_correct = 0
#     combo = 5
#     current_emotion = None
#     current_gesture = None
#     if not game_status:
#         game_status = True
#         game_thread = threading.Thread(target=game_loop)
#         game_thread.start()
#         return jsonify({"message": "Game started!"}), 200
#     else:
#         return jsonify({"message": "Game already running!"}), 400


# @app.route('/stop_game', methods=['POST'])
# def stop_game():
#     global game_status
#     game_status = False
#     return jsonify({"message": "Game stopped!"}), 200


# @app.route('/get_score_and_combo', methods=['GET'])
# def get_score_and_combo():
#     return jsonify({"current_score": current_score, "high_score": high_score, "combo": combo})


# @app.route('/get_emotion_and_gesture', methods=['GET'])
# def get_emotion_and_gesture():
#     return jsonify({"current_emotion": current_emotion, "current_gesture": current_gesture})


if __name__ == '__main__':
   socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)

