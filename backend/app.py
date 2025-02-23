import cv2
import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from torchvision import transforms

import random
import time
from collections import deque
import threading

score_lock = threading.Lock()


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

        return jsonify({"gesture": detected_gesture})  # Corrected this line

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})



# Constants and initial variables
emotion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "suprise"]
gesture = ["plam", "I", "fist", "thumb", "index", "ok", "plam_moved", "c", "down"]

#song_is_playing = False
emotion_queue = []
gesture_queue = []

global game_status

global high_score
global current_score
global num_correct



# this needs a massive fix
def game_loop():
    while game_status:
        # Pick random emotion and gesture
        random_emotion = random.choice(emotion)
        random_gesture = random.choice(gesture)

        current_time = time.time()
        emotion_queue.append((random_emotion, current_time))
        gesture_queue.append((random_gesture, current_time))

        # AI values from human camera 
        emotion_value_from_ai = random.choice(emotion) # REPLACE
        gesture_value_from_ai = random.choice(gesture) # REPLACE


        while emotion_queue and gesture_queue:
            current_time = time.time()

            current_emotion = emotion_queue.pop(0)
            current_gesture = gesture_queue.pop(0)

            # checks if deltatime is in differnce of 2

            def check_if_pose_is_same():
                if emotion_value_from_ai == random_emotion:
                    num_correct += 1
                else:
                    num_correct = 0

                if gesture_value_from_ai == random_gesture:
                    num_correct += 1
                else:
                    num_correct = 0




        # Combo scoring system
        if num_correct <= 3:
            current_score += 5
        elif 3 < num_correct <= 7:
            current_score *= 5
        elif 7 < num_correct <= 10:
            current_score *= 10
        elif 10 < num_correct <= 20:
            current_score *= 25
        elif num_correct > 20:
            current_score *= 100

        if current_score > high_score:
            high_score = current_score



@app.route('/start_game', methods=['POST'])
def start_game():
    #global game_status

    #global high_score
    #global current_score
    #global num_correct
    high_score = 0
    current_score = 0
    num_correct = 0

    global combo
    combo = 5

    global current_emotion
    global current_gesture

    if not game_status:
        game_status = True
        game_thread = threading.Thread(target=game_loop)
        game_thread.start()
        return jsonify({"message": "Game started!"}), 200
    else:
        return jsonify({"message": "Game already running!"}), 400

@app.route('/stop_game', methods=['POST'])
def stop_game():
    game_status = False
    return jsonify({"message": "Game stopped!"}), 200

@app.route('/get_score_and_combo', methods=['GET'])
def get_score_and_combo():
    with score_lock:
        return jsonify({"current_score": current_score, "high_score": high_score, "combo": combo})

@app.route('/get_emotion_and_gesture', methods=['GET'])
def get_emotion_and_gesture():
    with score_lock:
        return jsonify({"current_emotion":current_emotion, "current_gesture":current_gesture})


if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)

