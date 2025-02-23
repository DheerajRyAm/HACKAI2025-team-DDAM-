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

        return jsonify({"gesture": "detected_gesture"})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)})

@app.route()
def game_logic():
# Constants and initial variables
    emotion = ["angry", "disgust", "fear", "happy", "neutral", "sad", "suprise"]
    gesture = ["plam", "I", "fist", "fist_moved", "thumb", "index", "ok", "plam_moved", "c", "down"]
    high_score = 0
    current_score = 0
    num_correct = 0
    song_is_playing = False
    emotion_queue = deque()
    gesture_queue = deque()

    # Function to simulate AI output
    def get_ai_emotion_and_gesture():
        random_emotion = random.choice(emotion)  # Replace with AI interaction
        random_gesture = random.choice(gesture)  # Replace with AI interaction
        return random_emotion, random_gesture

    # Game loop logic, running asynchronously
    def game_loop():
        global current_score, high_score, num_correct, song_is_playing
        while game_status:
            # Pick random emotion and gesture
            random_emotion, random_gesture = get_ai_emotion_and_gesture()

            # Simulating AI values (these would be replaced with actual AI inputs)
            emotion_value_from_ai = random.choice(emotion) # demo
            gesture_value_from_ai = random.choice(gesture) # demo

            current_time = time.time()
            emotion_queue.append((random_emotion, current_time))
            gesture_queue.append((random_gesture, current_time))

            while emotion_queue and gesture_queue:
                emotion_check, emotion_time = emotion_queue[0]
                gesture_check, gesture_time = gesture_queue[0]
                if current_time - emotion_time > 3:
                    if emotion_value_from_ai == random_emotion:
                        num_correct += 1
                    else:
                        num_correct = 0

                if current_time - gesture_time > 3:
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

            # For debugging or stopping the game, you might want to print out the current score
            print(f"Current Score: {current_score}, High Score: {high_score}")

            # Sleep for some time to simulate a delay in between game iterations
            time.sleep(1)  # Adjust delay as needed for your game loop







@app.route('/start_game', methods=['POST'])
def start_game():
    global song_is_playing
    if not song_is_playing:
        song_is_playing = True
        game_thread = threading.Thread(target=game_loop)
        game_thread.start()
        return jsonify({"message": "Game started!"}), 200
    else:
        return jsonify({"message": "Game already running!"}), 400

@app.route('/stop_game', methods=['POST'])
def stop_game():
    global game_status
    game_status = False
    return jsonify({"message": "Game stopped!"}), 200

@app.route('/get_score', methods=['GET'])
def get_score():
    return jsonify({"current_score": current_score, "high_score": high_score})

@app.route('/get_emotion_and_gesture', methods=['GET'])
def get_emotion_and_gesture():
    random_emotion, random_gesture = get_ai_emotion_and_gesture()
    return jsonify({"emotion": random_emotion, "gesture": random_gesture})



if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=8000, debug=True, allow_unsafe_werkzeug=True)
