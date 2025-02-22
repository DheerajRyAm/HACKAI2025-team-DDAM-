from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return "WebRTC Signaling Server Running"

@socketio.on('offer')
def handle_offer(data):
    """ Broadcast offer to all clients except sender """
    emit('offer', data, broadcast=True, include_self=False)

@socketio.on('answer')
def handle_answer(data):
    """ Broadcast answer to all clients except sender """
    emit('answer', data, broadcast=True, include_self=False)

@socketio.on('ice-candidate')
def handle_ice_candidate(data):
    """ Broadcast ICE candidates """
    emit('ice-candidate', data, broadcast=True, include_self=False)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
