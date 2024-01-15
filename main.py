from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from YOLO_Video import video_detection
import cv2

app = Flask(__name__)
socketio = SocketIO(app)

def generate_frames(path_x=''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b' --frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    #Activar camara
    return Response(generate_frames(path_x=0),mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def connect():
    print('Web client connected')

@socketio.on('request_frame')
def request_frame():
    path_x = 0  # or provide the path of the video file
    for frame in generate_frames(path_x):
        socketio.emit('update_frame', {'image': frame})

if __name__ == '__main__':
    socketio.run(app, debug=True)
