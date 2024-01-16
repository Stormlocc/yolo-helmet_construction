from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from flask_cors import CORS  
from YOLO_Video import video_detection
import cv2

app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
#sound_active = False
def generate_frames(path_x=''):
    sound_active = False

    yolo_output = video_detection(path_x)
    try:
        for detection_, sound_active in video_detection(path_x):
            ref, buffer = cv2.imencode('.jpg', detection_)
            frame = buffer.tobytes()
            yield (
            b' --frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            b'Content-Sound-Active: {}\r\n\r\n' + frame + b'\r\n'
            #b'Content-Sound-Active: {}\r\n\r\n'.format(sound_active).encode() + frame + b'\r\n')
            #b'Content-Sound-Active: {}\r\n\r\n'+ frame + b'\r\n'
            )
    except cv2.error as e:
        print(f"Error en OpenCV:  {e}")
    except Exception as e:
        print(f"Error en generate_frames: {e}")
        #sound_active = False
    

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

@socketio.on('disconnect')
def disconnect():
    print('Web client disconnected')

@socketio.on('request_frame')
def request_frame():
    path_x = 0  # or provide the path of the video file
    for frame in generate_frames(path_x):
        socketio.emit('update_frame', {'image': frame})

if __name__ == '__main__':
    socketio.run(app, debug=True)
