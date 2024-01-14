from flask import Flask, Response, jsonify, request, render_template
from YOLO_Video import video_detection
import cv2

app = Flask(__name__)

app.config['SECRET_KEY'] = 'muhammadmoin'

#Denifir la salida de la deteccion de video
def generate_frames(path_x = ''):
    yolo_output = video_detection(path_x)
    for detection_ in yolo_output:
        ref, buffer = cv2.imencode('.jpg', detection_)
        #algunas aplicacionres requeiren encoder images a bytes
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


if __name__ == '__main__':
    app.run(debug=True)