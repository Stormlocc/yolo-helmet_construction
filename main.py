from flask import Flask, render_template, Response, url_for
from flask_socketio import SocketIO
from flask_cors import CORS
from YOLO_Video import video_detection
import cv2
from twilio.rest import Client

app = Flask(__name__)
socketio = SocketIO(app)
CORS(app)
account_sid = 'AC5fa7088'  # remplacen con el ID de su cuenta de Twilio
auth_token = '1f4338cdc0ca'  # remplacen con el api-token de su cuenta de Twilio
client = Client(account_sid, auth_token)

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
            )
    except cv2.error as e:
        print(f"Error en OpenCV:  {e}")
    except Exception as e:
        print(f"Error en generate_frames: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    # Activar camara
    return Response(generate_frames(path_x=0), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/send_notification')
def send_notification():
    sound_active = True
    if sound_active:
        send_whatsapp_alert()
    return '', 200, {'Content-Sound-Active': str(sound_active)}

def send_whatsapp_alert():
    message_body = "Alerta... Póngase el casco de seguridad"
    to_phone_number = 'whatsapp:+51981505082'  # numero real cualquiera

    try:
        message = client.messages.create(
            from_='whatsapp:+14155238886',  # tu numero en tu cuenta de Twilio
            body=message_body,
            to=to_phone_number
        )
        print(f"Mensaje de WhatsApp enviado con éxito a {to_phone_number}")
    except Exception as e:
        print(f"Error al enviar el mensaje de WhatsApp: {e}")


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