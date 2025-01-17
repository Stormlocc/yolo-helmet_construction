from ultralytics import YOLO
import cv2
import os
from playsound import playsound
import pywhatkit
import threading
import tempfile

sound_path = os.path.join(os.path.dirname(__file__), "pitido.mp3")
sound_active = False 

def mensaje_wpp(alerta):
    pywhatkit.sendwhatmsg_instantly("+51946760874", alerta, 5, True, 4)

def draw_and_label(img, box, class_name, confidence):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    color = COLORS.get(class_name, (0, 0, 255))  # Default to red if class not found
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.rectangle(img, (x1, y1), (x1 + 150, y1 - 25), color, -1, cv2.LINE_AA)
    label = f'{class_name} {confidence:.2f}'
    cv2.putText(img, label, (x1, y1 - 2), 0, 0.6, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

def video_detection(video_path):
    global sound_active 
    cap = cv2.VideoCapture(video_path)  # Update with the correct video path

    model = YOLO('best.pt')
    cont = 0
    while True:
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                confidence = box.conf[0]
                class_index = int(box.cls[0])
                class_name = None

                if 0 <= class_index < len(classNames):
                    class_name = classNames[class_index]

                if confidence > 0.8:
                    draw_and_label(img, box, class_name, confidence)
                    
                    if class_name == "Head":
                        playsound(sound_path)
                        sound_active = True

                    else:
                        sound_active = False

        '''yolo_output = video_detection(path_x)
        for detection_, sound_active in yolo_output:'''
        yield img, sound_active

    cap.release()
    cv2.destroyAllWindows()


# Define class names and colors
classNames = ["Head", "Casco"]
COLORS = {"Head": (0, 204, 255), "Casco": (222, 81, 175)}
