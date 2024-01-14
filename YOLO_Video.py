from ultralytics import YOLO
import cv2
import math


'''La idea es grabar un video en donde identifica objetos y muestra las etiquetas y la confianza'''
def video_detection(path_x):
    video_capture = path_x
    #Crear la webcam
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    #out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (frame_width,frame_height))

    model = YOLO('../YOLO-Weight/yolov8n.pt')

    #Donde se utiliza no es necesario utilzar, solo es para tener el ID
    classNames = ["person", "bicycle", "car", "motorcycle",
                "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
                "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
                "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
                "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
                "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
                "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
                "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
                "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" ]

    while True:
        success, img = cap.read()

        #stream true is more efficient
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2= box.xyxy[0]
                x1, y1, x2, y2 =  int(x1), int(y1), int(x2), int(y2)
                #print(x1,y1,x2,y2)
                cv2.rectangle(img, (x1,y1) ,(x2,y2), (255,0,255), 3)
                #print(box.conf[0])
                confianza = math.ceil((box.conf[0]*100)) / 100
                cls = int (box.cls[0])
                #Esto es para el ID
                class_name = classNames[cls]
                label = f'{class_name}{confianza}'
                #Buscar el tamaño de etiqueta cuadro
                t_size = cv2.getTextSize(label,0,fontScale=1,thickness=2)[0]

                c2 = x1 + t_size[0], y1-t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)    #filled
                cv2.putText(img, label, (x1,y1-2), 0 , 1, [255,255,0],thickness=1, lineType=cv2.LINE_AA) #colocar el text

        yield img

        #out.relace()
    cv2.destroyAllWindows()





















