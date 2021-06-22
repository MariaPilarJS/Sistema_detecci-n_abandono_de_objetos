import cv2
import numpy as np
from math import sqrt

def detect(frame):
    print("Ejecutando función detect")
    # Load Yolo
    net = cv2.dnn.readNet("/home/mpjimenez/darknet/yolov3.weights", "/home/mpjimenez/darknet/cfg/yolov3.cfg")
    with open("/home/mpjimenez/darknet/data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    #Funciones y metodos
    def getcenter_x(x, w):
        center_x_recalculado = int(x + w / 2)
        return center_x_recalculado

    def getcenter_y(y, h):
        center_y_recalculado = int(y + h/2)
        return center_y_recalculado

    def calcular_distancia(p1, p2):
        return int(sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))

    #Inicializaciones
    class_ids = []
    confidences = []
    boxes = []
    datos = []
    distancias = []
    personas = []
    equipajes = []
    center_x_pareja = 0
    center_y_pareja = 0
    distancia_min = 0

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3 and (class_id == 0 or class_id == 24 or class_id == 26 or class_id == 28):
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.3)
    print("detecciones realizadas")


    #Aqui queremos clasificar todas las detecciones en personas o equipajes
    #print("inicio bloque clasificacion")
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            #Clasificacion
            if class_ids[i] == 0:
                personas.append(np.array([[getcenter_x(x, w)], [getcenter_y(y, h)], [class_ids[i]], [w], [h]]))
            if class_ids[i] == 24 or class_ids[i] == 26 or class_ids[i] == 28:
                equipajes.append(np.array([[getcenter_x(x, w)], [getcenter_y(y, h)], [class_ids[i]], [w], [h]]))
    print("Clasificación de las detecciones")
    print("equipajes", equipajes)
    print("personas", personas)

    #print ("Realizando calculo distancia minima")
    for i in range(len(equipajes)):
            punto_equipaje = np.array([[equipajes[i][0]], [equipajes[i][1]]])
            #print("punto_equipaje", punto_equipaje)
            for j in range(len(personas)):
                punto_persona = np.array([[personas[j][0]], [personas[j][1]]])
                distancia_calculada = int(calcular_distancia(punto_equipaje, punto_persona))
                distancias.append(np.array([distancia_calculada, int(equipajes[i][0]), int(equipajes[i][1]), int(personas[j][0]), int(personas[j][1])]))
            if len(distancias) == 1:
                distancia_min = distancias[0][0]
            if len(distancias) > 1:
                for k in range(len(distancias) - 1):
                    if (k == 0):
                        if [distancias[k][0]] < [distancias[k+1][0]]:
                            distancia_min = distancias[k][0]
                        else:
                            distancia_min = distancias[k+1][0]
                    if (k > 0):
                        if [distancia_min] > [distancias[k+1][0]]:
                            distancia_min = distancias[k+1][0]
            print("Calculo de distancias")
            print("distancias", distancias)
            print("distancia_min", distancia_min)
            for k in range(len(distancias)):
                if [distancias[k][0]] == [distancia_min]:
                        center_x_pareja = int(distancias[k][3])
                        center_y_pareja = int(distancias[k][4])
                        #print("center_x_pareja", center_x_pareja)
                        #print("center_y_pareja", center_y_pareja)

            datos.append(np.array([equipajes[i][0],equipajes[i][1],equipajes[i][2],equipajes[i][3],equipajes[i][4],[center_x_pareja],[center_y_pareja]]))
            distancias = []

    for p in range(len(personas)):
        datos.append(np.array([personas[p][0],personas[p][1],personas[p][2],personas[p][3],personas[p][4]]))


    print("datos", datos)
    return datos