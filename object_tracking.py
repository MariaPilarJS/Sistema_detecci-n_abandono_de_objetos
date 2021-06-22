from detector import detect
import numpy as np
import cv2
import copy
from tracker import Tracker
import time
from math import sqrt




def main():

    # Create opencv video capture object
    cap = cv2.VideoCapture('/home/mpjimenez/Escritorio/videos/3personas3equipajes.mp4')

    # Create Object Tracker
    tracker = Tracker(160, 30, 10, 0)

    # Variables initialization
    font = cv2.FONT_ITALIC
    numero_frames = int(cap.get(cv2.CAP_PROP_FPS))
    starting_time = time.time()
    frame_id = 0
    distancia_min_portador = 0
    parejas = []
    track_id_portador = 0
    skip_frame_count = 0
    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                    (0, 255, 255), (255, 0, 255), (255, 127, 255),
                    (127, 0, 255), (127, 0, 127)]

    pause = False
    with open("/home/mpjimenez/darknet/data/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    #funciones y definiciones
    def calcular_distancia(p1, p2):
        return int(sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[0]) ** 2))

    def calcular_distancia(p1, p2):
        return int(sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2))


    # Infinite loop to process video frames
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.resize(frame, (900, 800), interpolation=cv2.INTER_CUBIC)
        centers = []
        centers_portador = []
        class_ids = []
        dimensiones = []
        distancias_portador = []

        frame_id += 1
        print("frame actual:", frame_id)

        # Detecciones
        datos = detect(frame)
        for a in range(len(datos)):
            centers.append(np.array([datos[a][0], datos[a][1]]))
            class_ids.append(np.array(datos[a][2]))
            dimensiones.append(np.array([datos[a][3], datos[a][4]]))
            if (len(datos[a])>5):
                centers_portador.append(np.array([datos[a][5], datos[a][6]]))

        #print("centers", centers)
        #print("class_ids", class_ids)
        #print("dimensiones", dimensiones)
        #print("centers_portador", centers_portador)
        # If centroids are detected then track them
        if (len(centers) > 0):
            # Track object using Kalman Filter
            tracker.update(centers, class_ids, dimensiones, centers_portador)
            for i in range(len(tracker.tracks)):
                #print("track_id", tracker.tracks[i].track_id)
                if (len(tracker.tracks[i].trace) > 2):
                    x = tracker.tracks[i].trace[len(tracker.tracks[i].trace)-1][0][0]
                    #print("x",x)
                    y = tracker.tracks[i].trace[len(tracker.tracks[i].trace)-1][1][0]
                    #print("y", y)
                    clase = int(tracker.tracks[i].class_id)
                    label = str(classes[clase])
                    color = colors[int(tracker.tracks[i].class_id)]
                    ancho = tracker.tracks[i].dimension[0]
                    alto = tracker.tracks[i].dimension[1]
                    #Identificacion de portador y analisis de distancias
                    if int(tracker.tracks[i].class_id) == 24 or int(tracker.tracks[i].class_id) == 26 or int(tracker.tracks[i].class_id) == 28:
                        punto_equipaje = np.array([[x], [y]])
                        print("punto_equipaje", punto_equipaje)
                        x_portador = int(tracker.tracks[i].center_portador[0])
                        y_portador = int(tracker.tracks[i].center_portador[1])
                        punto_portador = np.array([[x_portador], [y_portador]])
                        print("punto_portador_detector", punto_portador)
                        #Segundo bucle sobre tracks para averiguar que track tiene esas coordenadas
                        if (len(tracker.tracks[i].trace) == 3):
                            for j in range(len(tracker.tracks)):
                                if int(tracker.tracks[j].class_id) == 0:  # Recorremos los tracks personas
                                    x_posible_portador = int(tracker.tracks[j].trace[len(tracker.tracks[j].trace) - 1][0][0])
                                    y_posible_portador = int(tracker.tracks[j].trace[len(tracker.tracks[j].trace) - 1][1][0])
                                    punto_posible_portador = np.array([[x_posible_portador], [y_posible_portador]])
                                    #print("punto_posible_portador",punto_posible_portador)
                                    distancia = int(calcular_distancia(punto_portador, punto_posible_portador))
                                    distancias_portador.append(np.array([ distancia, tracker.tracks[j].track_id]))
                            print("distancias_portador", distancias_portador)
                            # Analisis de las distancias para determinar track_id del portador
                            print("Analisis de las distancias para determinar track_id del portador")
                            if (len(distancias_portador) == 1):
                                distancia_min_portador = distancias_portador[0][1]
                            if (len(distancias_portador) > 1):
                                for k in range(len(distancias_portador) - 1):
                                    if k==0:
                                        if [distancias_portador[k][0]] < [distancias_portador[k+1][0]]:
                                            distancia_min_portador = distancias_portador[k][0]
                                        else:
                                            distancia_min_portador = distancias_portador[k+1][0]
                                    if (k > 0):
                                        if [distancia_min_portador] > [distancias_portador[k+1][0]]:
                                            distancia_min_portador = distancias_portador[k+1][0]
                            #print("distancia_min_portador", distancia_min_portador)
                            for k in range(len(distancias_portador)):
                                if [distancias_portador[k][0]] == [distancia_min_portador]:
                                    track_id_portador = distancias_portador[k][1]
                                    print ("track_id_portador", track_id_portador)
                                    parejas.append((np.array([tracker.tracks[i].track_id, track_id_portador])))

                            #print("track_id_portador",track_id_portador )
                            distancias_portador = []
                        print("parejas", parejas)
                        #En este punto ya tenemos el track_id_portador, entonces ahora cuando len(tracker.tracks[i].trace) > 3 detectar si se separan trayectorias
                        if (len(tracker.tracks[i].trace) > 3):
                            print("Comienza analisis de trayectorias")
                            for p in range(len(parejas)):
                                if parejas[i][0] == tracker.tracks[i].track_id:
                                    track_portador = parejas[i][1]
                                #print("track_portador", track_portador)
                                for j in range(len(tracker.tracks)):
                                    if tracker.tracks[j].track_id == track_portador:
                                        x_portador_final = int(tracker.tracks[j].trace[len(tracker.tracks[j].trace) - 1][0][0])
                                        y_portador_final = int(tracker.tracks[j].trace[len(tracker.tracks[j].trace) - 1][1][0])
                                        punto_portador_final = np.array([[x_portador_final], [y_portador_final]])
                                        #print("punto_portador_final", punto_portador_final)
                                        distancia_clave = calcular_distancia(punto_equipaje, punto_portador_final)
                                        if distancia_clave > 200:
                                            print("SE ESTÁ PRODUCIENDO UN ABANDONO POR PARTE DEL USUARIO", track_portador )
                            print("Enquipaje",tracker.tracks[i].track_id)
                            print("Portador",track_portador)
                            print("distancia entre ambos", distancia_clave)
                    # Cuadrado identificador
                    cv2.rectangle(frame, (int(x - ancho / 2), int(y - alto / 2)),(int(x + ancho / 2), int(y + alto / 2)), color, 2)
                    cv2.putText(frame, label + str(tracker.tracks[i].track_id),(int(x - ancho / 2), int(y + alto / 2)), font, 1, color, 2)
                    # Display the resulting tracking frame
                    elapse_time = time.time() - starting_time
                    fps = frame_id / elapse_time
                    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 1, (0, 0, 0), 3)
                    cv2.imshow('Tracking', frame)


        # Check for key strokes
        k = cv2.waitKey(50) & 0xff
        if k == 27:  # 'esc' key has been pressed, exit program.
            break
        if k == 112:  # 'p' has been pressed. this will pause/resume the code.
            pause = not pause
            if (pause is True):
                print("Code is paused. Press 'p' to resume..")
                while (pause is True):
                    # stay in this loop until
                    key = cv2.waitKey(30) & 0xff
                    if key == 112:
                        pause = False
                        print("Resume code..!!")
                        break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # execute main
    main()