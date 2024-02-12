from ultralytics import YOLO
import cv2

# Inicializar la captura de video desde la cámara (cámara predeterminada)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("http://10.153.254.156:8080/videofeed")

# Establecer el ancho y alto deseados del marco
cap.set(3, 1280)
cap.set(4, 720)

# Cargar el modelo YOLO
model = YOLO(r'best.pt')

while True:
    # Leer un fotograma desde la cámara
    ret, frame = cap.read()

    # Realizar la predicción utilizando el modelo YOLO
    results = model.predict(frame,imgsz=640,conf=0.6)

    if len(results) != 0:  # Verifica si hay predicciones en xyxy
        for res in results:  # Itera sobre las coordenadas del cuadro delimitador
            print("¨Detect Obect")
        
        annotated_frames=results[0].plot()
        # Mostrar el marco con los cuadros delimitadores
        cv2.imshow('Marco Anotado', annotated_frames)

    # Esperar la pulsación de una tecla (ajustar el retardo si es necesario)
    key = cv2.waitKey(1)

    # Romper el bucle si se presiona la tecla 'Esc'
    if key == 27:
        break

# Liberar la cámara y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
