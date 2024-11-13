import cv2
import numpy as np
import random
from perceptron import Perceptron
from neuronalNetwork import NeuronalNetwork

# Variables globales para almacenar las coordenadas de los rectángulos
rectangulos = []
start_point = None
drawing = False

# Función que maneja el evento del clic del mouse
def draw_rectangle(event, x, y, flags, param):
    global start_point, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        start_point = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_img = img.copy()
        cv2.rectangle(temp_img, start_point, (x, y), (0, 255, 0), 1)
        cv2.imshow("Imagen", temp_img)

    elif event == cv2.EVENT_LBUTTONUP:
        end_point = (x, y)
        drawing = False
        rectangulos.append((start_point, end_point))
        cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Imagen", img)
        print(f"Rectángulo seleccionado desde {start_point} hasta {end_point}")

# Cargar la imagen
img_path = 'resources/image1.png'  # Cambia a la ruta de tu imagen
img = cv2.imread(img_path)

if img is None:
    print("Error al cargar la imagen.")
else:
    cv2.imshow("Imagen", img)
    cv2.setMouseCallback("Imagen", draw_rectangle)

    while len(rectangulos) < 2:
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.getWindowProperty("Imagen", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    num_puntos = int(input("Ingrese el número de puntos a generar: "))
    p1, p2 = [], []

    for _ in range(num_puntos):
        x = random.randint(min(rectangulos[0][0][0], rectangulos[0][1][0]), max(rectangulos[0][0][0], rectangulos[0][1][0]))
        y = random.randint(min(rectangulos[0][0][1], rectangulos[0][1][1]), max(rectangulos[0][0][1], rectangulos[0][1][1]))
        b, g, r = img[y, x]
        p1.append((y, x, int(b), int(g), int(r)))

        x = random.randint(min(rectangulos[1][0][0], rectangulos[1][1][0]), max(rectangulos[1][0][0], rectangulos[1][1][0]))
        y = random.randint(min(rectangulos[1][0][1], rectangulos[1][1][1]), max(rectangulos[1][0][1], rectangulos[1][1][1]))
        b, g, r = img[y, x]
        p2.append((y, x, int(b), int(g), int(r)))

    img_result = img.copy()
    for rect in rectangulos:
        cv2.rectangle(img_result, rect[0], rect[1], (0, 255, 0), 2)

    for punto in p1:
        y, x, b, g, r = punto
        cv2.circle(img_result, (x, y), 3, (100, 255, 255), -1)

    for punto in p2:
        y, x, b, g, r = punto
        cv2.circle(img_result, (x, y), 3, (255, 100, 100), -1)

    cv2.imshow('Imagen con puntos generados', img_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    max_y, max_x = img.shape[:2]
    for i, punto in enumerate(p1):
        y_norm = punto[0] / max_y
        x_norm = punto[1] / max_x
        b_norm = punto[2] / 255
        g_norm = punto[3] / 255
        r_norm = punto[4] / 255
        p1[i] = [y_norm, x_norm, b_norm, g_norm, r_norm]

    for i, punto in enumerate(p2):
        y_norm = punto[0] / max_y
        x_norm = punto[1] / max_x
        b_norm = punto[2] / 255
        g_norm = punto[3] / 255
        r_norm = punto[4] / 255
        p2[i] = [y_norm, x_norm, b_norm, g_norm, r_norm]

    weights = [float(input(f"Peso {i}: ")) for i in range(6)]
    training_rate = float(input("Training rate: "))
    perceptron = Perceptron(weights, training_rate)
    neuronalNetwork = NeuronalNetwork(p1, p2, perceptron)

    neuronalNetwork.train()

    print("Pesos ajustados:")
    for i, weight in enumerate(perceptron.weights):
        print(f"Peso {i + 1}: {weight}")


    # Utiliza `img_result` para dibujar la línea de decisión sobre la imagen con los puntos
    if img_result is None:
        print("Error al cargar la imagen.")
    else:
        # Extraer los primeros dos pesos
        weight_x = perceptron.weights[0]
        weight_y = perceptron.weights[1]
        
        # Calcular la pendiente de la línea usando los pesos
        if weight_y != 0:
            slope = -weight_x / weight_y  # La pendiente de la línea
        else:
            slope = float('inf')  # Línea vertical si el peso de y es cero

        # Obtener dimensiones de la imagen
        height, width = img_result.shape[:2]

        # Calcular puntos de inicio y fin para la línea
        if slope != float('inf'):
            # Para una línea oblicua
            start_point = (0, int(slope * 0))  # Punto en x = 0
            end_point = (width, int(slope * width))  # Punto en x = ancho de la imagen

            # Asegura que los puntos estén dentro de la imagen
            start_point = (0, min(max(start_point[1], 0), height - 1))
            end_point = (width - 1, min(max(end_point[1], 0), height - 1))
        else:
            # Para una línea vertical
            x_coord = int(-weight_x / weight_y)
            start_point = (x_coord, 0)
            end_point = (x_coord, height - 1)

        # Dibuja la línea en la imagen con los puntos generados
        cv2.line(img_result, start_point, end_point, (0, 0, 255), 2)  # Color rojo

        # Muestra la imagen final con los puntos y la línea de decisión
        cv2.imshow('Imagen con puntos y línea de decisión', img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Imprimir los pesos ajustados
        print("Pesos ajustados:")
        for i, weight in enumerate(perceptron.weights):
            print(f"Peso {i + 1}: {weight}")